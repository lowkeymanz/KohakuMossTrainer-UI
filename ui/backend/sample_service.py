"""采样与对比网格服务(M3):出图任务 + 产物图库。

- 采样链路完全复用引擎 CLI 契约:`kohakumoss-config new sample` 生成配置
  (生成期已 _check 校验)-> `kogine run <sample.py> --config <cfg>`;
  运行脚本路径以配置 docstring 打印的 bundle 路径为准(M2 实测 bundle 与
  仓库 scripts/ 快照可能不同版),探测不到再逐级回退;
- 对比网格无 starter 生成(compare_grid 不是 new sample 的驱动脚本),直接
  `kogine run <compare_grid.py> --set ...`;列表旋钮按引擎实测用 JSON 方括号
  语法传值(PROMPTS 由 prompt_list 解析,CHECKPOINTS 为列表默认值);
- 图库落点 ui/state/samples/<名>/ 与 ui/state/grids/<名>/(state/ 已在
  ui/.gitignore 内,不污染上游仓库);
- 进 argv / 配置字面量的路径一律转 posix(Windows 实测:new sample 的
  ADAPTERS/OUT 原样写入配置,反斜杠会被 Python 转义序列破坏);
- 采样/网格与训练/缓存/导出按 GPU 独占语义互斥;
- Windows 上自动注入实测验证过的防崩组合(基座解压 + 流式不锁页,见
  _WIN_SAMPLING_WORKAROUND),绕开引擎侧跨文件 mmap 崩溃;
- 所有公开方法返回结构化 dict,不抛异常。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

try:
    from . import run_service
    from .task_manager import gpu_busy
    from .train_service import resolve_scripts
except ImportError:  # pragma: no cover
    import run_service
    from task_manager import gpu_busy
    from train_service import resolve_scripts

# 图库根目录(ui/state 下,与数据集库同层)
GALLERY_ROOT = Path(__file__).resolve().parents[1] / "state" / "samples"
GRID_ROOT = Path(__file__).resolve().parents[1] / "state" / "grids"

SAMPLE_SCRIPT_REL = "scripts/sample/sample.py"
GRID_SCRIPT_REL = "scripts/sample/compare_grid.py"

# Windows 引擎侧已知崩溃(2026-10-25 官方原生路径实测定位):DiT 以流式 profile
# 加载时其 safetensors mmap 张量整程存活,引擎加载文本编码器前的格式探测
# (torch framework 读文件)触发 access violation 硬崩(3221225477)。
# 已验证规避组合(8GB 显存 Windows 跑通出图):BASE_PRECISION=compute 将 int8
# 基座解压为稠密权重(文本编码器加载前原文件映射已释放)+ MEMORY_OVERRIDES
# pinned:False(解压后的流式缓冲约 9.5GB 不锁页,依赖系统虚拟内存)。
# attention_varlen 的 triton 降级走引擎磁盘失败档案(~/.cache/kohakumoss_kernels/
# failures/backends.json),不可用 MEMORY_OVERRIDES.kernels/KERNELS 硬钉扎:钉扎
# 无回退,cpu 侧绑定会因 tilelang 不支持 cpu 直接报错(训练侧实测)。
# 旧方案(MEMORY_PROFILE 置空 + TEXT_DEVICE=cpu)未经验证且 TEXT_DEVICE 不影响
# 崩溃点(探测发生在放置之前),已弃用。Linux 无此崩溃,保持引擎原生行为。
_WIN_SAMPLING_WORKAROUND = sys.platform == "win32"
_WIN_SAMPLING_SETS = {"BASE_PRECISION": "compute",
                      "MEMORY_OVERRIDES": "{'pinned': False}"}

IMAGE_EXTS = run_service.IMAGE_EXTS

# 产物名白名单:与 config_service 的配置名规则一致(字母/数字开头,可含 . _ -)
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def _posix(p: str | Path) -> str:
    return Path(str(p)).as_posix()


def _clean_prompts(value) -> list[str]:
    if isinstance(value, str):
        value = [value]
    return [str(v).strip() for v in (value or []) if str(v).strip()]


class SampleService:
    """采样/网格任务与图库的 UI 侧封装;app.py 路由与冒烟脚本共用。"""

    def __init__(self, adapter, task_manager, config_service) -> None:
        self.adapter = adapter
        self.tasks = task_manager
        self.configs = config_service

    # ---------- 内部 ----------

    def _kogine(self):
        return self.adapter.find_executable("kogine")

    def _gpu_busy_result(self) -> dict | None:
        busy = gpu_busy(self.tasks.list_tasks())
        if busy is None:
            return None
        return {"ok": False, "error":
                f"已有 GPU 任务在运行({busy['kind']},id={busy['id'][:8]}),"
                "请等待完成或先停止", "task": busy}

    def _resolve_run_ckpt(self, body: dict) -> tuple[dict, Path, Path, dict] | dict:
        """公共前置:run + checkpoint + 基模路径;失败返回错误 dict。"""
        run = str(body.get("run", "")).strip()
        if not run:
            return {"ok": False, "error": "run 名称不能为空"}
        run_dir = run_service.resolve_run_dir(self.adapter.repo_root, run)
        if not run_dir.is_dir():
            return {"ok": False, "error": f"run 不存在:runs/moss/{run}"}
        cfg = run_service.read_run_config(run_dir) or {}
        weights = str(cfg.get("WEIGHTS") or "").strip()
        if not weights:
            return {"ok": False, "error":
                    "run 的 config.json 中没有 WEIGHTS(基模路径),无法采样"}
        ckpt = run_service.find_checkpoint_dir(run_dir, body.get("checkpoint"))
        if ckpt is None:
            return {"ok": False, "error":
                    f"该 run 下没有可用 checkpoint:{body.get('checkpoint') or '(最新)'}"}
        return {"ok": True, "run": run, "run_dir": run_dir, "ckpt": ckpt,
                "weights": weights, "config": cfg}

    # ---------- 采样任务 ----------

    def start(self, body: dict) -> dict:
        """{run, checkpoint?, prompts?, vram, precision?, name?, multiplier?,
        steps?} -> new sample 生成配置 -> sample.py 出图任务。

        配置名默认 <run>-s<步>-sample,force 覆盖(UI 自管的生成物)。
        """
        body = body or {}
        exe = self._kogine()
        if exe is None:
            return {"ok": False, "engine_missing": True,
                    "error": "未找到训练引擎,请先到「环境与模型」页检查安装"}
        try:
            vram = float(body.get("vram"))
        except (TypeError, ValueError):
            return {"ok": False, "error": "必须给出显存档位 vram(GiB 数字)"}
        if vram <= 0:
            return {"ok": False, "error": f"vram 必须是正数 GiB:{vram}"}
        resolved = self._resolve_run_ckpt(body)
        if not resolved.get("ok"):
            return resolved
        run, ckpt = resolved["run"], resolved["ckpt"]
        trainable = ckpt / "trainable.safetensors"
        if not trainable.is_file():
            return {"ok": False, "error":
                    f"{ckpt.name} 缺少 trainable.safetensors,无法采样"}
        prompts = _clean_prompts(body.get("prompts"))
        if not prompts:
            prompts = _clean_prompts(resolved["config"].get("SAMPLE_PROMPTS"))
        if not prompts:
            return {"ok": False, "error":
                    "没有采样提示词:请填写 prompts,或该 run 的配置未记录 SAMPLE_PROMPTS"}
        busy = self._gpu_busy_result()
        if busy is not None:
            return busy
        m = re.match(r"^step_(\d+)$", ckpt.name)
        step = m.group(1).lstrip("0") or "0" if m else ckpt.name
        name = str(body.get("name") or f"{run}-s{step}-sample").strip()
        if not _NAME_RE.fullmatch(name):
            return {"ok": False, "error":
                    "采样任务名只允许字母/数字开头,可含 . _ -,长度 1-64"}
        out = GALLERY_ROOT / name
        try:
            out.mkdir(parents=True, exist_ok=True)
        except OSError as err:
            return {"ok": False, "error": f"创建图库目录失败:{err}"}
        mult = body.get("multiplier")
        try:
            mult = 1.0 if mult in (None, "") else float(mult)
        except (TypeError, ValueError):
            return {"ok": False, "error": f"multiplier 不是数字:{mult}"}
        form = {
            "weights": resolved["weights"],
            "name": name,
            "adapter": [f"{_posix(trainable)}:{mult}"],
            "samples": _posix(out),
            "prompt": prompts,
            "vram": vram,
            "force": True,  # UI 自管生成物:同名采样配置直接覆盖重建
        }
        if body.get("precision"):
            form["precision"] = str(body["precision"])
        res = self.configs.new_sample(form)
        if not res.get("ok"):
            return {"ok": False, "error":
                    res.get("error") or "new sample 生成采样配置失败",
                    "stdout": res.get("stdout", ""), "stderr": res.get("stderr", "")}
        script = resolve_scripts(res.get("content") or "").get("sample.py")
        if script is None:
            bundle = self.adapter.find_bundle_script("sample/sample.py")
            script = _posix(bundle) if bundle else SAMPLE_SCRIPT_REL
        argv = [str(exe), "run", script, "--config", res["path"]]
        if body.get("steps") not in (None, ""):
            try:
                argv += ["--set", f"STEPS={int(body['steps'])}"]
            except (TypeError, ValueError):
                return {"ok": False, "error": f"steps 不是整数:{body['steps']}"}
        if _WIN_SAMPLING_WORKAROUND:
            # 实测防崩组合;配置生成器按显存档位选的 profile 保持不动
            # (compute 解压 + 不锁页在流式 profile 下验证通过)。
            for k, v in _WIN_SAMPLING_SETS.items():
                argv += ["--set", f"{k}={v}"]
        task = self.tasks.start("sample", argv, config_path=res["path"],
                                run_dir=str(out),
                                overrides={"PROMPTS": prompts, "ADAPTER": form["adapter"][0]})
        return {"ok": True, "task": task, "name": name, "out": _posix(out),
                "config": res["path"], "prompts": prompts}

    # ---------- 对比网格任务 ----------

    def start_grid(self, body: dict) -> dict:
        """{run, checkpoints(列表), prompts?, vram?, name?, steps?, cfg?,
        size?, precision?, memory_profile?} -> compare_grid.py --set 任务。"""
        body = body or {}
        exe = self._kogine()
        if exe is None:
            return {"ok": False, "engine_missing": True,
                    "error": "未找到训练引擎,请先到「环境与模型」页检查安装"}
        refs = body.get("checkpoints")
        if isinstance(refs, (str, int)):
            refs = [refs]
        refs = [str(r).strip() for r in (refs or []) if str(r).strip()]
        if not refs:
            return {"ok": False, "error": "至少选择一个参与对比的 checkpoint"}
        run = str(body.get("run", "")).strip()
        if not run:
            return {"ok": False, "error": "run 名称不能为空"}
        run_dir = run_service.resolve_run_dir(self.adapter.repo_root, run)
        if not run_dir.is_dir():
            return {"ok": False, "error": f"run 不存在:runs/moss/{run}"}
        cfg = run_service.read_run_config(run_dir) or {}
        weights = str(cfg.get("WEIGHTS") or "").strip()
        if not weights:
            return {"ok": False, "error":
                    "run 的 config.json 中没有 WEIGHTS(基模路径),无法生成网格"}
        ckpt_paths: list[str] = []
        for ref in refs:
            ckpt = run_service.find_checkpoint_dir(run_dir, ref)
            if ckpt is None:
                return {"ok": False, "error": f"checkpoint 不存在:{ref}"}
            trainable = ckpt / "trainable.safetensors"
            if not trainable.is_file():
                return {"ok": False, "error":
                        f"{ckpt.name} 缺少 trainable.safetensors,无法参与对比"}
            ckpt_paths.append(_posix(trainable))
        prompts = _clean_prompts(body.get("prompts"))
        if not prompts:
            prompts = _clean_prompts(cfg.get("SAMPLE_PROMPTS"))
        if not prompts:
            return {"ok": False, "error":
                    "没有对比提示词:请填写 prompts,或该 run 的配置未记录 SAMPLE_PROMPTS"}
        busy = self._gpu_busy_result()
        if busy is not None:
            return busy
        name = str(body.get("name") or f"{run}-grid").strip()
        if not _NAME_RE.fullmatch(name):
            return {"ok": False, "error":
                    "网格任务名只允许字母/数字开头,可含 . _ -,长度 1-64"}
        out = GRID_ROOT / name
        try:
            out.mkdir(parents=True, exist_ok=True)
        except OSError as err:
            return {"ok": False, "error": f"创建网格目录失败:{err}"}
        overrides = {"WEIGHTS": _posix(weights), "OUT": _posix(out),
                     # 引擎实测:列表旋钮用 JSON 方括号语法传值
                     "CHECKPOINTS": json.dumps(ckpt_paths),
                     "PROMPTS": json.dumps(prompts)}
        if body.get("steps") not in (None, ""):
            overrides["STEPS"] = str(int(body["steps"]))
        if body.get("cfg") not in (None, ""):
            overrides["CFG"] = str(float(body["cfg"]))
        if body.get("size"):
            overrides["SIZES"] = json.dumps([str(body["size"])])
        if body.get("precision"):
            overrides["PRECISION"] = str(body["precision"])
        if body.get("memory_profile"):
            overrides["MEMORY_PROFILE"] = str(body["memory_profile"])
        if _WIN_SAMPLING_WORKAROUND:
            # compare_grid 默认 MEMORY_PROFILE=""(resident):compute 会把 int8
            # 基座解压成约 9.5GB 稠密权重,resident 放置在小显存卡上必 OOM,
            # 故未显式指定时一并按实测验证过的流式档(minimal)排队。
            overrides.setdefault("MEMORY_PROFILE", "minimal")
            for k, v in _WIN_SAMPLING_SETS.items():
                overrides.setdefault(k, v)
        bundle = self.adapter.find_bundle_script("sample/compare_grid.py")
        script = _posix(bundle) if bundle else GRID_SCRIPT_REL
        argv = [str(exe), "run", script]
        for k, v in overrides.items():
            argv += ["--set", f"{k}={v}"]
        task = self.tasks.start("grid", argv, config_path=None,
                                run_dir=str(out), overrides=overrides)
        return {"ok": True, "task": task, "name": name, "out": _posix(out),
                "checkpoints": ckpt_paths, "prompts": prompts}

    # ---------- 图库 ----------

    @staticmethod
    def _group(kind: str, d: Path) -> dict:
        images: list[dict] = []
        metadata: dict[str, dict] = {}
        try:
            entries = sorted(d.iterdir(), key=lambda p: p.name.lower())
        except OSError:
            entries = []
        for p in entries:
            if not p.is_file():
                continue
            if p.suffix.lower() in IMAGE_EXTS:
                st = p.stat()
                images.append({"name": p.name, "path": _posix(p),
                               "size": st.st_size, "mtime": int(st.st_mtime)})
            elif p.suffix.lower() == ".json" and p.stat().st_size < 2_000_000:
                try:
                    metadata[p.name] = json.loads(p.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    pass
        try:
            mtime = int(d.stat().st_mtime)
        except OSError:
            mtime = 0
        return {"kind": kind, "name": d.name, "out": _posix(d),
                "mtime": mtime, "images": images, "metadata": metadata}

    def list_gallery(self) -> dict:
        """采样图库 + 网格结果:按修改时间倒序;空目录/不存在均正常返回空列表。"""
        groups: list[dict] = []
        for kind, root in (("sample", GALLERY_ROOT), ("grid", GRID_ROOT)):
            try:
                dirs = sorted((p for p in root.iterdir() if p.is_dir()),
                              key=lambda p: p.name.lower()) if root.is_dir() else []
            except OSError:
                dirs = []
            for d in dirs:
                groups.append(self._group(kind, d))
        groups.sort(key=lambda g: g["mtime"], reverse=True)
        return {"ok": True, "groups": groups, "error": None}

    def resolve_gallery_file(self, raw: str) -> dict:
        """把请求路径解析为图库内图片(供 GET /api/gallery-files 内联)。

        白名单 = ui/state/samples/ 与 ui/state/grids/ 两棵子树:相对路径按各自
        根解析,绝对路径也允许,但 resolve 后必须仍在白名单内;只放行图片扩展名。
        """
        raw = (raw or "").strip()
        if not raw:
            return {"ok": False, "error": "path 不能为空"}
        p = Path(raw)
        candidates = []
        for base in (GALLERY_ROOT, GRID_ROOT):
            try:
                base_r = base.resolve()
            except OSError:
                continue
            if not p.is_absolute():
                candidates.append((base_r / p))
            candidates.append(p)
        for cand in candidates:
            try:
                rp = cand.resolve()
            except OSError:
                continue
            for base in (GALLERY_ROOT, GRID_ROOT):
                try:
                    if rp.is_relative_to(base.resolve()):
                        if rp.suffix.lower() not in IMAGE_EXTS:
                            return {"ok": False, "error":
                                    "仅允许图片文件(png/jpg/jpeg/webp/bmp)"}
                        if not rp.is_file():
                            return {"ok": False, "not_found": True,
                                    "error": f"文件不存在:{raw}"}
                        return {"ok": True, "path": rp}
                except OSError:
                    continue
        return {"ok": False, "forbidden": True,
                "error": "图片路径不在图库范围内,已拒绝"}
