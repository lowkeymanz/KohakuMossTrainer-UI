"""导出服务(M3):run checkpoint -> LoRA/合并模型文件,export 任务化。

- 引擎导出契约(实测):`kogine run scripts/export/export.py --set WEIGHTS=<基模>
  --set ADAPTER=<ckpt>/trainable.safetensors --set OUT=<目标> --set FORMAT=<格式>`;
  运行脚本优先用已安装 bundle 路径(M2 实测 bundle 与仓库 scripts/ 快照可能不同版),
  探测不到再回退仓库相对路径;
- FORMAT 选项运行时探测、不硬编码引擎清单:
  1) 束脚本 FORMAT 行注释解析出基础值(comfy/lycoris/merged/diffusers);
  2) 仓库 src/kohakumoss/adapters/export/cli.py 的 ALIASES 键(如 hf)并入
     (engine_env 把 src/ 前置 PYTHONPATH,运行时生效的正是这份代码);
  3) 两路都失败才降级为文档 5 值并标 probed=False;
- 导出与训练/缓存/采样/网格按 GPU 独占语义互斥(task_manager.GPU_EXCLUSIVE_KINDS);
- 进 argv 的路径一律转 posix 正斜杠(Windows 实测:反斜杠路径中的 \\r \\t 等
  会被引擎写配置时的字面量转义破坏);
- 所有公开方法返回结构化 dict,不抛异常。
"""

from __future__ import annotations

import re
from pathlib import Path

try:
    from . import run_service
    from .task_manager import gpu_busy
except ImportError:  # pragma: no cover
    import run_service
    from task_manager import gpu_busy

EXPORT_SCRIPT_REL = "scripts/export/export.py"

# 探测失败时的降级清单(需求文档 §4.1 的 5 值)
FALLBACK_FORMATS = ("comfy", "lycoris", "hf", "merged", "diffusers")

# 面向普通训练用户的格式说明(UI 文案层;值清单仍运行时探测,
# 探测出的未知值原样透传 label=value,不丢选项)。
FORMAT_LABELS = {
    "comfy": "ComfyUI LoRA(-comfy.safetensors,拖入 ComfyUI 即用)",
    "lycoris": "LyCORIS LoRA(kohya_ss 等生态通用)",
    "hf": "HF/PEFT LoRA(diffusers 生态通用)",
    "merged": "合并模型单文件(权重已并入基模)",
    "diffusers": "合并模型 diffusers 目录(权重已并入基模)",
}

# 束脚本 FORMAT 行:`FORMAT = "comfy"  # "comfy" | "lycoris" | ...`
_FORMAT_LINE_RE = re.compile(r'^FORMAT\s*=\s*"[^"]*"\s*#\s*(.+)$', re.M)
_QUOTED_RE = re.compile(r'"([A-Za-z_][A-Za-z0-9_]*)"')
# 仓库 cli.py 的别名行:`ALIASES = {"hf": (...), "comfy": (...)}`
_ALIASES_RE = re.compile(r"^ALIASES\s*=\s*\{(.*)\}", re.M)
_ALIAS_KEY_RE = re.compile(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:')


def _posix(p: str | Path) -> str:
    return Path(str(p)).as_posix()


class ExportService:
    """export_checkpoint 的 UI 侧封装;app.py 路由与冒烟脚本共用。"""

    def __init__(self, adapter, task_manager) -> None:
        self.adapter = adapter
        self.tasks = task_manager

    # ---------- 格式选项运行时探测 ----------

    def _export_script_path(self) -> tuple[str, str | None]:
        """运行脚本路径(bundle 优先)+ 其可读文本路径(探测 FORMAT 用)。"""
        bundle = self.adapter.find_bundle_script("export/export.py")
        if bundle is not None:
            return _posix(bundle), str(bundle)
        repo = self.adapter.repo_root / EXPORT_SCRIPT_REL
        return EXPORT_SCRIPT_REL, str(repo) if repo.is_file() else None

    def formats(self) -> dict:
        """导出格式选项列表 [{value,label}];probed=是否来自运行时解析。"""
        values: list[str] = []
        probed = False
        _, script_text_path = self._export_script_path()
        if script_text_path:
            try:
                text = Path(script_text_path).read_text(encoding="utf-8",
                                                        errors="replace")
                m = _FORMAT_LINE_RE.search(text)
                if m:
                    values = _QUOTED_RE.findall(m.group(1))
                    probed = bool(values)
            except OSError:
                pass
        # ALIASES 键(bundle 脚本注释常缺 hf,但实际运行的 cli.py 接受别名)
        cli_py = self.adapter.repo_root / "src" / "kohakumoss" / \
            "adapters" / "export" / "cli.py"
        try:
            am = _ALIASES_RE.search(cli_py.read_text(encoding="utf-8",
                                                     errors="replace"))
            if am:
                for key in _ALIAS_KEY_RE.findall(am.group(1)):
                    if key not in values:
                        values.append(key)
                probed = probed or bool(values)
        except OSError:
            pass
        if not values:
            values = list(FALLBACK_FORMATS)
            probed = False
        return {"ok": True, "probed": probed, "error": None,
                "formats": [{"value": v, "label": FORMAT_LABELS.get(v, v)}
                            for v in values]}

    # ---------- 导出任务 ----------

    def start(self, body: dict) -> dict:
        """{run, checkpoint, format, multiplier?, save_dtype?, style?} -> export 任务。

        OUT 命名对齐引擎自动导出契约(src exports.py:<run>-<格式>.safetensors,
        diffusers 为目录),落在该 checkpoint 的 export/ 子目录内。
        """
        body = body or {}
        exe = self.adapter.find_executable("kogine")
        if exe is None:
            return {"ok": False, "engine_missing": True,
                    "error": "未找到训练引擎,请先到「环境与模型」页检查安装"}
        run = str(body.get("run", "")).strip()
        if not run:
            return {"ok": False, "error": "run 名称不能为空"}
        run_dir = run_service.resolve_run_dir(self.adapter.repo_root, run)
        if not run_dir.is_dir():
            return {"ok": False, "error": f"run 不存在:runs/moss/{run}"}
        ckpt = run_service.find_checkpoint_dir(
            run_dir, body.get("checkpoint"))
        if ckpt is None:
            return {"ok": False, "error":
                    f"checkpoint 不存在:{body.get('checkpoint')}(请从列表选择)"}
        trainable = ckpt / "trainable.safetensors"
        if not trainable.is_file():
            return {"ok": False, "error":
                    f"{ckpt.name} 缺少 trainable.safetensors,无法导出"}
        cfg = run_service.read_run_config(run_dir) or {}
        weights = str(cfg.get("WEIGHTS") or "").strip()
        if not weights:
            return {"ok": False, "error":
                    "run 的 config.json 中没有 WEIGHTS(基模路径),无法导出"}
        fmt = str(body.get("format", "")).strip()
        allowed = [f["value"] for f in self.formats().get("formats", [])]
        if fmt not in allowed:
            return {"ok": False, "error":
                    f"导出格式不合法:{fmt or '(空)'}(可选 {', '.join(allowed)})"}
        overrides = {"WEIGHTS": _posix(weights), "ADAPTER": _posix(trainable),
                     "FORMAT": fmt}
        # 命名对齐引擎自动导出:diffusers 是目录,其余单文件带扩展名
        suffix = "" if fmt == "diffusers" else ".safetensors"
        out = ckpt / "export" / f"{run}-{fmt}{suffix}"
        overrides["OUT"] = _posix(out)
        mult = body.get("multiplier")
        if mult not in (None, ""):
            try:
                overrides["MULTIPLIER"] = repr(float(mult))
            except (TypeError, ValueError):
                return {"ok": False, "error": f"multiplier 不是数字:{mult}"}
        for key, knob in (("save_dtype", "SAVE_DTYPE"), ("style", "STYLE")):
            val = str(body.get(key) or "").strip()
            if val:
                if not re.fullmatch(r"[A-Za-z0-9_.-]+", val):
                    return {"ok": False, "error": f"{key} 含非法字符:{val}"}
                overrides[knob] = val
        busy = gpu_busy(self.tasks.list_tasks())
        if busy is not None:
            return {"ok": False, "error":
                    f"已有 GPU 任务在运行({busy['kind']},id={busy['id'][:8]}),"
                    "请等待完成或先停止", "task": busy}
        script, _ = self._export_script_path()
        argv = [str(exe), "run", script]
        for k, v in overrides.items():
            argv += ["--set", f"{k}={v}"]
        task = self.tasks.start("export", argv, config_path=None,
                                run_dir=str(run_dir), overrides=overrides)
        return {"ok": True, "task": task, "run": run,
                "checkpoint": ckpt.name, "format": fmt,
                "out": overrides["OUT"]}
