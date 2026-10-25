"""训练闭环服务(M2):启动/停止/继续/状态,数据检查与缓存任务,产物快照。

- 引擎交互全部经 subprocess 参数数组(kogine run ...),不 import 引擎模块;
- 任务生命周期复用 TaskManager(train/cache_* 长任务,stdout 落盘,重启恢复);
- run 目录在任务启动时按配置里的 RUN_NAME 预先定位(引擎产物契约);
- 同一时刻只允许一个 GPU 独占任务(train/cache),采样导出留 M3;
- 启动训练时若离线缓存产物缺失,自动排队 缓存文本 -> 缓存图像 -> 训练
  (TaskManager after= 依赖链),用户无需手动先跑缓存;
- Windows 上启用训练中预览时自动注入实测验证过的防崩组合(见
  _WIN_PREVIEW_OVERRIDES 注释),Linux 保持引擎原生行为;
- 所有公开方法返回结构化 dict,不抛异常(路由层再转 HTTP 码)。
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

try:
    from . import run_service
    from .defaults_service import DefaultsService, load_overrides
    from .metrics_reader import RunWatcher
    from .task_manager import GPU_EXCLUSIVE_KINDS, RUNNING, gpu_busy
except ImportError:  # pragma: no cover
    import run_service
    from defaults_service import DefaultsService, load_overrides
    from metrics_reader import RunWatcher
    from task_manager import GPU_EXCLUSIVE_KINDS, RUNNING, gpu_busy

TRAIN_SCRIPT = "scripts/train/moss.py"
INSPECT_SCRIPT = "scripts/data/inspect_data.py"
CACHE_SCRIPTS = {"cache_text": "scripts/data/cache_text.py",
                 "cache_latents": "scripts/data/cache_latents.py"}

# 离线缓存产物契约(实测):<数据集目录>/cache/text.db 与 latents.db;
# 任一数据集缺 text.db -> 需跑 cache_text,缺 latents.db -> 需跑 cache_latents。
CACHE_ARTIFACTS = {"cache_text": "text.db", "cache_latents": "latents.db"}
# 缓存模式旋钮:值为 "offline" 时才依赖产物(引擎默认 offline);"off"/"online" 不检查。
CACHE_MODE_KNOBS = {"cache_text": "TEXT_CACHE", "cache_latents": "LATENT_CACHE"}

# Windows 引擎侧崩溃(2026-10-25 官方原生路径实测定位):DiT 以流式 profile 加载时
# 其 safetensors mmap 张量整程存活,引擎加载文本编码器前的格式探测(torch framework
# 读文件)触发访问违例硬崩(3221225477);训练中开预览(SAMPLE_EVERY>0)必经此路径。
# 已验证规避组合(8GB 显存 Windows 跑通出图):BASE_PRECISION=compute 将 int8 基座
# 解压为稠密权重(文本编码器加载前原文件映射已释放)+ MEMORY_OVERRIDES pinned:False
# (解压后的流式缓冲约 9.5GB 不锁页,依赖系统虚拟内存;锁页会在低物理内存机器上失败)。
# 注意:不要用 KERNELS 把 attention_varlen 硬钉到 tilelang 来绕崩溃——注册表对
# 钉扎后端无回退(registry.resolve "a pinned backend gets no fallback"),而训练
# 构建先在 cpu 设备绑定一次操作(tilelang 不支持 cpu),钉扎会直接炸在构建期
# (实测 NoImplementationError)。attention_varlen 的 triton 失败应走引擎自身的
# 磁盘失败档案(~/.cache/kohakumoss_kernels/failures/backends.json):档案命中时
# 该内核按优先级链自动降级(实测 05:33 轮:档案 + 本组合可正常走到内核调优)。
# 两旋钮均为引擎官方 --set 通道,不改引擎代码;Linux 无此崩溃,保持原生行为。
_WIN_PREVIEW_OVERRIDES = {"BASE_PRECISION": "compute",
                          "MEMORY_OVERRIDES": "{'pinned': False}"}

# inspect 要加载 torch + 扫描数据集,30s 常不够;给足但设上限(实测调优)
INSPECT_TIMEOUT = 300
# 状态快照返回给前端的日志尾部行数
LOG_TAIL_LINES = 300

# 引擎在生成的配置 docstring 里打印了权威运行命令(M1 实测契约)。
# 本环境 .venv 装的是 kohakumoss 新版(非 editable),仓库 scripts/ 是旧快照,
# 旧脚本声明的旋钮名(TRANSFORMER/SAVE_MERGED/FSDP_RESHARD...)会被新版校验判
# "unknown knob"(冒烟实测),故运行命令以 docstring 打印的脚本路径为准,
# 仓库相对路径仅作手写配置的回退。
_RUN_CMD_RE = re.compile(
    r"kogine run (\S*scripts/(?:train|data|sample|export)/\w+\.py) --config")


def resolve_scripts(text: str) -> dict[str, str]:
    """配置原文 docstring -> {脚本文件名: 引擎打印的路径}。"""
    out: dict[str, str] = {}
    for m in _RUN_CMD_RE.finditer(text):
        path = m.group(1)
        out[Path(path.replace("\\", "/")).name] = path
    return out


def extract_json(text: str) -> dict | None:
    """从引擎输出(可能混有日志前缀行)里提取第一个平衡的 JSON 对象。

    inspect_data 经 log.info(json.dumps(report, indent=1)) 输出:首行带日志
    前缀、后续行原文,故不能按行 parse;用花括号计数(跳过字符串字面量)
    截取第一个完整对象。找不到/不平衡返回 None。
    """
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except ValueError:
                    return None
    return None


class TrainService:
    """训练/数据任务对 UI 的门面;app.py 与 WS 轮询循环共用。"""

    def __init__(self, adapter, task_manager, config_service) -> None:
        self.adapter = adapter
        self.tasks = task_manager
        self.configs = config_service
        # task_id -> RunWatcher(WS 循环的增量读取器;快照另用临时 watcher)
        self.watchers: dict[str, RunWatcher] = {}
        self._defaults_service = None

    # ---------- 内部 ----------

    def _defaults(self) -> DefaultsService:
        """惰性 DefaultsService(override 键校验;解析失败环境自动放行)。"""
        if self._defaults_service is None:
            self._defaults_service = DefaultsService(self.adapter)
        return self._defaults_service

    def _kogine(self):
        return self.adapter.find_executable("kogine")

    def _gpu_busy(self) -> dict | None:
        """返回正在运行的 GPU 独占任务(无则 None);M3 起含采样/网格/导出。"""
        return gpu_busy(self.tasks.list_tasks())

    def _resolve_config(self, name: str) -> tuple[Path | None, str | None]:
        """配置名 -> configs/user/<name>.py 的仓库内相对路径 + 绝对路径。"""
        target, err = self.configs._safe_user_path(name)
        if err:
            return None, err
        if not target.is_file():
            return None, f"配置文件不存在:configs/user/{target.name}"
        return target, None

    @staticmethod
    def _preview_enabled(text: str, overrides: dict) -> bool:
        """训练中预览是否生效:高级设置 SAMPLE_EVERY 优先,其次配置文件,默认 0。"""
        val = overrides.get("SAMPLE_EVERY")
        if val is None:
            m = re.search(r"^SAMPLE_EVERY\s*=\s*([0-9]+)", text, re.M)
            val = m.group(1) if m else "0"
        try:
            return int(str(val).strip()) > 0
        except ValueError:
            return False

    @staticmethod
    def _missing_caches(text: str) -> list[str]:
        """按产物文件检查离线缓存缺口 -> 需补跑的缓存任务类型(依赖序)。

        DATASETS 条目的 "path" 即数据集目录;缓存模式旋钮非 "offline" 的
        类型不检查(引擎默认 offline,配置文件未显式写出时按默认处理)。
        任一数据集缺产物即需要跑对应缓存任务。
        """
        out: list[str] = []
        for kind in ("cache_text", "cache_latents"):
            knob = CACHE_MODE_KNOBS[kind]
            m = re.search(rf'^{knob}\s*=\s*"(\w+)"', text, re.M)
            if m and m.group(1) != "offline":
                continue
            for p in re.findall(r'"path":\s*"([^"]+)"', text):
                try:
                    if not (Path(p) / "cache" / CACHE_ARTIFACTS[kind]).is_file():
                        out.append(kind)
                        break
                except OSError:  # 路径异常:交给引擎报错,这里按缺失处理
                    out.append(kind)
                    break
        return out

    def _cache_argv(self, exe, config_name: str,
                    kind: str) -> tuple[list[str] | None, str | None, str | None]:
        """缓存任务参数数组构造(start_cache 与训练前置链共用)。
        返回 (argv, 数据配置相对路径, 错误)。"""
        script = CACHE_SCRIPTS.get(kind)
        if script is None:
            return None, None, f"缓存类型不合法:{kind}"
        target, err = self._resolve_config(config_name)
        if err:
            return None, None, err
        data_cfg = target.with_name(f"{target.stem}_data.py")
        if not data_cfg.is_file():
            return None, None, ("缓存任务缺少对应的数据配置(含编码器路径),未找到;"
                                "请先用配置页重新生成该配置")
        try:
            script = resolve_scripts(data_cfg.read_text(encoding="utf-8",
                                                        errors="replace")).get(
                Path(script).name, script)
        except OSError:
            pass
        rel = data_cfg.relative_to(self.adapter.repo_root).as_posix()
        return [str(exe), "run", script, "--config", rel], rel, None

    def watcher_for(self, task: dict) -> RunWatcher:
        """WS 循环用:按任务持有增量 watcher(run_dir 变化自动重置)。"""
        w = self.watchers.get(task["id"])
        if w is None:
            w = RunWatcher()
            self.watchers[task["id"]] = w
        w.attach(task.get("run_dir"))
        return w

    # ---------- 训练任务 ----------

    def start(self, config_name: str, overrides: dict | None = None,
              resume: bool = False) -> dict:
        """启动训练:kogine run scripts/train/moss.py --config <cfg> [--set K=V ...]。

        run 目录已存在(有 metrics.jsonl 或 checkpoints)时:
        - resume=False -> 返回 run_exists=True(409 语义),提示这是续训场景;
        - resume=True  -> 追加 --set RESUME=latest 从最新 checkpoint 继续。
        高级参数(deferred_overrides)在这里经 --set 生效(M1 约定)。
        """
        exe = self._kogine()
        if exe is None:
            return {"ok": False, "engine_missing": True,
                    "error": "未找到训练引擎,请先到「环境与模型」页检查安装"}
        target, err = self._resolve_config(config_name)
        if err:
            return {"ok": False, "error": err}
        busy = self._gpu_busy()
        if busy is not None:
            return {"ok": False, "error":
                    f"已有 GPU 任务在运行({busy['kind']},id={busy['id'][:8]}),"
                    "请先停止", "task": busy}
        try:
            text = target.read_text(encoding="utf-8")
        except OSError as read_err:
            return {"ok": False, "error": f"读取配置失败:{read_err}"}
        run_name = run_service.parse_run_name(text, target.stem)
        script = resolve_scripts(text).get("moss.py", TRAIN_SCRIPT)
        run_dir = run_service.resolve_run_dir(self.adapter.repo_root, run_name)
        has_run = (run_dir / "metrics.jsonl").is_file() or \
            any((run_dir / "checkpoints").glob("step_*"))
        if has_run and not resume:
            return {"ok": False, "run_exists": True, "run_name": run_name,
                    "run_dir": str(run_dir).replace("\\", "/"),
                    "error": f"run 目录已存在产物(runs/moss/{run_name}):"
                             "如需从最新 checkpoint 继续请勾选续训(resume=true);"
                             "全新重跑会由引擎把旧产物移名保留"}
        overrides = {str(k): str(v) for k, v in (overrides or {}).items()}
        if not overrides:
            # 未显式传覆盖 -> 读配置向导持久化的高级设置(ui/state/overrides)
            overrides = load_overrides(config_name)
        bad_keys, ov_warnings = self._defaults().check_override_keys(overrides)
        if bad_keys:
            return {"ok": False,
                    "error": "高级设置包含引擎未知键:" + ", ".join(bad_keys)}
        # Windows + 预览开启:注入实测防崩组合(用户高级设置里已有的键不覆盖)
        win_injected = False
        if sys.platform == "win32" and self._preview_enabled(text, overrides):
            if all(k not in overrides for k in _WIN_PREVIEW_OVERRIDES):
                overrides.update(_WIN_PREVIEW_OVERRIDES)
                win_injected = True
        rel = target.relative_to(self.adapter.repo_root).as_posix()
        argv: list[str] = [str(exe), "run", script, "--config", rel]
        if resume:
            overrides.setdefault("RESUME", "latest")
        for k, v in overrides.items():
            argv += ["--set", f"{k}={v}"]
        # 离线缓存缺口 -> 自动排队前置任务(缓存文本 -> 缓存图像 -> 训练);
        # 先构造全部参数数组,任何一个失败都不启动半条链。
        notes: list[str] = []
        missing = self._missing_caches(text)
        plans: list[tuple[str, list[str], str]] = []
        for kind in missing:
            cargv, crel, cerr = self._cache_argv(exe, config_name, kind)
            if cerr:
                return {"ok": False, "error": cerr}
            plans.append((kind, cargv, crel))
        prep: list[dict] = []
        after: str | None = None
        for kind, cargv, crel in plans:
            t = self.tasks.start(kind, cargv, config_path=crel, run_dir=None,
                                 after=after)
            prep.append(t)
            after = t["id"]
        task = self.tasks.start("train", argv, config_path=rel,
                                run_dir=str(run_dir), overrides=overrides,
                                after=after)
        res = {"ok": True, "task": task, "run_name": run_name,
               "resumed": bool(resume), "override_count": len(overrides)}
        if prep:
            res["queued"] = True
            res["prep"] = [t["kind"] for t in prep]
            notes.append("数据缓存尚未就绪,训练前将自动完成准备,就绪后自动开始训练")
        if win_injected:
            res["preview_workaround"] = True
            notes.append("Windows 预览图经内存流式生成,首步稍慢属正常")
        if notes:
            res["note"] = ";".join(notes)
        if ov_warnings:
            res["override_warnings"] = ov_warnings
        return res

    def stop(self, task_id: str) -> dict:
        """停止 = 终止进程(CTRL_BREAK_EVENT 命中引擎 SIGTERM handler,优雅退出)。

        排队中(pending)的任务被停止时,沿依赖链向上游回溯:未开始的上游
        任务一并取消,运行中的上游任务终止(取消排队 = 取消整条链)。
        """
        task = self.tasks.get(task_id)
        if task is None:
            return {"ok": False, "error": f"任务不存在:{task_id}"}
        if task["kind"] not in GPU_EXCLUSIVE_KINDS:
            return {"ok": False, "error": f"非可终止的引擎任务:{task['kind']}"}
        upstream = task.get("after_task")
        try:
            row = self.tasks.terminate(task_id)
        except (KeyError, RuntimeError) as err:
            return {"ok": False, "error": str(err)}
        seen = {task_id}
        while upstream and upstream not in seen:
            seen.add(upstream)
            up = self.tasks.get(upstream)
            if up is None or up["status"] not in ("pending", "running"):
                break
            try:
                self.tasks.terminate(upstream)
            except (KeyError, RuntimeError):
                break  # 句柄丢失等无法终止:停在当前节点,后继自然级联
            upstream = up.get("after_task")
        return {"ok": True, "task": row}

    def resume(self, task_id: str) -> dict:
        """继续 = 以 --set RESUME=latest 重启原任务(上游无暂停信号)。"""
        task = self.tasks.get(task_id)
        if task is None:
            return {"ok": False, "error": f"任务不存在:{task_id}"}
        if task["kind"] != "train":
            return {"ok": False, "error": "仅训练任务支持继续"}
        busy = self._gpu_busy()
        if busy is not None:
            return {"ok": False, "error": "已有 GPU 任务在运行,请先停止", "task": busy}
        try:
            row = self.tasks.resume(task_id)
        except (KeyError, RuntimeError) as err:
            return {"ok": False, "error": str(err)}
        return {"ok": True, "task": row}

    # ---------- 状态与快照 ----------

    def latest_train_task(self, config_name: str | None = None) -> dict | None:
        """最近一个训练任务;给出 config_name 时只看该配置的任务(按文件名匹配)。"""
        for t in self.tasks.list_tasks():  # started_at DESC
            if t["kind"] == "train":
                if config_name is not None:
                    cp = t.get("config_path") or ""
                    if Path(cp).stem != config_name:
                        continue
                return t
        return None

    def status(self, config_name: str | None = None) -> dict:
        """所选配置最近一次训练任务 + run 产物快照(前端首屏数据源);
        未指定配置或该配置无任务时 phase=idle(面板留空)。"""
        task = self.latest_train_task(config_name)
        if task is None:
            return {"ok": True, "task": None, "phase": "idle"}
        return {"ok": True, **self.snapshot(task)}

    def snapshot(self, task: dict) -> dict:
        """一次性全量读 run 产物(临时 watcher,不影响 WS 增量状态)。"""
        run_dir = task.get("run_dir")
        w = RunWatcher()
        w.attach(run_dir)
        rows, _ = w.poll_metrics()
        samples = w.poll_samples()
        cfg = w.read_config()
        # 日志尾部:log.txt 优先,任务 stdout 兜底(进程早期崩溃时 run 目录还不存在)
        lines = w.poll_log(task.get("log_path"))
        tail = lines[-LOG_TAIL_LINES:]
        total = None
        if isinstance(cfg, dict):
            try:
                total = int(cfg.get("STEPS"))
            except (TypeError, ValueError):
                total = None
        step = max((r["step"] for r in rows if isinstance(r.get("step"), int)),
                   default=None)
        phase = self._phase(task, run_dir, rows)
        eta = None
        if phase == "running" and total and step is not None and len(rows) >= 2:
            timed = [r for r in rows if isinstance(r.get("time"), (int, float))
                     and isinstance(r.get("step"), int)]
            if len(timed) >= 2:
                dt = timed[-1]["time"] - timed[0]["time"]
                ds = timed[-1]["step"] - timed[0]["step"]
                if dt > 0 and ds > 0 and step < total:
                    eta = round((total - step) * dt / ds, 1)
        return {
            "ok": True,
            "task": task,
            "phase": phase,
            "run_dir": run_dir,
            "total_steps": total,
            "step": step,
            "eta_seconds": eta,
            "sample_every": (cfg or {}).get("SAMPLE_EVERY"),
            "validate_every": (cfg or {}).get("VALIDATE_EVERY"),
            "config": cfg,
            "metrics": rows,
            "log_tail": tail,
            "samples": samples,
        }

    @staticmethod
    def _phase(task: dict, run_dir: str | None, rows: list[dict]) -> str:
        """状态机:idle 之外的 UI 相位。进程活但 metrics 未出现 = initializing
        (内核预热可能数分钟,不能误判卡死)。"""
        status = task["status"]
        if status == RUNNING:
            return "running" if rows else "initializing"
        return {"exited_ok": "finished", "exited_fail": "failed",
                "killed": "stopped"}.get(status, status)

    # ---------- 数据检查与缓存 ----------

    def inspect(self, config_name: str, timeout: float = INSPECT_TIMEOUT) -> dict:
        """同步跑 inspect_data.py,返回解析后的 JSON(catalog/plan/captions)。

        配置优先取 <name>_data.py(引擎生成的数据配置,带缓存编码器旋钮,
        其打印的 next 命令即用之);不存在则回退主配置(其自身旋钮均归
        train/moss.py 所有,同样可跑)。
        """
        exe = self._kogine()
        if exe is None:
            return {"ok": False, "engine_missing": True,
                    "error": "未找到训练引擎,请先到「环境与模型」页检查安装"}
        target, err = self._resolve_config(config_name)
        if err:
            return {"ok": False, "error": err}
        data_cfg = target.with_name(f"{target.stem}_data.py")
        use = data_cfg if data_cfg.is_file() else target
        rel = use.relative_to(self.adapter.repo_root).as_posix()
        try:
            script = resolve_scripts(use.read_text(encoding="utf-8",
                                                    errors="replace")).get(
                "inspect_data.py", INSPECT_SCRIPT)
        except OSError:
            script = INSPECT_SCRIPT
        try:
            proc = self.adapter._run([str(exe), "run", script,
                                      "--config", rel], timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"数据检查超时({timeout}s)", "config": rel}
        except OSError as err2:
            return {"ok": False, "error": f"训练引擎启动失败:{err2}", "config": rel}
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        report = extract_json(combined)
        if proc.returncode != 0 or report is None:
            return {"ok": False, "config": rel, "exit_code": proc.returncode,
                    "error": (None if proc.returncode == 0
                              else f"inspect 退出码 {proc.returncode}")
                    or "输出中未解析到 JSON",
                    "stdout": proc.stdout or "", "stderr": proc.stderr or ""}
        return {"ok": True, "config": rel, "exit_code": proc.returncode,
                "report": report}

    def start_cache(self, config_name: str, kind: str) -> dict:
        """启动文本/latent 缓存长任务(走 TaskManager;需 _data.py 配置)。"""
        exe = self._kogine()
        if exe is None:
            return {"ok": False, "engine_missing": True,
                    "error": "未找到训练引擎,请先到「环境与模型」页检查安装"}
        argv, rel, err = self._cache_argv(exe, config_name, kind)
        if err:
            return {"ok": False, "error": err}
        busy = self._gpu_busy()
        if busy is not None:
            return {"ok": False, "error": "已有 GPU 任务在运行,请先停止", "task": busy}
        task = self.tasks.start(kind, argv, config_path=rel, run_dir=None)
        return {"ok": True, "task": task}
