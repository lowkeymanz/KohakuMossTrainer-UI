"""引擎适配层(兼容性防火墙)。

UI 后端与训练引擎之间的**唯一** subprocess 出口:
- 不 import 任何引擎模块,只通过 CLI 可执行文件交互;
- 进程启动一律使用参数数组(禁止 shell 字符串拼接);
- 引擎未安装/半安装时,探测必须优雅返回"未就绪",绝不抛异常。
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

# 引擎在 venv 中安装的控制台入口点(需求文档 §2.1)
ENGINE_EXECUTABLES = (
    "kohakumoss-doctor",
    "kohakumoss-config",
    "kohakumoss-tune",
    "kogine",
)

# 短命令默认超时(秒);doctor 需要加载 torch 与内核矩阵,单独放宽
DEFAULT_TIMEOUT = 30
DOCTOR_TIMEOUT = 900  # doctor 含内核编译探测,冷缓存时可耗时数分钟
# probe 结果缓存时长(秒):doctor 要冷启动 torch,页面每刷新都重跑代价极高;
# 环境页"重新探测"传 fresh=True 绕过。
PROBE_TTL = 60.0


def engine_env(repo_root: Path) -> dict[str, str]:
    """引擎子进程环境:把仓库 `src/` 前置到 PYTHONPATH,使 `import kohakumoss`
    解析到 fork 自带源码(与仓库脚本、ComfyUI 权重配套),而不是 `.venv` 里可能
    更新过的已安装副本(新版不认仓库脚本的旋钮与 ComfyUI 权重布局)。
    `src/` 不存在时原样回退已安装包。"""
    env = dict(os.environ)
    src = repo_root / "src"
    if src.is_dir():
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(src) + (os.pathsep + existing if existing else "")
    return env


class EngineAdapter:
    """封装所有对引擎 CLI 的调用;上游接口变化时只改本文件。"""

    def __init__(self, repo_root: str | os.PathLike | None = None,
                 venv_dir: str | os.PathLike | None = None) -> None:
        # 默认布局:ui/backend/adapter.py -> parents[2] == 仓库根
        self.repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
        # 引擎环境固定在仓库根 .venv(UI 自身的 ui/.venv 与此无关)
        self.venv_dir = Path(venv_dir) if venv_dir else self.repo_root / ".venv"
        # UI 状态目录(ui/state),doctor 的 JSON 临时报告写在这里
        self.state_dir = Path(__file__).resolve().parents[1] / "state"

    # ---------- venv 布局探测 ----------

    def bin_dir(self) -> Path:
        """Windows 为 .venv/Scripts,POSIX 为 .venv/bin。"""
        if os.name == "nt":
            return self.venv_dir / "Scripts"
        return self.venv_dir / "bin"

    def find_executable(self, name: str) -> Path | None:
        """在 venv 的可执行目录中查找入口点,找不到返回 None(绝不抛异常)。"""
        try:
            d = self.bin_dir()
            if not d.is_dir():
                return None
            if os.name == "nt":
                for suffix in (".exe", ".cmd", ".bat", ""):
                    p = d / (name + suffix)
                    if p.is_file():
                        return p
                return None
            p = d / name
            return p if p.is_file() else None
        except OSError:
            return None

    def find_bundle_script(self, rel: str) -> Path | None:
        """已安装引擎 wheel 布局中的束脚本 site-packages/kohakumoss/bundle/scripts/<rel>。

        M2 冒烟实测:安装包脚本与仓库 scripts/ 快照可能不同版,引擎生成的配置
        docstring 打印的正是束脚本路径;无配置可打印的脚本(export/compare_grid)
        按同一布局探测。Windows 为 Lib/site-packages,POSIX 为 lib/python*/site-packages;
        找不到返回 None(调用方回退仓库相对路径)。"""
        try:
            hits: list[Path] = []
            hits += list(self.venv_dir.glob(
                f"Lib/site-packages/kohakumoss/bundle/scripts/{rel}"))
            hits += list(self.venv_dir.glob(
                f"lib/python*/site-packages/kohakumoss/bundle/scripts/{rel}"))
            return hits[0] if hits else None
        except OSError:
            return None

    # ---------- 唯一的短命令出口 ----------

    def _run(self, argv: list[str], timeout: float = DEFAULT_TIMEOUT) -> subprocess.CompletedProcess:
        """以参数数组执行子进程,cwd=仓库根(脚本/配置/runs 均相对根解析)。"""
        return subprocess.run(
            [str(a) for a in argv],
            cwd=str(self.repo_root),
            env=engine_env(self.repo_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            shell=False,
        )

    # ---------- 能力探测 ----------

    def doctor_json(self, timeout: float = DOCTOR_TIMEOUT) -> dict:
        """运行 kohakumoss-doctor --json <文件> 并读回解析。

        上游 CLI 的 --json 参数是**输出文件路径**(报告不打印到 stdout),
        因此写入 ui/state 下的临时文件再读取;字段一律透传,不硬编码清单。
        失败时抛 RuntimeError(由 probe 兜底捕获)。
        """
        exe = self.find_executable("kohakumoss-doctor")
        if exe is None:
            raise RuntimeError("kohakumoss-doctor 不存在(引擎未安装)")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.state_dir / "doctor_report.json"
        try:
            try:
                self._run([exe, "--json", str(report_path)], timeout=timeout)
            except subprocess.TimeoutExpired as err:
                raise RuntimeError(f"kohakumoss-doctor 超时({timeout}s),引擎可能半安装") from err
            except OSError as err:
                raise RuntimeError(f"kohakumoss-doctor 无法执行:{err}") from err
            if not report_path.is_file():
                raise RuntimeError("kohakumoss-doctor 未生成 JSON 报告")
            try:
                return json.loads(report_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as err:
                raise RuntimeError(f"doctor JSON 报告解析失败:{err}") from err
        finally:
            try:
                report_path.unlink(missing_ok=True)
            except OSError:
                pass

    def probe(self, fresh: bool = False) -> dict:
        """探测引擎就绪状态;任何情况下都返回 dict,绝不抛异常。

        结果缓存 PROBE_TTL 秒(doctor 冷启动 torch 代价高),fresh=True 绕过。

        返回:
            engine_ready: 四个入口点齐全且 doctor JSON 可解析时为 True
            doctor:       kohakumoss-doctor --json 的解析结果(未就绪时为 None)
            missing:      缺失的入口点名称列表
            error:        执行/解析失败的原因(无则为 None)
        """
        now = time.monotonic()
        cached = getattr(self, "_probe_cache", None)
        if cached and not fresh and now - cached[0] < PROBE_TTL:
            return cached[1]
        result = {"engine_ready": False, "doctor": None, "missing": [], "error": None}
        try:
            missing = [n for n in ENGINE_EXECUTABLES if self.find_executable(n) is None]
            result["missing"] = missing
            if missing:
                # 入口点不齐:直接返回"未就绪",不执行任何可能半安装的 exe
                self._probe_cache = (now, result)
                return result
            try:
                result["doctor"] = self.doctor_json()
                result["engine_ready"] = True
            except RuntimeError as err:
                result["error"] = str(err)
        except Exception as err:  # 最后一道保险:探测永不崩溃
            result["error"] = f"探测异常:{err!r}"
        self._probe_cache = (now, result)
        return result
