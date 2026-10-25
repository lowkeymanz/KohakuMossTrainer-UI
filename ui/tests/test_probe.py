"""probe() 在引擎缺失/半安装时必须优雅返回,绝不抛异常。

所有用例都指向 tmp 目录伪造的 venv,**绝不触碰仓库根 .venv**。
"""

import os
from pathlib import Path

from adapter import ENGINE_EXECUTABLES, EngineAdapter


def _make_venv(tmp_path: Path) -> Path:
    """按当前平台布局建出 venv 可执行目录。"""
    bin_dir = tmp_path / "Scripts" if os.name == "nt" else tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    return bin_dir


def test_probe_missing_engine_returns_not_ready(tmp_path):
    """venv 完全不存在:engine_ready=False,missing 列出全部入口点,无异常。"""
    adapter = EngineAdapter(repo_root=tmp_path / "repo", venv_dir=tmp_path / "no-venv")
    result = adapter.probe()
    assert result["engine_ready"] is False
    assert result["doctor"] is None
    assert sorted(result["missing"]) == sorted(ENGINE_EXECUTABLES)
    assert result["error"] is None


def test_probe_partial_install_lists_missing(tmp_path):
    """只装了部分入口点:missing 只列缺的,且不去执行半安装环境。"""
    bin_dir = _make_venv(tmp_path)
    (bin_dir / ("kohakumoss-doctor.exe" if os.name == "nt" else "kohakumoss-doctor")).write_text("x")
    adapter = EngineAdapter(repo_root=tmp_path, venv_dir=tmp_path)
    result = adapter.probe()
    assert result["engine_ready"] is False
    assert "kohakumoss-doctor" not in result["missing"]
    assert "kogine" in result["missing"]


def test_probe_broken_executable_does_not_raise(tmp_path):
    """入口点文件存在但不可执行(半安装典型症状):engine_ready=False + error 文案。"""
    bin_dir = _make_venv(tmp_path)
    for name in ENGINE_EXECUTABLES:
        suffix = ".exe" if os.name == "nt" else ""
        (bin_dir / (name + suffix)).write_text("not a real executable")
    adapter = EngineAdapter(repo_root=tmp_path, venv_dir=tmp_path)
    result = adapter.probe()  # 不应抛异常
    assert result["engine_ready"] is False
    assert result["doctor"] is None
    assert result["error"]  # 有可读的失败原因


def test_doctor_json_raises_when_exe_missing(tmp_path):
    """底层 doctor_json 在缺 exe 时抛 RuntimeError(由 probe 兜底)。"""
    adapter = EngineAdapter(repo_root=tmp_path, venv_dir=tmp_path / "none")
    try:
        adapter.doctor_json()
        assert False, "应当抛出 RuntimeError"
    except RuntimeError:
        pass


def test_probe_result_cached_until_fresh(tmp_path):
    """probe 缓存:doctor 只跑一次;fresh=True 或缓存过期才重跑。"""
    calls = []

    class _Counting(EngineAdapter):
        def find_executable(self, name):
            return Path("fake-exe")

        def doctor_json(self):
            calls.append(1)
            return {"devices": ["cuda:0"]}

    a = _Counting(repo_root=tmp_path, venv_dir=tmp_path)
    r1 = a.probe()
    r2 = a.probe()
    assert len(calls) == 1 and r2["engine_ready"] is True
    a.probe(fresh=True)
    assert len(calls) == 2
    a._probe_cache = (a._probe_cache[0] - 999.0, a._probe_cache[1])  # 模拟过期
    a.probe()
    assert len(calls) == 3
