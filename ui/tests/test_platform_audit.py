"""M4 双平台审计测试(防回归):
1. ui/backend 内所有 subprocess.run 必须带 text=True + encoding="utf-8"
   + errors="replace"(stdout 含中文/编码混杂时跨平台解码一致);
2. 禁止 shell=True / os.system / subprocess.getoutput(argv 一律参数数组);
3. 禁止 os.path.*(路径全部 pathlib;os.pathsep 仅用于 PYTHONPATH 拼接,放行);
4. 文本模式 open() 必须显式 encoding;
5. UI 后端绝不 import 引擎模块(零耦合铁律的静态守卫);
6. 平台分支点行为核对:venv 布局、采样绕开开关、终止信号分支。
"""

import ast
import os
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1] / "backend"
PY_FILES = sorted(BACKEND.glob("*.py"))


def _calls(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            yield node


def _func_name(node: ast.Call) -> str:
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        base = getattr(f.value, "id", getattr(f.value, "attr", ""))
        return f"{base}.{f.attr}"
    return ""


def _kw(node: ast.Call, name: str):
    for kw in node.keywords:
        if kw.arg == name:
            return kw.value
    return None


def test_no_backend_files_missing():
    assert len(PY_FILES) >= 10  # 审计范围非空


def test_subprocess_run_decodes_utf8_everywhere():
    """任何 subprocess.run 调用必须显式 utf-8+replace,否则 Linux 默认
    locale 编码与 Windows 不一致,中文日志会炸。"""
    for path in PY_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for call in _calls(tree):
            if _func_name(call) != "subprocess.run":
                continue
            text = _kw(call, "text")
            enc = _kw(call, "encoding")
            err = _kw(call, "errors")
            assert isinstance(text, ast.Constant) and text.value is True, \
                f"{path.name}: subprocess.run 缺 text=True"
            assert isinstance(enc, ast.Constant) and enc.value == "utf-8", \
                f"{path.name}: subprocess.run 缺 encoding='utf-8'"
            assert isinstance(err, ast.Constant) and err.value == "replace", \
                f"{path.name}: subprocess.run 缺 errors='replace'"


def test_no_shell_string_execution():
    for path in PY_FILES:
        src = path.read_text(encoding="utf-8")
        assert "shell=True" not in src, f"{path.name}: 禁止 shell=True"
        assert "os.system" not in src, f"{path.name}: 禁止 os.system"
        assert "subprocess.getoutput" not in src and \
            "subprocess.getstatusoutput" not in src
        tree = ast.parse(src)
        for call in _calls(tree):
            if _func_name(call) == "subprocess.Popen":
                sh = _kw(call, "shell")
                assert not (isinstance(sh, ast.Constant) and sh.value is True), \
                    f"{path.name}: Popen 禁止 shell=True"


def test_no_os_path_only_pathlib():
    """路径处理全部 pathlib;os.pathsep/env 不算路径操作,放行。"""
    for path in PY_FILES:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
        for call in _calls(tree):
            name = _func_name(call)
            assert not name.startswith("os.path."), \
                f"{path.name}: 用了 {name},请改 pathlib"


def test_text_mode_open_has_encoding():
    for path in PY_FILES:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
        for call in _calls(tree):
            if _func_name(call) != "open":
                continue
            mode = ""
            if len(call.args) >= 2 and isinstance(call.args[1], ast.Constant):
                mode = str(call.args[1].value)
            for kw in call.keywords:
                if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                    mode = str(kw.value.value)
            if "b" in mode:
                continue  # 二进制模式无需编码
            assert _kw(call, "encoding") is not None, \
                f"{path.name}: 文本模式 open() 缺 encoding={call.lineno}"


def test_backend_never_imports_engine():
    """零耦合铁律:ui/backend 不 import 任何 kohakumoss*/kogine 引擎模块。"""
    banned = ("kohakumoss", "kohakuengine", "kogine")
    for path in PY_FILES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            for m in mods:
                assert not m.split(".")[0].startswith(
                    tuple(banned)), f"{path.name}: import 了引擎模块 {m}"


# ---------- 平台分支点行为核对 ----------

def test_venv_bin_dir_platform_branch(tmp_path, monkeypatch):
    """bin_dir:Windows -> Scripts,POSIX -> bin(find_bundle_script 的
    Lib/site-packages 与 lib/python* 双 glob 同理覆盖两平台)。"""
    from adapter import EngineAdapter
    a = EngineAdapter(repo_root=tmp_path, venv_dir=tmp_path / "v")
    monkeypatch.setattr(os, "name", "nt")
    assert a.bin_dir() == tmp_path / "v" / "Scripts"
    monkeypatch.setattr(os, "name", "posix")
    assert a.bin_dir() == tmp_path / "v" / "bin"
    # bundle 布局:POSIX 的 lib/python3.12/site-packages 也能探到
    posix_bundle = (tmp_path / "v" / "lib" / "python3.12" / "site-packages" /
                    "kohakumoss" / "bundle" / "scripts")
    (posix_bundle / "export").mkdir(parents=True)
    hit = posix_bundle / "export" / "export.py"
    hit.write_text("x", encoding="utf-8")
    assert a.find_bundle_script("export/export.py") == hit


def test_engine_env_uses_pathsep(tmp_path):
    """PYTHONPATH 拼接用 os.pathsep(Windows ';' / Linux ':'),不能写死。"""
    from adapter import engine_env
    src = tmp_path / "src"
    src.mkdir()
    env = engine_env(tmp_path)
    assert env["PYTHONPATH"].startswith(str(src))
    assert os.pathsep in env["PYTHONPATH"] or env["PYTHONPATH"] == str(src)


def test_win_sampling_workaround_branch_point():
    """采样崩溃绕开开关必须严格绑定 win32(Linux 无此 bug,保持引擎原生行为)。"""
    import sample_service
    src = (BACKEND / "sample_service.py").read_text(encoding="utf-8")
    assert '_WIN_SAMPLING_WORKAROUND = sys.platform == "win32"' in src
    assert sample_service._WIN_SAMPLING_WORKAROUND == (sys.platform == "win32")


def test_terminate_signal_branch_in_source():
    """终止信号平台分支:Windows CTRL_BREAK_EVENT(进程组),POSIX SIGTERM;
    两者都必须存在且各自受平台条件保护。"""
    src = (BACKEND / "task_manager.py").read_text(encoding="utf-8")
    assert "signal.CTRL_BREAK_EVENT" in src
    assert "signal.SIGTERM" in src
    assert "CREATE_NEW_PROCESS_GROUP" in src
    assert 'os.name == "nt"' in src
    # Windows 存活检测不能用 os.kill(pid, 0)(实测会把信号当退出码)
    assert "OpenProcess" in src and "GetExitCodeProcess" in src
