"""M4 契约测试:CLI argv 金样 + 真实 --help 对照。

金样固化 M1-M3 实测出的引擎 CLI 行为偏差:
- `new train` **没有 --set**(高级覆盖在训练启动时经 `kogine run --set` 生效);
- `--data` 单值(多数据集由 UI 生成后追加 DATASETS,不进 argv);
- `--out` 总是显式传(不传时 wheel 安装的引擎把默认输出目录退化为 cwd);
- 训练/采样/缓存走 `kogine run <script> --config <cfg> [--set K=V ...]`;
- 导出/网格只有 --set、没有 --config;列表旋钮用 JSON 方括号语法传值。

--help 对照用根 .venv 的真实引擎 CLI(只读探测,不跑训练/采样);
引擎缺失的环境(如 CI)自动 skip。
"""

import json
import re
import subprocess
from pathlib import Path

import pytest

from adapter import EngineAdapter
from config_service import (ConfigService, build_from_argv, build_sample_argv,
                            build_train_argv)
from export_service import ExportService
from sample_service import SampleService
from train_service import TrainService

# 复用 M3 测试的假 adapter/任务表(同目录,pytest prepend 模式可 import)
from test_m3_export_sample import FakeAdapter, FakeTasks, make_run

EXE = "kohakumoss-config.exe"
KOGINE = "kogine.exe"

# ---------- 金样:参数数组构造(与上游 flag 一一对应) ----------

FULL_TRAIN_FORM = {
    "weights": "D:/models/moss",
    "data": ["D:/ds/100_cat"],
    "name": "my-lora",
    "algo": "lora",
    "source": "kohya",
    "steps": 1600,
    "prompt": ["a cat", "a {trigger} cat"],
    "vram": 8,
    "precision": "bf16",
    "gpus": 2,
    "force": True,
}


def test_golden_new_train_argv():
    argv, err = build_train_argv(EXE, FULL_TRAIN_FORM)
    assert err is None
    assert argv == [
        EXE, "new", "train",
        "--weights", "D:/models/moss",
        "--data", "D:/ds/100_cat",
        "--name", "my-lora",
        "--algo", "lora",
        "--source", "kohya",
        "--steps", "1600",
        "--prompt", "a cat",
        "--prompt", "a {trigger} cat",
        "--vram", "8",
        "--precision", "bf16",
        "--gpus", "2",
        "--force",
    ]


def test_golden_new_sample_argv():
    argv, err = build_sample_argv(EXE, {
        "weights": "D:/models/moss", "name": "shot",
        "adapter": ["D:/ckpt/trainable.safetensors:0.8", "D:/b.sft"],
        "samples": "D:/gallery/shot", "prompt": ["a cat"],
        "vram": 8, "precision": "fp16", "force": True,
    })
    assert err is None
    assert argv == [
        EXE, "new", "sample",
        "--weights", "D:/models/moss",
        "--name", "shot",
        "--adapter", "D:/ckpt/trainable.safetensors:0.8",
        "--adapter", "D:/b.sft",
        "--samples", "D:/gallery/shot",
        "--prompt", "a cat",
        "--vram", "8",
        "--precision", "fp16",
        "--force",
    ]


def test_golden_from_preset_argv():
    argv, err = build_from_argv(EXE, "lora_12g_stream", "configs/user/base.py", True)
    assert err is None
    assert argv == [EXE, "new", "--from", "lora_12g_stream",
                    "configs/user/base.py", "--force"]


def test_golden_train_start_argv_and_resume(tmp_path):
    """训练启动金样:kogine run <script> --config <rel> [--set ...];
    续训追加 --set RESUME=latest(M2 契约:上游无暂停信号)。"""
    fa = FakeAdapter(tmp_path)
    ft = FakeTasks()
    cfg_dir = fa.repo_root / "configs" / "user"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "demo.py").write_text(
        'RUN_NAME = "demo-run"\nSTEPS = 1600\n', encoding="utf-8")
    ts = TrainService(fa, ft, ConfigService(fa, datasets_dir=fa.repo_root / "ds"))
    res = ts.start("demo", {"LR": "1e-4", "STEPS": "30"})
    assert res["ok"]
    assert res["task"]["argv"] == [
        str(fa.find_executable("kogine")), "run", "scripts/train/moss.py",
        "--config", "configs/user/demo.py",
        "--set", "LR=1e-4", "--set", "STEPS=30"]
    ft.rows.clear()
    res2 = ts.start("demo", {}, resume=True)
    assert res2["task"]["argv"][-2:] == ["--set", "RESUME=latest"]


class NewTrainRecorder(FakeAdapter):
    """记录 new train 的 argv 并模拟引擎按 --out 写出配置文件。"""

    def __init__(self, repo_root):
        super().__init__(repo_root)
        self.train_argv = None

    def _run(self, argv, timeout=30):
        argv = [str(a) for a in argv]
        if argv[1:3] == ["new", "train"]:
            self.train_argv = argv
            out = Path(argv[argv.index("--out") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text('RUN_NAME = "x"\n', encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")


def test_new_train_always_passes_out_and_never_set(tmp_path):
    """M1 实测偏差固化:new train 无 --set;--out 显式指向 configs/user/<name>.py
    (不传时 wheel 安装引擎写到 cwd);--data 恰好一个。"""
    fa = NewTrainRecorder(tmp_path)
    cs = ConfigService(fa, datasets_dir=tmp_path / "ds")
    res = cs.new_train({**FULL_TRAIN_FORM,
                        "overrides": {"LR": "1e-4"}})  # overrides 不进 argv
    assert res["ok"], res
    argv = fa.train_argv
    assert "--set" not in argv
    assert argv.count("--data") == 1
    assert argv[argv.index("--out") + 1].replace("\\", "/").endswith(
        "configs/user/my-lora.py")
    assert res["deferred_overrides"] == {"LR": "1e-4"}


def test_export_grid_only_set_no_config(tmp_path):
    """M3 偏差固化:export/compare_grid 无 starter 配置,argv 只有 --set、
    没有 --config;网格列表旋钮用 JSON 方括号语法。"""
    fa = FakeAdapter(tmp_path)
    ft = FakeTasks()
    make_run(fa.repo_root, steps=(40,))
    es = ExportService(fa, ft)
    res = es.start({"run": "demo-run", "format": "comfy"})
    assert res["ok"], res
    argv = res["task"]["argv"]
    assert "--config" not in argv and argv.count("--set") == 4
    ft.rows.clear()  # 释放 GPU 独占,便于继续测网格
    ss = SampleService(fa, ft, ConfigService(fa, datasets_dir=tmp_path / "ds"))
    res2 = ss.start_grid({"run": "demo-run", "checkpoints": ["40"],
                          "prompts": ["a cat"]})
    assert res2["ok"], res2
    argv2 = res2["task"]["argv"]
    assert "--config" not in argv2
    pairs = argv2[3::2]
    assert "--set" in pairs
    cp = argv2[argv2.index("--set") + 1]  # 首个 WEIGHTS=...
    assert cp.startswith("WEIGHTS=")
    # CHECKPOINTS 值是合法 JSON 数组(引擎实测的方括号传参语法)
    for tok in argv2:
        if tok.startswith("CHECKPOINTS="):
            json.loads(tok[len("CHECKPOINTS="):])
            break
    else:
        pytest.fail("argv 中应有 CHECKPOINTS= JSON 值")


# ---------- 真实引擎 CLI --help 对照(只读探测,引擎缺失则 skip) ----------

@pytest.fixture(scope="module")
def real_help():
    """跑根 .venv 真实 CLI 的 --help(纯只读,不触碰训练),结果缓存。"""
    adapter = EngineAdapter()  # 默认仓库根 + .venv
    texts = {}
    for key, argv in (
        ("config", ["kohakumoss-config", "--help"]),
        ("new_train", ["kohakumoss-config", "new", "train", "--help"]),
        ("new_sample", ["kohakumoss-config", "new", "sample", "--help"]),
        ("kogine", ["kogine", "--help"]),
        ("kogine_run", ["kogine", "run", "--help"]),
    ):
        exe = adapter.find_executable(argv[0])
        if exe is None:
            pytest.skip(f"引擎未安装(缺 {argv[0]}),跳过 --help 对照")
        try:
            proc = adapter._run([exe] + argv[1:], timeout=180)
        except (subprocess.TimeoutExpired, OSError) as err:
            pytest.skip(f"引擎 --help 探测失败:{err}")
        assert proc.returncode == 0, proc.stderr[-500:]
        texts[key] = proc.stdout
    return texts


def _flags(text: str) -> set[str]:
    return set(re.findall(r"--([a-z][a-z0-9-]+)", text))


def test_help_new_train_flags_superset(real_help):
    """UI 发出的每个 new train flag 都必须在真实 --help 中存在;
    且真实 help 不含 --set(M1 偏差:--set 是 kogine run 的机制)。"""
    flags = _flags(real_help["new_train"])
    ui_flags = _flags(" ".join(build_train_argv(EXE, FULL_TRAIN_FORM)[0]))
    assert ui_flags <= flags, f"UI 用了引擎不认识的 flag: {ui_flags - flags}"
    assert {"weights", "data", "source", "algo", "gpus", "name", "steps",
            "prompt", "vram", "precision", "out", "force"} <= flags
    assert "set" not in flags


def test_help_new_sample_flags_superset(real_help):
    flags = _flags(real_help["new_sample"])
    ui_flags = _flags(" ".join(build_sample_argv(EXE, {
        "weights": "w", "name": "n", "adapter": ["a"], "samples": "s",
        "prompt": ["p"], "vram": 8})[0]))
    assert ui_flags <= flags, f"UI 用了引擎不认识的 flag: {ui_flags - flags}"


def test_help_config_subcommands(real_help):
    """UI 只使用 list / new / check 三个子命令,必须都在 help 里。"""
    text = real_help["config"]
    for sub in ("list", "new", "check"):
        assert re.search(rf"\b{sub}\b", text), f"config --help 缺子命令 {sub}"


def test_help_kogine_run_set_and_config(real_help):
    """kogine run 支持 --config 与可重复 --set(M2 契约的载体)。"""
    flags = _flags(real_help["kogine_run"])
    assert {"config", "set"} <= flags
    assert re.search(r"--set KEY=VALUE.*repeatable", real_help["kogine_run"],
                     re.S)
    assert "run" in real_help["kogine"]


def test_doctor_json_file_path_contract(tmp_path, monkeypatch):
    """M0 实测契约:kohakumoss-doctor --json <输出文件路径>(报告不走 stdout),
    doctor_json 必须写临时文件再读回解析。"""
    calls = []

    class FakeDoctor(EngineAdapter):
        def find_executable(self, name):
            return tmp_path / "kohakumoss-doctor.exe" if name == "kohakumoss-doctor" else None

        def _run(self, argv, timeout=30):
            calls.append([str(a) for a in argv])
            Path(calls[-1][2]).write_text('{"devices": ["cuda:0"]}',
                                          encoding="utf-8")
            return subprocess.CompletedProcess(calls[-1], 0, stdout="", stderr="")

    adapter = FakeDoctor(repo_root=tmp_path, venv_dir=tmp_path)
    adapter.state_dir = tmp_path / "state"
    report = adapter.doctor_json()
    assert report == {"devices": ["cuda:0"]}
    assert calls[0][1] == "--json" and Path(calls[0][2]).parent == tmp_path / "state"
    assert not Path(calls[0][2]).exists()  # 临时报告读完即删
