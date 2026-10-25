"""M4 契约测试:日志格式 fixture 回放 + 产物布局 fixture。

全部走假 adapter + tmp 仓库,不真跑引擎训练/采样:
- fixtures/metrics.jsonl / log.txt / inspect_output.txt 按 M2/M3 实测的引擎
  输出格式落盘,回放给 RunWatcher / extract_json,固化解析契约;
- run 目录布局(checkpoints/step_* 四件套、samples/、config.json、
  fresh_run 移名轮转)按引擎 io.py 实测契约构造并回放给 run_service /
  TrainService.snapshot。
"""

import json
import shutil
import time
from pathlib import Path

import pytest

from config_service import ConfigService
from metrics_reader import RunWatcher, parse_metric_line
from train_service import TrainService, extract_json, resolve_scripts
import run_service
from test_m3_export_sample import FakeAdapter, FakeTasks

FIXTURES = Path(__file__).resolve().parent / "fixtures"


# ---------- 日志格式 fixture 回放 ----------

def test_metrics_fixture_replay_full_tail(tmp_path):
    """真实格式的 metrics.jsonl:全部行可解析,validation 行的
    val_loss / val_loss@<sigma> 键原样透传给前端。"""
    run_dir = tmp_path / "runs" / "moss" / "demo"
    run_dir.mkdir(parents=True)
    shutil.copy(FIXTURES / "metrics.jsonl", run_dir / "metrics.jsonl")
    w = RunWatcher()
    w.attach(run_dir)
    rows, rotated = w.poll_metrics()
    assert not rotated
    assert [r["step"] for r in rows] == [1, 2, 3, 4, 5, 6, 7]
    val = rows[4]
    assert val["val_loss"] == pytest.approx(0.5821)
    assert "val_loss@1.0" in val
    # 再 poll 无增量(偏移已消费)
    assert w.poll_metrics() == ([], False)


def test_metrics_fixture_partial_line_and_rotation(tmp_path):
    """残行不消费(M2 契约);fresh_run 移名后文件变短 -> 轮转标志 +
    整文件重推(metrics.jsonl.1 是引擎 fresh_run 的实测命名)。"""
    run_dir = tmp_path / "runs" / "moss" / "demo"
    run_dir.mkdir(parents=True)
    src = (FIXTURES / "metrics.jsonl").read_text(encoding="utf-8")
    lines = src.splitlines()
    (run_dir / "metrics.jsonl").write_text("\n".join(lines[:5]) + "\n",
                                           encoding="utf-8")
    w = RunWatcher()
    w.attach(run_dir)
    rows, _ = w.poll_metrics()
    assert len(rows) == 5
    # 追加半行(进程写一半)
    with open(run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
        f.write(lines[5][:20])
    assert w.poll_metrics()[0] == []
    with open(run_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
        f.write(lines[5][20:] + "\n")
    rows2, _ = w.poll_metrics()
    assert [r["step"] for r in rows2] == [6]
    # 引擎重跑:旧文件移名 metrics.jsonl.1,新文件更短 -> 轮转
    (run_dir / "metrics.jsonl.1").write_text(src, encoding="utf-8")
    (run_dir / "metrics.jsonl").write_text(lines[0] + "\n", encoding="utf-8")
    rows3, rotated3 = w.poll_metrics()
    assert rotated3 and [r["step"] for r in rows3] == [1]


def test_log_fixture_replay_and_error_lines(tmp_path):
    """log.txt 行格式回放:ERROR 行原样透传(前端高亮);任务原始 stdout
    与 log.txt 并读合并(M2 修正:崩溃 traceback 只进任务 stdout,旧实现
    log.txt 出现后停读兜底导致 UI 看不到报错)。"""
    run_dir = tmp_path / "runs" / "moss" / "demo"
    run_dir.mkdir(parents=True)
    shutil.copy(FIXTURES / "log.txt", run_dir / "log.txt")
    fake_task_log = tmp_path / "task.log"
    fake_task_log.write_text("early stdout line\n", encoding="utf-8")
    w = RunWatcher()
    w.attach(run_dir)
    lines = w.poll_log(fake_task_log)
    assert len(lines) == 11
    assert any("ERROR" in ln for ln in lines)
    assert any("early stdout" in ln for ln in lines)  # 任务 stdout 不再被丢弃


def test_inspect_fixture_json_extraction():
    """inspect_data 经 log.info(json.dumps(report, indent=1)) 输出:
    首行带日志前缀、正文多行、含花括号字符串 -> extract_json 必须取出完整对象。"""
    text = (FIXTURES / "inspect_output.txt").read_text(encoding="utf-8")
    report = extract_json(text)
    assert report is not None
    assert report["catalog"]["images"] == 24
    assert report["captions"][1]["text"] == "a cat, {braces} in text"


def test_warmup_reject_deviation_pinned(tmp_path):
    """M3 实测偏差:steps 过小在生成期被 warmup 校验拒绝(退出码非 0、
    不落盘配置)。UI 契约:原样透传引擎 stderr,不吞错误。"""
    import subprocess

    class WarmupFail(FakeAdapter):
        def _run(self, argv, timeout=30):
            argv = [str(a) for a in argv]
            err = (FIXTURES / "warmup_reject.txt").read_text(encoding="utf-8")
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr=err)

    cs = ConfigService(WarmupFail(tmp_path), datasets_dir=tmp_path / "ds")
    res = cs.new_train({"weights": "w", "data": ["d"], "name": "tiny-steps",
                        "steps": 10})
    assert not res["ok"]
    assert "warmup" in res["error"]
    assert not (tmp_path / "configs" / "user" / "tiny-steps.py").exists()


# ---------- 产物布局 fixture ----------

RUN_LAYOUT_CONFIG_JSON = {
    "RUN_NAME": "demo-run", "STEPS": 40, "ALGO": "lora",
    "WEIGHTS": "D:/models/moss", "SAMPLE_EVERY": 20, "VALIDATE_EVERY": 10,
    "SAMPLE_PROMPTS": ["a cat"],
}


def _materialize_run(run_dir: Path):
    """按引擎 io.py 实测契约构造 run 目录:
    checkpoints/step_*/ 四件套(trainable/ema/optim/state)、samples/、
    export/、metrics.jsonl、log.txt、config.json。"""
    run_dir.mkdir(parents=True)
    for step in (20, 40):
        d = run_dir / "checkpoints" / f"step_{step:08d}"
        d.mkdir(parents=True)
        (d / "trainable.safetensors").write_bytes(b"\x00" * 8)
        (d / "optim.pt").write_bytes(b"\x00" * 4)
        (d / "state.json").write_text('{"step": %d}' % step, encoding="utf-8")
        if step == 40:  # ema 仅启用 EMA 时才有(M3 实测)
            (d / "ema.safetensors").write_bytes(b"\x00" * 8)
    (run_dir / "samples").mkdir()
    (run_dir / "samples" / "step_00000020_0.png").write_bytes(b"\x89PNG fake")
    exp = run_dir / "export"
    exp.mkdir()
    (exp / "demo-run-comfy.safetensors").write_bytes(b"\x00")  # 引擎自动导出
    (exp / "demo-run-lycoris.safetensors").write_bytes(b"\x00")
    shutil.copy(FIXTURES / "metrics.jsonl", run_dir / "metrics.jsonl")
    shutil.copy(FIXTURES / "log.txt", run_dir / "log.txt")
    (run_dir / "config.json").write_text(json.dumps(RUN_LAYOUT_CONFIG_JSON),
                                         encoding="utf-8")
    return run_dir


def test_run_layout_checkpoint_listing(tmp_path):
    run_dir = _materialize_run(tmp_path / "runs" / "moss" / "demo-run")
    res = run_service.list_checkpoints(tmp_path, "demo-run")
    assert res["ok"]
    ck = {c["dir"]: c for c in res["checkpoints"]}
    assert list(ck) == ["step_00000020", "step_00000040"]
    assert ck["step_00000020"]["complete"] and not ck["step_00000020"]["has_ema"]
    assert ck["step_00000040"]["has_ema"]
    # 默认引用 = 最新 usable checkpoint
    assert run_service.find_checkpoint_dir(run_dir, None).name == "step_00000040"
    # 引擎自动导出产物在 run/export/ 下被列出(M3 实测:训练结束自动 comfy+lycoris)
    runs = run_service.list_runs(tmp_path)["runs"]
    assert runs[0]["export_count"] == 2 and runs[0]["sample_count"] == 1
    assert runs[0]["algo"] == "lora" and runs[0]["last_step"] == 40


def test_run_layout_sample_whitelist(tmp_path):
    _materialize_run(tmp_path / "runs" / "moss" / "demo-run")
    ok = run_service.resolve_sample_file(
        tmp_path, "runs/moss/demo-run/samples/step_00000020_0.png")
    assert ok["ok"]
    assert run_service.resolve_sample_file(
        tmp_path, "runs/moss/demo-run/checkpoints/step_00000040/trainable.safetensors"
    )["forbidden"]


def test_run_layout_snapshot(tmp_path):
    """产物快照回放:metrics 行数/总步数/sample_every/日志尾部/样图列表,
    全部来自布局 fixture,不跑真引擎。"""
    fa = FakeAdapter(tmp_path)
    ft = FakeTasks()
    _materialize_run(tmp_path / "runs" / "moss" / "demo-run")
    ts = TrainService(fa, ft, ConfigService(fa, datasets_dir=tmp_path / "ds"))
    task = {"id": "t1", "kind": "train", "status": "exited_ok",
            "run_dir": str(tmp_path / "runs" / "moss" / "demo-run"),
            "log_path": None, "overrides": {}, "argv": []}
    snap = ts.snapshot(task)
    assert snap["phase"] == "finished"
    assert snap["total_steps"] == 40
    assert snap["step"] == 7  # fixture metrics 最后一行
    assert snap["sample_every"] == 20 and snap["validate_every"] == 10
    assert len(snap["metrics"]) == 7
    assert len(snap["samples"]) == 1
    assert snap["samples"][0].endswith("step_00000020_0.png")
    assert any("ERROR" in ln for ln in snap["log_tail"])


def test_resolve_scripts_docstring_contract(tmp_path):
    """M1 实测契约:引擎生成的配置 docstring 打印权威运行命令
    (bundle 安装时是 site-packages 绝对路径),UI 以它为准定位脚本。"""
    text = (
        '"""demo config."""\n'
        "RUN_NAME = \"demo-run\"\n"
        "kogine run D:/x/site-packages/kohakumoss/bundle/scripts/train/moss.py"
        " --config D:/x/configs/user/demo.py\n"
        "kogine run D:/x/site-packages/kohakumoss/bundle/scripts/data/"
        "inspect_data.py --config D:/x/configs/user/demo_data.py\n"
    )
    got = resolve_scripts(text)
    assert got["moss.py"].endswith("bundle/scripts/train/moss.py")
    assert got["inspect_data.py"].endswith("bundle/scripts/data/inspect_data.py")
    assert run_service.parse_run_name(text, "fallback") == "demo-run"
