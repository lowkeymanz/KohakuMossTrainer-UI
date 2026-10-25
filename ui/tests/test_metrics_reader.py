"""metrics.jsonl / log.txt / samples 增量读取测试(RunWatcher)。

覆盖:增量 tail、半行容错、轮转/重写(变短)重置、按 step 去重、
attach 换目录重置、样图新文件扫描、log.txt 与任务 stdout 切换去重。
"""

import json

from metrics_reader import RunWatcher, parse_metric_line


def write_lines(path, lines):
    with open(path, "a", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln if ln.endswith("\n") else ln + "\n")


def row(step, loss=0.5, **kw):
    return json.dumps({"step": step, "time": 1000.0 + step, "loss": loss, **kw})


def test_incremental_tail_and_partial_line(tmp_path):
    m = tmp_path / "metrics.jsonl"
    w = RunWatcher()
    w.attach(tmp_path)
    # 文件不存在:静默空
    rows, rotated = w.poll_metrics()
    assert rows == [] and rotated is False
    write_lines(m, [row(10), row(20)])
    rows, rotated = w.poll_metrics()
    assert [r["step"] for r in rows] == [10, 20] and rotated is False
    # 无新增 -> 空
    assert w.poll_metrics()[0] == []
    # 半行:未写完的最后一行不消费,补全剩余内容后一起出
    with open(m, "a", encoding="utf-8") as f:
        f.write(row(30)[: 20])
    assert w.poll_metrics()[0] == []
    with open(m, "a", encoding="utf-8") as f:
        f.write(row(30)[20:] + "\n")
    rows, _ = w.poll_metrics()
    assert [r["step"] for r in rows] == [30]


def test_rotation_shrink_resets_and_flags(tmp_path):
    """fresh_run:引擎把旧文件移名、新建同名文件 -> 文件变短 -> 重置 + run_reset。"""
    m = tmp_path / "metrics.jsonl"
    w = RunWatcher()
    w.attach(tmp_path)
    write_lines(m, [row(0), row(10), row(20)])
    assert [r["step"] for r in w.poll_metrics()[0]] == [0, 10, 20]
    m.unlink()
    write_lines(m, [row(0, loss=0.9)])  # 新 run 从 step 0 重来
    rows, rotated = w.poll_metrics()
    assert rotated is True
    assert [r["step"] for r in rows] == [0]
    assert rows[0]["loss"] == 0.9  # 是新内容而非旧缓存


def test_resume_rewrite_dedup_by_step(tmp_path):
    """resume_at:文件被重写为前缀(变短)-> 重置整文件重推;纯追加时按 step 去重。"""
    m = tmp_path / "metrics.jsonl"
    w = RunWatcher()
    w.attach(tmp_path)
    write_lines(m, [row(s) for s in (0, 10, 20, 30, 40)])
    assert len(w.poll_metrics()[0]) == 5
    # 模拟 resume_at:晚于恢复点的行移旁文件,重写为更短前缀 -> 变短触发轮转重置
    m.write_text("".join(row(s) + "\n" for s in (0, 10, 20)), encoding="utf-8")
    rows, rotated = w.poll_metrics()
    assert rotated is True
    assert [r["step"] for r in rows] == [0, 10, 20]
    # 前端收到 run_reset 清屏 + 整文件重推重建 -> 无重复段;
    # 续训纯追加(未变短):只推新行,旧 step 不重复
    write_lines(m, [row(30), row(40)])
    rows, rotated = w.poll_metrics()
    assert rotated is False and [r["step"] for r in rows] == [30, 40]


def test_attach_new_dir_resets_all(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir(); b.mkdir()
    write_lines(a / "metrics.jsonl", [row(10)])
    write_lines(b / "metrics.jsonl", [row(10)])
    w = RunWatcher()
    w.attach(a)
    assert len(w.poll_metrics()[0]) == 1
    w.attach(b)  # 换 run 目录(如 resume 新任务)-> 全量重置,b 的 step 10 重新可见
    assert len(w.poll_metrics()[0]) == 1


def test_parse_metric_line_tolerates_junk():
    assert parse_metric_line("not json") is None
    assert parse_metric_line("") is None
    assert parse_metric_line("[1,2]") is None  # 非 dict
    assert parse_metric_line(row(5))["step"] == 5


def test_samples_scan_new_files_only(tmp_path):
    s = tmp_path / "samples" / "step_00000010"
    s.mkdir(parents=True)
    (s / "000.png").write_bytes(b"x")
    w = RunWatcher()
    w.attach(tmp_path)
    first = w.poll_samples()
    assert len(first) == 1 and first[0].endswith("000.png")
    assert w.poll_samples() == []
    (s / "001.png").write_bytes(b"y")
    assert [p.rsplit("/", 1)[-1] for p in w.poll_samples()] == ["001.png"]
    # 非图片忽略
    (s / "settings.json").write_text("{}", encoding="utf-8")
    assert w.poll_samples() == []


def test_samples_filter_grid_summary(tmp_path):
    """samples/ 根下的汇总对比表 grid.png 不进样图墙,step_X/ 内单张照常。"""
    root = tmp_path / "samples"
    step = root / "step_00000010"
    step.mkdir(parents=True)
    (step / "000.png").write_bytes(b"x")
    (root / "grid.png").write_bytes(b"summary")  # 引擎每轮重写的拼图
    w = RunWatcher()
    w.attach(tmp_path)
    got = w.poll_samples()
    assert [p.rsplit("/", 1)[-1] for p in got] == ["000.png"]
    # 后续轮次:新 step 图照常进,新写入的 grid.png 覆盖仍被过滤
    (root / "grid.png").write_bytes(b"summary2")
    step2 = root / "step_00000020"
    step2.mkdir()
    (step2 / "000.png").write_bytes(b"z")
    assert [p.rsplit("/", 1)[-1] for p in w.poll_samples()] == ["000.png"]


def test_log_fallback_then_switch_to_log_txt_dedups(tmp_path):
    """任务原始 stdout 与 log.txt 每次并读合并:重叠行按内容去重,
    只进原始输出的行(崩溃栈等)在 log.txt 出现后仍能被读到。"""
    task_log = tmp_path / "state.log"
    run = tmp_path / "run"  # 尚不存在
    write_lines(task_log, ["early line", "init line"])
    w = RunWatcher()
    w.attach(run)
    assert w.poll_log(task_log) == ["early line", "init line"]
    # log.txt 出现(内容含原始输出已推过的行 + 新行)
    run.mkdir()
    write_lines(run / "log.txt", ["early line", "init line", "step 10 loss=0.5"])
    lines = w.poll_log(task_log)
    assert lines == ["step 10 loss=0.5"]  # 前两条被内容去重
    # 两个源的新行都要能读到:原始输出独有行(如 traceback)不再被漏掉
    write_lines(task_log, ["stdout-only"])
    write_lines(run / "log.txt", ["new in log.txt"])
    assert w.poll_log(task_log) == ["stdout-only", "new in log.txt"]


def test_log_traceback_visible_after_log_txt_exists(tmp_path):
    """M2 实测 bug 回归:引擎崩溃时 log.txt 只有 logger 的 INFO 行,
    Python traceback 只进任务原始输出落盘文件;log.txt 已存在的情况下,
    poll_log 必须仍能把崩溃栈读出来,否则 UI 看不到报错。"""
    task_log = tmp_path / "state.log"
    run = tmp_path / "run"
    run.mkdir()
    write_lines(run / "log.txt", ["[INFO] run started", "[INFO] building loader"])
    write_lines(task_log, ["[INFO] run started", "[INFO] building loader",
                           "Traceback (most recent call last):",
                           "ValueError: kohya source: no images"])
    w = RunWatcher()
    w.attach(run)
    lines = w.poll_log(task_log)
    assert "Traceback (most recent call last):" in lines
    assert "ValueError: kohya source: no images" in lines
    assert lines.count("[INFO] run started") == 1  # 镜像行只推一次


def test_read_config_caches(tmp_path):
    w = RunWatcher()
    w.attach(tmp_path)
    assert w.read_config() is None  # 尚未生成,可重试
    (tmp_path / "config.json").write_text(json.dumps({"STEPS": 30}), encoding="utf-8")
    assert w.read_config() == {"STEPS": 30}
    (tmp_path / "config.json").write_text("{broken", encoding="utf-8")
    # 已成功读取后缓存,损坏覆盖不影响
    assert w.read_config() == {"STEPS": 30}
