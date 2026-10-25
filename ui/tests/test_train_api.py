"""M2 训练端点测试:假 adapter/TaskManager,不依赖真引擎。

覆盖:argv 构造(--set/RESUME)、run 已存在 409、GPU 独占拒绝、stop/resume
委派、status 快照、inspect 解析、cache 任务、/api/run-files 白名单、
WS 事件推送(metric/logline/gpu/task_status/run_reset)。
"""

import json
import subprocess
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app as app_module
import train_service as ts_mod
from config_service import ConfigService
from train_service import TrainService

CONFIG_TEXT = '''"""demo."""
from kohakuengine import use_config

use_config("../moss/_base/base_lora.py")

RUN_NAME = "demo-run"
STEPS = 30
'''


class FakeAdapter:
    def __init__(self, repo_root):
        self.repo_root = repo_root
        self.procs = []  # 预置 inspect 用的 CompletedProcess
        self.last_run_argv = None

    def find_executable(self, name):
        return rf"{self.repo_root}\.venv\Scripts\{name}.exe" if name == "kogine" else None

    def _run(self, argv, timeout=30):
        self.last_run_argv = list(argv)
        if self.procs:
            return self.procs.pop(0)
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")


class FakeTasks:
    def __init__(self):
        self.rows = {}
        self.terminations = []
        self.resumptions = []

    def add(self, row):
        self.rows[row["id"]] = row

    def list_tasks(self):
        return list(self.rows.values())

    def get(self, tid):
        return self.rows.get(tid)

    def start(self, kind, argv, config_path=None, run_dir=None, overrides=None,
              after=None):
        row = {"id": f"t{len(self.rows)}", "kind": kind,
               "status": "pending" if after else "running",
               "argv": list(argv), "config_path": config_path, "run_dir": run_dir,
               "overrides": dict(overrides or {}), "pid": None if after else 4242,
               "log_path": None, "started_at": time.time(), "exit_code": None,
               "after_task": after}
        self.add(row)
        return row

    def terminate(self, tid, timeout=15.0):
        self.terminations.append(tid)
        self.rows[tid]["status"] = "killed"
        return self.rows[tid]

    def resume(self, tid):
        self.resumptions.append(tid)
        old = self.rows[tid]
        return self.start(old["kind"], list(old["argv"]) + ["--set", "RESUME=latest"],
                          config_path=old["config_path"], run_dir=old["run_dir"],
                          overrides={**old["overrides"], "RESUME": "latest"})


class FakeGpu:
    available = False
    def snapshot(self):
        return None


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """tmp 仓库根 + configs/user/demo.py;假引擎与假任务表。"""
    cfg_dir = tmp_path / "configs" / "user"
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "demo.py").write_text(CONFIG_TEXT, encoding="utf-8")
    fa = FakeAdapter(tmp_path)
    ft = FakeTasks()
    ts = TrainService(fa, ft, ConfigService(fa, datasets_dir=tmp_path / "ds"))
    monkeypatch.setattr(app_module, "adapter", fa)
    monkeypatch.setattr(app_module, "task_manager", ft)
    monkeypatch.setattr(app_module, "train_service", ts)
    monkeypatch.setattr(app_module, "gpu_monitor", FakeGpu())
    app_module._task_status_seen.clear()
    return fa, ft, ts


def test_start_rejects_data_companion(env):
    """配套 _data.py 不是可训练实体:选到它必须拒绝,不能把数据配置传给 moss.py。"""
    fa, ft, ts = env
    (fa.repo_root / "configs" / "user" / "demo_data.py").write_text("X = 1\n", encoding="utf-8")
    res = ts.start("demo_data")
    assert not res["ok"] and "_data" in res["error"]
    assert ft.list_tasks() == []  # 未误启动任何任务


def test_start_builds_argv_and_locates_run_dir(env):
    fa, ft, ts = env
    res = ts.start("demo", {"LR": "2e-4", "STEPS": "30"})
    assert res["ok"]
    task = res["task"]
    assert task["argv"] == [rf"{fa.repo_root}\.venv\Scripts\kogine.exe", "run",
                            "scripts/train/moss.py", "--config", "configs/user/demo.py",
                            "--set", "LR=2e-4", "--set", "STEPS=30"]
    # RUN_NAME 从配置文本解析,run_dir 预定位
    assert task["run_dir"].replace("\\", "/").endswith("runs/moss/demo-run")
    assert task["kind"] == "train"


def test_start_prefers_engine_printed_script(env):
    """冒烟实测:仓库 scripts/ 与安装包新版不兼容时,以配置 docstring 打印的
    `kogine run <bundle>/scripts/train/moss.py --config ...` 路径为准。"""
    fa, ft, ts = env
    (fa.repo_root / "configs" / "user" / "printed.py").write_text(
        CONFIG_TEXT.replace("demo-run", "printed-run") +
        '\n    kogine run D:/x/site-packages/kohakumoss/bundle/scripts/train/'
        'moss.py --config D:/x/configs/user/printed.py\n',
        encoding="utf-8")
    res = ts.start("printed")
    assert res["ok"]
    assert res["task"]["argv"][2] == (
        "D:/x/site-packages/kohakumoss/bundle/scripts/train/moss.py")
    # 无打印命令的手写配置回退仓库相对路径(先停掉占用 GPU 的上一任务)
    ts.stop(res["task"]["id"])
    assert ts.start("demo")["task"]["argv"][2] == "scripts/train/moss.py"


def test_start_existing_run_requires_resume(env):
    fa, ft, ts = env
    run = fa.repo_root / "runs" / "moss" / "demo-run"
    run.mkdir(parents=True)
    (run / "metrics.jsonl").write_text('{"step": 10}\n', encoding="utf-8")
    res = ts.start("demo")
    assert not res["ok"] and res["run_exists"]
    res2 = ts.start("demo", resume=True)
    assert res2["ok"]
    assert res2["task"]["argv"][-2:] == ["--set", "RESUME=latest"]


def test_start_gpu_busy_rejected(env):
    fa, ft, ts = env
    ft.add({"id": "busy", "kind": "cache_text", "status": "running", "argv": [],
            "overrides": {}, "run_dir": None, "log_path": None, "pid": 1})
    res = ts.start("demo")
    assert not res["ok"] and "GPU" in res["error"]


def test_start_missing_config_and_engine(env):
    fa, ft, ts = env
    assert not ts.start("ghost")["ok"]
    assert not ts.start("../evil")["ok"]  # 非法名
    fa2 = FakeAdapter(fa.repo_root)
    fa2.find_executable = lambda name: None
    ts2 = TrainService(fa2, ft, ConfigService(fa2, datasets_dir=fa2.repo_root / "ds"))
    assert ts2.start("demo").get("engine_missing")


def test_stop_and_resume(env):
    fa, ft, ts = env
    task = ts.start("demo")["task"]
    stopped = ts.stop(task["id"])
    assert stopped["ok"] and stopped["task"]["status"] == "killed"
    assert ft.terminations == [task["id"]]
    resumed = ts.resume(task["id"])
    assert resumed["ok"]
    assert resumed["task"]["argv"][-2:] == ["--set", "RESUME=latest"]
    assert not ts.stop("nope")["ok"]
    assert not ts.resume("nope")["ok"]


def test_status_snapshot(env):
    fa, ft, ts = env
    assert ts.status()["phase"] == "idle"
    task = ts.start("demo")["task"]
    run = Path(task["run_dir"])
    run.mkdir(parents=True)
    (run / "config.json").write_text(json.dumps({"STEPS": 30, "SAMPLE_EVERY": 0,
                                                 "VALIDATE_EVERY": 0}),
                                     encoding="utf-8")
    (run / "metrics.jsonl").write_text(
        '{"step": 10, "time": 1000.0, "loss": 0.5}\n'
        '{"step": 20, "time": 1010.0, "loss": 0.4, "val_loss": 0.6}\n', encoding="utf-8")
    st = ts.status()
    assert st["phase"] == "running"
    assert st["step"] == 20 and st["total_steps"] == 30
    assert st["sample_every"] == 0 and st["validate_every"] == 0
    assert len(st["metrics"]) == 2
    assert st["metrics"][1]["val_loss"] == 0.6
    assert st["eta_seconds"] is not None


def test_phase_initializing_without_metrics(env):
    fa, ft, ts = env
    task = ts.start("demo")["task"]
    assert ts.status()["phase"] == "initializing"  # 进程活、产物未出
    Path(task["run_dir"]).mkdir(parents=True)
    (Path(task["run_dir"]) / "metrics.jsonl").write_text('{"step": 0, "loss": 1}\n',
                                                         encoding="utf-8")
    assert ts.status()["phase"] == "running"


def test_inspect_endpoint(env, monkeypatch):
    fa, ft, ts = env
    report = {"catalog": {"images": 6}, "plan": {"buckets": []}, "captions": ["a"]}
    fa.procs.append(subprocess.CompletedProcess(
        [], 0, stdout="",
        stderr="2026 INFO scripts.inspect_data: " + json.dumps(report) + "\n"))
    res = ts.inspect("demo")
    assert res["ok"] and res["report"] == report
    assert "scripts/data/inspect_data.py" in fa.last_run_argv
    # 非法/缺失配置
    assert not ts.inspect("ghost")["ok"]


def test_inspect_prefers_data_config(env):
    fa, ft, ts = env
    data_cfg = fa.repo_root / "configs" / "user" / "demo_data.py"
    data_cfg.write_text("use_config('./demo.py')\n", encoding="utf-8")
    fa.procs.append(subprocess.CompletedProcess([], 0, stdout='{"catalog": {}}',
                                                stderr=""))
    res = ts.inspect("demo")
    assert res["ok"] and res["config"] == "configs/user/demo_data.py"


def test_inspect_unparsable_output(env):
    fa, ft, ts = env
    fa.procs.append(subprocess.CompletedProcess([], 0, stdout="no json", stderr=""))
    res = ts.inspect("demo")
    assert not res["ok"] and res["error"] == "输出中未解析到 JSON"


def test_cache_tasks(env):
    fa, ft, ts = env
    assert not ts.start_cache("demo", "bogus")["ok"]
    # 无配套数据配置:拒绝(文案面向用户,不出现内部文件名)
    r = ts.start_cache("demo", "cache_text")
    assert not r["ok"] and "数据配置" in r["error"]
    (fa.repo_root / "configs" / "user" / "demo_data.py").write_text(
        "use_config('./demo.py')\n", encoding="utf-8")
    r = ts.start_cache("demo", "cache_latents")
    assert r["ok"]
    assert r["task"]["kind"] == "cache_latents"
    assert r["task"]["argv"][2] == "scripts/data/cache_latents.py"
    assert r["task"]["argv"][4] == "configs/user/demo_data.py"


# ---------- 离线缓存缺口 -> 自动排队前置任务(M4) ----------

def _cfg_with_dataset(repo_root, name, ds_path):
    """写一份含 DATASETS 的配置(路径指向 ds_path)。"""
    (repo_root / "configs" / "user" / f"{name}.py").write_text(
        f'"""demo."""\nRUN_NAME = "{name}-run"\nSTEPS = 30\n'
        f'DATASETS = [{{"path": "{ds_path}", "source": "folder"}}]\n',
        encoding="utf-8")


def test_start_queues_missing_caches(env):
    """数据集缺 text.db / latents.db:训练前置排队两个缓存任务,链式 after。"""
    fa, ft, ts = env
    ds = (fa.repo_root / "ds1").as_posix()
    _cfg_with_dataset(fa.repo_root, "q", ds)
    (fa.repo_root / "configs" / "user" / "q_data.py").write_text(
        "use_config('./q.py')\n", encoding="utf-8")
    res = ts.start("q")
    assert res["ok"] and res["queued"]
    assert res["prep"] == ["cache_text", "cache_latents"]
    assert res["task"]["status"] == "pending"
    rows = {t["id"]: t for t in ft.list_tasks()}
    # 缓存文本先跑(running)-> 缓存图像 pending 等它 -> 训练 pending 等图像
    ct = next(t for t in rows.values() if t["kind"] == "cache_text")
    cl = next(t for t in rows.values() if t["kind"] == "cache_latents")
    tr = res["task"]
    assert ct["status"] == "running" and ct["after_task"] is None
    assert cl["status"] == "pending" and cl["after_task"] == ct["id"]
    assert tr["status"] == "pending" and tr["after_task"] == cl["id"]


def test_start_skips_ready_caches(env):
    """text.db + latents.db 齐全:不排队,训练直接 running。"""
    fa, ft, ts = env
    ds = fa.repo_root / "ds2"
    (ds / "cache").mkdir(parents=True)
    (ds / "cache" / "text.db").write_text("x", encoding="utf-8")
    (ds / "cache" / "latents.db").write_text("x", encoding="utf-8")
    _cfg_with_dataset(fa.repo_root, "ready", ds.as_posix())
    res = ts.start("ready")
    assert res["ok"] and not res.get("queued")
    assert res["task"]["status"] == "running"


def test_start_queue_needs_data_config(env):
    """缺配套数据配置时排队失败(不启动半条链)。"""
    fa, ft, ts = env
    ds = (fa.repo_root / "ds3").as_posix()
    _cfg_with_dataset(fa.repo_root, "nodata", ds)
    res = ts.start("nodata")
    assert not res["ok"] and "数据配置" in res["error"]
    assert ft.list_tasks() == []


# ---------- Windows 预览防崩组合注入(M4) ----------

def test_win_preview_injected(env, monkeypatch):
    """Windows + SAMPLE_EVERY>0:注入 compute + 不锁页,用户未设的键被补齐。"""
    fa, ft, ts = env
    monkeypatch.setattr(ts_mod.sys, "platform", "win32")
    (fa.repo_root / "configs" / "user" / "prev.py").write_text(
        '"""demo."""\nRUN_NAME = "prev-run"\nSTEPS = 30\nSAMPLE_EVERY = 50\n',
        encoding="utf-8")
    res = ts.start("prev")
    assert res["ok"] and res["preview_workaround"]
    ov = res["task"]["overrides"]
    assert ov["BASE_PRECISION"] == "compute"
    assert ov["MEMORY_OVERRIDES"] == "{'pinned': False}"
    assert "KERNELS" not in ov  # 钉扎无回退,cpu 侧绑定会炸,禁止注入
    assert "--set" in res["task"]["argv"]
    assert "Windows" in res["note"]


def test_win_preview_off_no_injection(env, monkeypatch):
    """SAMPLE_EVERY=0(预览关闭)不注入。"""
    fa, ft, ts = env
    monkeypatch.setattr(ts_mod.sys, "platform", "win32")
    (fa.repo_root / "configs" / "user" / "off.py").write_text(
        '"""demo."""\nRUN_NAME = "off-run"\nSTEPS = 30\nSAMPLE_EVERY = 0\n',
        encoding="utf-8")
    res = ts.start("off")
    assert res["ok"] and not res.get("preview_workaround")
    assert "BASE_PRECISION" not in res["task"]["overrides"]


def test_win_preview_respects_user_override(env, monkeypatch):
    """用户高级设置已含 BASE_PRECISION:整组不覆盖(尊重显式选择)。"""
    fa, ft, ts = env
    monkeypatch.setattr(ts_mod.sys, "platform", "win32")
    (fa.repo_root / "configs" / "user" / "usr.py").write_text(
        '"""demo."""\nRUN_NAME = "usr-run"\nSTEPS = 30\nSAMPLE_EVERY = 50\n',
        encoding="utf-8")
    res = ts.start("usr", {"BASE_PRECISION": "int8"})
    assert res["ok"] and not res.get("preview_workaround")
    assert res["task"]["overrides"]["BASE_PRECISION"] == "int8"


def test_linux_no_preview_injection(env, monkeypatch):
    """Linux 即使开预览也不注入(引擎无此崩溃)。"""
    fa, ft, ts = env
    monkeypatch.setattr(ts_mod.sys, "platform", "linux")
    (fa.repo_root / "configs" / "user" / "lnx.py").write_text(
        '"""demo."""\nRUN_NAME = "lnx-run"\nSTEPS = 30\nSAMPLE_EVERY = 50\n',
        encoding="utf-8")
    res = ts.start("lnx")
    assert res["ok"] and not res.get("preview_workaround")
    assert "BASE_PRECISION" not in res["task"]["overrides"]


def test_stop_pending_cancels_upstream_chain(env):
    """停止排队中的训练:沿依赖链向上取消未开始/运行中的前置任务。"""
    fa, ft, ts = env
    ct = ft.start("cache_text", ["x"])
    cl = ft.start("cache_latents", ["x"], after=ct["id"])
    tr = ft.start("train", ["x"], after=cl["id"])
    res = ts.stop(tr["id"])
    assert res["ok"]
    # 三个都被终止(训练 + 两级缓存)
    assert set(ft.terminations) >= {tr["id"], cl["id"], ct["id"]}


def test_http_endpoints(env):
    fa, ft, ts = env
    client = TestClient(app_module.app)
    # 409:run 已存在
    run = fa.repo_root / "runs" / "moss" / "demo-run"
    run.mkdir(parents=True)
    (run / "checkpoints" / "step_00000010").mkdir(parents=True)
    res = client.post("/api/train/start", json={"config": "demo"})
    assert res.status_code == 409
    res = client.post("/api/train/start", json={"config": "demo", "resume": True})
    assert res.status_code == 200 and res.json()["ok"]
    tid = res.json()["task"]["id"]
    # 面板按配置过滤:未选/选错配置 = idle,选中所属配置才见任务
    assert client.get("/api/train/status").json()["phase"] == "idle"
    assert client.get("/api/train/status", params={"config": "other"}).json()["phase"] == "idle"
    assert client.get("/api/train/status", params={"config": "demo"}).json()["phase"] == "initializing"
    assert client.get("/api/tasks").json()["tasks"]
    assert client.post(f"/api/train/{tid}/stop").json()["ok"]
    assert client.post(f"/api/train/{tid}/resume").json()["ok"]
    assert not client.post("/api/train/ghost/stop").json()["ok"]


def test_run_files_http_whitelist(env):
    fa, ft, ts = env
    client = TestClient(app_module.app)
    img = fa.repo_root / "runs" / "moss" / "demo-run" / "samples" / "step_00000010"
    img.mkdir(parents=True)
    png = img / "000.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    res = client.get("/api/run-files", params={"path": str(png)})
    assert res.status_code == 200 and res.content.startswith(b"\x89PNG")
    # 穿越 -> 403
    res = client.get("/api/run-files",
                     params={"path": "runs/moss/demo-run/samples/../../../../.venv/x.png"})
    assert res.status_code == 403
    # checkpoints 下 -> 403
    res = client.get("/api/run-files",
                     params={"path": "runs/moss/demo-run/samples/../checkpoints/x.png"})
    assert res.status_code == 403
    # 不存在 -> 404
    assert client.get("/api/run-files", params={"path": "runs/moss/a/samples/x.png"}
                      ).status_code == 404


def test_ws_receives_events(env):
    """WS 集成:lifespan 监控循环把产物增量推给客户端。"""
    fa, ft, ts = env
    task = ts.start("demo")["task"]
    run = Path(task["run_dir"])
    run.mkdir(parents=True)
    task["log_path"] = str(run / "fallback.log")
    with TestClient(app_module.app) as client:
        with client.websocket_connect("/ws") as ws:
            hello = ws.receive_json()
            assert hello["type"] == "hello"
            # 产物在连接后写入,等待 tick 推送
            with open(run / "metrics.jsonl", "a", encoding="utf-8") as f:
                f.write('{"step": 10, "time": 1.0, "loss": 0.5}\n')
            with open(run / "fallback.log", "a", encoding="utf-8") as f:
                f.write("prewarm: kernels\n")
            # 注:task_status(running) 可能在 ws 连接前的首个 tick 已广播(无观察者),
            # 前端进入页面时以 REST 快照补齐;task_status 事件本身由第二段 killed 断言覆盖
            got_metric = got_log = False
            deadline = time.time() + 20
            while time.time() < deadline and not (got_metric and got_log):
                ev = ws.receive_json()
                t = ev.get("type")
                if t == "metric" and ev["data"]["step"] == 10:
                    got_metric = True
                elif t == "logline" and "prewarm" in ev["line"]:
                    got_log = True
            assert got_metric and got_log
            # 停止任务 -> task_status killed 事件
            ts.stop(task["id"])
            deadline = time.time() + 20
            while time.time() < deadline:
                ev = ws.receive_json()
                if ev.get("type") == "task_status" and ev["task"]["status"] == "killed":
                    break
            else:
                pytest.fail("未收到 killed 状态事件")
