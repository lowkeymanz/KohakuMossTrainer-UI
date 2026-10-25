"""M3 导出与采样测试:假 adapter/TaskManager,不依赖真引擎。

覆盖:run_service 的 checkpoint 浏览/find_checkpoint_dir/list_runs/导出白名单、
export_service 的格式运行时探测与导出 argv 构造、sample_service 的
new sample 链路/网格 --set 构造/图库与白名单、M3 HTTP 端点语义。
"""

import json
import os
import subprocess
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app as app_module
import sample_service
from config_service import ConfigService
from export_service import ExportService
from sample_service import SampleService

# 模拟 `new sample` 生成的配置:docstring 打印权威运行命令(M1 实测契约)
SAMPLE_CFG_TEXT = '''"""demo sample."""
from kohakuengine import use_config

use_config("../moss/_base/base_sample.py")

WEIGHTS = "D:/base/model"
OUT = "D:/gallery/demo"
ADAPTERS = ["D:/ckpt/trainable.safetensors:1.0"]
PROMPTS = ["a cat"]

    kogine run D:/x/site-packages/kohakumoss/bundle/scripts/sample/sample.py --config D:/x/configs/user/demo.py
'''

BUNDLE_EXPORT_TEXT = '''"""export bundle script."""
FORMAT = "comfy"  # "comfy" | "lycoris" | "merged" | "diffusers"
'''

CLI_ALIASES_TEXT = '''"""engine export cli."""
MERGED = {"merged": "merged"}
ALIASES = {"hf": ("lycoris", "diffusers"), "comfy": ("comfy", "merged")}
'''


class FakeAdapter:
    def __init__(self, repo_root, has_kogine=True):
        self.repo_root = Path(repo_root)
        self.has_kogine = has_kogine
        self.bundle = {}  # rel -> Path(模拟已安装 bundle 束脚本)
        self.last_run_argv = None
        self.sample_cfg_text = SAMPLE_CFG_TEXT

    def find_executable(self, name):
        if not self.has_kogine:
            return None
        if name in ("kogine", "kohakumoss-config"):
            return Path(rf"{self.repo_root}\.venv\Scripts\{name}.exe")
        return None

    def find_bundle_script(self, rel):
        return self.bundle.get(rel)

    def _run(self, argv, timeout=30):
        argv = [str(a) for a in argv]
        self.last_run_argv = argv
        # 模拟引擎 `new sample`:按 --out 写出生成的配置文件
        if argv[1:3] == ["new", "sample"]:
            out = Path(argv[argv.index("--out") + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(self.sample_cfg_text, encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")


class FakeTasks:
    def __init__(self):
        self.rows = {}

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
        self.rows[tid]["status"] = "killed"
        return self.rows[tid]


def make_run(repo_root, run="demo-run", steps=(20, 40), weights="D:\\base\\model.safetensors",
             prompts=("a cat", "a dog")):
    """造 runs/moss/<run>:checkpoint 四件套 + config.json 快照。"""
    run_dir = Path(repo_root) / "runs" / "moss" / run
    run_dir.mkdir(parents=True, exist_ok=True)
    for s in steps:
        d = run_dir / "checkpoints" / f"step_{s:08d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "trainable.safetensors").write_bytes(b"t")
        (d / "optim.pt").write_bytes(b"o")
        (d / "state.json").write_text("{}", encoding="utf-8")
    cfg = {"WEIGHTS": weights, "SAMPLE_PROMPTS": list(prompts),
           "STEPS": max(steps) if steps else None, "ALGO": "lora"}
    (run_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return run_dir


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """tmp 仓库根;假引擎与假任务表;图库根重定向到 tmp(不污染 ui/state)。"""
    (tmp_path / "configs" / "user").mkdir(parents=True)
    fa = FakeAdapter(tmp_path)
    ft = FakeTasks()
    cs = ConfigService(fa, datasets_dir=tmp_path / "ds")
    es = ExportService(fa, ft)
    ss = SampleService(fa, ft, cs)
    monkeypatch.setattr(app_module, "adapter", fa)
    monkeypatch.setattr(app_module, "task_manager", ft)
    monkeypatch.setattr(app_module, "config_service", cs)
    monkeypatch.setattr(app_module, "export_service", es)
    monkeypatch.setattr(app_module, "sample_service", ss)
    monkeypatch.setattr(sample_service, "GALLERY_ROOT", tmp_path / "gallery")
    monkeypatch.setattr(sample_service, "GRID_ROOT", tmp_path / "grids")
    # argv 断言跨平台确定:Windows 崩溃绕开分支由专项测试单独验证
    monkeypatch.setattr(sample_service, "_WIN_SAMPLING_WORKAROUND", False)
    app_module._task_status_seen.clear()
    return fa, ft, es, ss


# ---------- run_service:M3 浏览辅助 ----------

def test_list_checkpoints_flags(tmp_path):
    import run_service
    run_dir = make_run(tmp_path, steps=(20, 40))
    # 30 步:缺 optim -> complete=False usable=True;缺 ema
    d30 = run_dir / "checkpoints" / "step_00000030"
    d30.mkdir(parents=True)
    (d30 / "trainable.safetensors").write_bytes(b"t")
    (d30 / "state.json").write_text("{}", encoding="utf-8")
    # 40 步:补 ema + export 产物(文件 + diffusers 目录)
    d40 = run_dir / "checkpoints" / "step_00000040"
    (d40 / "ema.safetensors").write_bytes(b"e")
    exp = d40 / "export"
    exp.mkdir()
    (exp / "demo-run-comfy.safetensors").write_bytes(b"s" * 7)
    (exp / "demo-run-diffusers").mkdir()
    res = run_service.list_checkpoints(tmp_path, "demo-run")
    assert res["ok"]
    ck = {c["dir"]: c for c in res["checkpoints"]}
    assert [c["dir"] for c in res["checkpoints"]] == [  # 步数升序
        "step_00000020", "step_00000030", "step_00000040"]
    assert ck["step_00000020"]["complete"] and ck["step_00000020"]["usable"]
    assert not ck["step_00000020"]["has_ema"]
    assert ck["step_00000030"]["usable"] and not ck["step_00000030"]["complete"]
    assert ck["step_00000040"]["has_ema"]
    assert ck["step_00000040"]["files"]["trainable.safetensors"]["size"] == 1
    exports = {e["name"]: e for e in ck["step_00000040"]["exports"]}
    assert exports["demo-run-comfy.safetensors"]["size"] == 7
    assert exports["demo-run-diffusers"]["folder"] is True
    ghost = run_service.list_checkpoints(tmp_path, "ghost")
    assert not ghost["ok"] and ghost["not_found"]


def test_find_checkpoint_dir(tmp_path):
    import run_service
    run_dir = make_run(tmp_path, steps=(20, 40))
    assert run_service.find_checkpoint_dir(run_dir, "step_00000020").name == "step_00000020"
    assert run_service.find_checkpoint_dir(run_dir, "40").name == "step_00000040"
    assert run_service.find_checkpoint_dir(run_dir, None).name == "step_00000040"
    # 最新步未写完(无 state.json)-> 回退上一个 usable
    (run_dir / "checkpoints" / "step_00000040" / "state.json").unlink()
    assert run_service.find_checkpoint_dir(run_dir, None).name == "step_00000020"
    assert run_service.find_checkpoint_dir(run_dir, "999") is None
    assert run_service.find_checkpoint_dir(run_dir, "step_00000099") is None
    assert run_service.find_checkpoint_dir(tmp_path / "nope", None) is None


def test_list_runs_summary(tmp_path):
    import run_service
    run_dir = make_run(tmp_path, run="a", steps=(40,))
    (run_dir / "export").mkdir()
    (run_dir / "export" / "a-comfy.safetensors").write_bytes(b"x")
    (run_dir / "samples").mkdir()
    (run_dir / "samples" / "0.png").write_bytes(b"\x89PNG")
    (run_dir / "samples" / "note.txt").write_text("x", encoding="utf-8")
    run_b = make_run(tmp_path, run="b", steps=())
    t = time.time()
    os.utime(run_dir, (t - 100, t - 100))  # b 更新 -> 倒序排前
    os.utime(run_b, (t, t))
    res = run_service.list_runs(tmp_path)
    assert res["ok"] and [r["run"] for r in res["runs"]] == ["b", "a"]
    ra = next(r for r in res["runs"] if r["run"] == "a")
    assert ra["checkpoint_count"] == 1 and ra["last_step"] == 40
    assert ra["export_count"] == 1 and ra["sample_count"] == 1
    assert ra["algo"] == "lora" and ra["weights"] == "D:\\base\\model.safetensors"
    empty = run_service.list_runs(tmp_path / "no-runs-here")
    assert empty["ok"] and empty["runs"] == []


def test_resolve_export_file(tmp_path):
    import run_service
    run_dir = make_run(tmp_path, steps=(40,))
    exp = run_dir / "checkpoints" / "step_00000040" / "export"
    exp.mkdir()
    f = exp / "demo-run-comfy.safetensors"
    f.write_bytes(b"z")
    assert run_service.resolve_export_file(tmp_path, str(f))["ok"]
    run_exp = run_dir / "export"
    run_exp.mkdir()
    g = run_exp / "demo.safetensors"
    g.write_bytes(b"z")
    assert run_service.resolve_export_file(tmp_path, "runs/moss/demo-run/export/demo.safetensors")["ok"]
    # 越出 runs/moss / 非 export 子树 -> forbidden
    assert run_service.resolve_export_file(
        tmp_path, "runs/moss/demo-run/config.json")["forbidden"]
    assert run_service.resolve_export_file(
        tmp_path, "configs/user/demo.py")["forbidden"]
    assert run_service.resolve_export_file(
        tmp_path, "runs/moss/demo-run/export/../../config.json")["forbidden"]
    # 缺失 / 目录(diffusers 文件夹不可下载) -> not_found
    assert run_service.resolve_export_file(tmp_path, str(exp / "nope.safetensors"))["not_found"]
    d = exp / "demo-run-diffusers"
    d.mkdir()
    assert run_service.resolve_export_file(tmp_path, str(d))["not_found"]
    assert not run_service.resolve_export_file(tmp_path, "")["ok"]


# ---------- export_service ----------

def test_formats_probe_bundle_plus_aliases(tmp_path):
    fa = FakeAdapter(tmp_path)
    bundle = tmp_path / "bundle" / "export.py"
    bundle.parent.mkdir(parents=True)
    bundle.write_text(BUNDLE_EXPORT_TEXT, encoding="utf-8")
    fa.bundle["export/export.py"] = bundle
    cli = tmp_path / "src" / "kohakumoss" / "adapters" / "export" / "cli.py"
    cli.parent.mkdir(parents=True)
    cli.write_text(CLI_ALIASES_TEXT, encoding="utf-8")
    res = ExportService(fa, FakeTasks()).formats()
    assert res["ok"] and res["probed"]
    values = [f["value"] for f in res["formats"]]
    assert values == ["comfy", "lycoris", "merged", "diffusers", "hf"]
    assert res["formats"][0]["label"].startswith("ComfyUI LoRA")


def test_formats_fallback_unprobed(tmp_path):
    res = ExportService(FakeAdapter(tmp_path), FakeTasks()).formats()
    assert res["ok"] and not res["probed"]
    assert [f["value"] for f in res["formats"]] == [
        "comfy", "lycoris", "hf", "merged", "diffusers"]


def test_export_start_builds_argv(env):
    fa, ft, es, ss = env
    run_dir = make_run(fa.repo_root, steps=(40,))
    ckpt = run_dir / "checkpoints" / "step_00000040"
    res = es.start({"run": "demo-run", "checkpoint": "step_00000040",
                    "format": "comfy", "multiplier": "0.8", "save_dtype": "bf16"})
    assert res["ok"]
    task = res["task"]
    assert task["kind"] == "export" and task["config_path"] is None
    assert Path(task["run_dir"]) == run_dir
    ov = task["overrides"]
    assert ov["WEIGHTS"] == "D:/base/model.safetensors"  # 反斜杠 -> posix
    assert ov["ADAPTER"] == (ckpt / "trainable.safetensors").as_posix()
    assert ov["OUT"] == (ckpt / "export" / "demo-run-comfy.safetensors").as_posix()
    assert ov["FORMAT"] == "comfy" and ov["MULTIPLIER"] == "0.8"
    assert ov["SAVE_DTYPE"] == "bf16"
    # bundle 探测不到 -> 回退仓库相对脚本路径;--set 对与 overrides 一一对应
    assert task["argv"][:3] == [str(fa.find_executable("kogine")), "run",
                                "scripts/export/export.py"]
    assert task["argv"][3:] == sum([["--set", f"{k}={v}"] for k, v in ov.items()], [])
    # bundle 存在时优先束脚本(先清掉上一 running 任务,避免 GPU 互斥拒绝)
    ft.rows.clear()
    bundle = fa.repo_root / "bundle" / "export.py"
    bundle.parent.mkdir(parents=True, exist_ok=True)
    bundle.write_text(BUNDLE_EXPORT_TEXT, encoding="utf-8")
    fa.bundle["export/export.py"] = bundle
    (fa.repo_root / "runs" / "moss" / "demo-run" / "checkpoints" /
     "step_00000040" / "state.json").write_text("{}", encoding="utf-8")
    res2 = es.start({"run": "demo-run", "format": "diffusers"})
    assert res2["ok"]
    assert res2["task"]["argv"][2] == bundle.as_posix()
    # diffusers 是目录:OUT 不带扩展名
    assert res2["out"] == (ckpt / "export" / "demo-run-diffusers").as_posix()


def test_export_start_rejections(env):
    fa, ft, es, ss = env
    run_dir = make_run(fa.repo_root, steps=(40,))
    assert not es.start({})["ok"]  # run 为空
    assert not es.start({"run": "ghost", "format": "comfy"})["ok"]
    assert not es.start({"run": "demo-run", "checkpoint": "step_00000099",
                         "format": "comfy"})["ok"]
    assert not es.start({"run": "demo-run", "format": "bogus"})["ok"]
    assert not es.start({"run": "demo-run", "format": "comfy",
                         "multiplier": "abc"})["ok"]
    assert not es.start({"run": "demo-run", "format": "comfy",
                         "save_dtype": "a b"})["ok"]
    # 缺 trainable.safetensors
    (run_dir / "checkpoints" / "step_00000040" / "trainable.safetensors").unlink()
    assert not es.start({"run": "demo-run", "format": "comfy"})["ok"]
    # GPU 被训练占用
    make_run(fa.repo_root, run="other", steps=(10,))
    ft.add({"id": "busy", "kind": "train", "status": "running", "argv": [],
            "overrides": {}, "run_dir": None, "log_path": None, "pid": 1})
    res = es.start({"run": "other", "format": "comfy"})
    assert not res["ok"] and "GPU" in res["error"]
    # 引擎缺失
    es2 = ExportService(FakeAdapter(fa.repo_root, has_kogine=False), ft)
    assert es2.start({"run": "other", "format": "comfy"}).get("engine_missing")


# ---------- sample_service ----------

def test_sample_start_builds_argv(env):
    fa, ft, es, ss = env
    run_dir = make_run(fa.repo_root, steps=(40,))
    ckpt = run_dir / "checkpoints" / "step_00000040"
    res = ss.start({"run": "demo-run", "vram": 8, "steps": 8})
    assert res["ok"], res
    # 配置名默认 <run>-s<step>-sample;prompts 缺省用 run 的 SAMPLE_PROMPTS
    assert res["name"] == "demo-run-s40-sample"
    assert res["prompts"] == ["a cat", "a dog"]
    task = res["task"]
    assert task["kind"] == "sample"
    assert task["config_path"] == "configs/user/demo-run-s40-sample.py"
    assert Path(task["run_dir"]) == sample_service.GALLERY_ROOT / res["name"]
    # 运行脚本以生成配置 docstring 打印的 bundle 路径为准
    assert task["argv"] == [str(fa.find_executable("kogine")), "run",
                            "D:/x/site-packages/kohakumoss/bundle/scripts/sample/sample.py",
                            "--config", "configs/user/demo-run-s40-sample.py",
                            "--set", "STEPS=8"]
    # new sample 收到的表单:adapter=path:mult、samples 为 posix 图库目录
    assert fa.last_run_argv[1:3] == ["new", "sample"]
    a = fa.last_run_argv[fa.last_run_argv.index("--adapter") + 1]
    assert a == f"{(ckpt / 'trainable.safetensors').as_posix()}:1.0"
    assert fa.last_run_argv[fa.last_run_argv.index("--samples") + 1] == \
        (sample_service.GALLERY_ROOT / res["name"]).as_posix()
    assert task["overrides"]["ADAPTER"] == a
    assert task["overrides"]["PROMPTS"] == ["a cat", "a dog"]


def test_sample_start_errors(env):
    fa, ft, es, ss = env
    make_run(fa.repo_root, steps=(40,))
    assert ss.start({"run": "demo-run"}).get("error")  # 无 vram
    assert not ss.start({"run": "demo-run", "vram": 0})["ok"]
    assert not ss.start({"run": "ghost", "vram": 8})["ok"]
    # 无 prompts 且 run 配置未记录 SAMPLE_PROMPTS
    run2 = make_run(fa.repo_root, run="nop", steps=(5,))
    (run2 / "config.json").write_text(json.dumps({"WEIGHTS": "W"}), encoding="utf-8")
    assert not ss.start({"run": "nop", "vram": 8})["ok"]
    # 非法任务名
    assert not ss.start({"run": "demo-run", "vram": 8, "name": "../evil"})["ok"]
    # GPU 占用:不进 new sample
    ft.add({"id": "busy", "kind": "sample", "status": "running", "argv": [],
            "overrides": {}, "run_dir": None, "log_path": None, "pid": 1})
    res = ss.start({"run": "demo-run", "vram": 8, "prompts": "x"})
    assert not res["ok"] and "GPU" in res["error"]
    ft.rows.pop("busy")
    # 非法 steps
    assert not ss.start({"run": "demo-run", "vram": 8, "steps": "abc"})["ok"]
    # 引擎缺失
    ss2 = SampleService(FakeAdapter(fa.repo_root, has_kogine=False), ft,
                        ConfigService(FakeAdapter(fa.repo_root, has_kogine=False)))
    assert ss2.start({"run": "demo-run", "vram": 8}).get("engine_missing")


def test_grid_start_argv(env):
    fa, ft, es, ss = env
    run_dir = make_run(fa.repo_root, steps=(20, 40))
    p20 = (run_dir / "checkpoints" / "step_00000020" /
           "trainable.safetensors").as_posix()
    p40 = (run_dir / "checkpoints" / "step_00000040" /
           "trainable.safetensors").as_posix()
    res = ss.start_grid({"run": "demo-run", "checkpoints": ["step_00000020", "40"],
                         "prompts": ["a cat"], "steps": 40, "cfg": 7.5,
                         "size": "1024x1024", "precision": "fp16",
                         "memory_profile": "lora_8g"})
    assert res["ok"], res
    task = res["task"]
    assert task["kind"] == "grid"
    assert task["argv"][:3] == [str(fa.find_executable("kogine")), "run",
                                "scripts/sample/compare_grid.py"]  # 无 bundle -> 仓库路径
    pairs = task["argv"][4::2]  # 每个 --set 的 K=V 值
    assert task["argv"][3::2] == ["--set"] * len(pairs)
    assert f"CHECKPOINTS={json.dumps([p20, p40])}" in pairs  # JSON 方括号列表语法
    assert f"PROMPTS={json.dumps(['a cat'])}" in pairs
    assert "WEIGHTS=D:/base/model.safetensors" in pairs
    assert f"OUT={(sample_service.GRID_ROOT / 'demo-run-grid').as_posix()}" in pairs
    assert "STEPS=40" in pairs and "CFG=7.5" in pairs
    assert f"SIZES={json.dumps(['1024x1024'])}" in pairs
    assert "PRECISION=fp16" in pairs and "MEMORY_PROFILE=lora_8g" in pairs
    # 校验分支
    assert not ss.start_grid({"run": "demo-run", "checkpoints": []})["ok"]
    assert not ss.start_grid({"run": "demo-run", "checkpoints": ["step_9"],
                              "prompts": ["x"]})["ok"]
    (run_dir / "checkpoints" / "step_00000020" / "trainable.safetensors").unlink()
    assert not ss.start_grid({"run": "demo-run", "checkpoints": ["step_00000020"],
                              "prompts": ["x"]})["ok"]


def test_windows_sampling_workaround(env, monkeypatch):
    """Windows 绕开分支(2026-10-25 实测验证组合):采样追加基座解压 + 不锁页;
    网格未显式选 profile 时按流式档 minimal 排队,显式选择不被覆盖。"""
    fa, ft, es, ss = env
    make_run(fa.repo_root, steps=(40,))
    monkeypatch.setattr(sample_service, "_WIN_SAMPLING_WORKAROUND", True)
    ov = "MEMORY_OVERRIDES={'pinned': False}"
    res = ss.start({"run": "demo-run", "vram": 8})
    assert res["ok"], res
    assert res["task"]["argv"][-4:] == ["--set", "BASE_PRECISION=compute", "--set", ov]
    ft.rows.clear()  # 释放 GPU 独占
    res2 = ss.start_grid({"run": "demo-run", "checkpoints": ["40"],
                          "prompts": ["a cat"]})
    assert res2["ok"], res2
    pairs = res2["task"]["argv"][4::2]
    assert ("MEMORY_PROFILE=minimal" in pairs and "BASE_PRECISION=compute" in pairs
            and ov in pairs)
    ft.rows.clear()
    res3 = ss.start_grid({"run": "demo-run", "checkpoints": ["40"],
                          "prompts": ["a cat"], "memory_profile": "low"})
    assert res3["ok"], res3
    pairs3 = res3["task"]["argv"][4::2]
    assert ("MEMORY_PROFILE=low" in pairs3 and "BASE_PRECISION=compute" in pairs3
            and ov in pairs3)


def test_gallery_list_and_resolve(env):
    fa, ft, es, ss = env
    # 图库根不存在时正常返回空列表
    assert ss.list_gallery() == {"ok": True, "groups": [], "error": None}
    g = sample_service.GALLERY_ROOT / "run-a-s40-sample"
    g.mkdir(parents=True)
    (g / "0_123.png").write_bytes(b"\x89PNG\r\n\x1a\nimg")
    (g / "metadata-0_123.json").write_text('{"seed": 123}', encoding="utf-8")
    gr = sample_service.GRID_ROOT / "run-a-grid"
    gr.mkdir(parents=True)
    (gr / "grid.png").write_bytes(b"\x89PNG")
    res = ss.list_gallery()
    assert res["ok"] and len(res["groups"]) == 2
    by = {x["name"]: x for x in res["groups"]}
    assert by["run-a-s40-sample"]["kind"] == "sample"
    assert by["run-a-s40-sample"]["images"][0]["name"] == "0_123.png"
    assert by["run-a-s40-sample"]["metadata"]["metadata-0_123.json"]["seed"] == 123
    assert by["run-a-grid"]["kind"] == "grid"
    # 白名单
    assert ss.resolve_gallery_file("run-a-s40-sample/0_123.png")["ok"]  # 相对
    assert ss.resolve_gallery_file(str(gr / "grid.png"))["ok"]  # 绝对
    assert ss.resolve_gallery_file("run-a-grid/../../evil.png")["forbidden"]
    assert ss.resolve_gallery_file("run-a-s40-sample/../../evil.png")["forbidden"]
    assert "图片" in ss.resolve_gallery_file("run-a-s40-sample/metadata-0_123.json")["error"]
    assert ss.resolve_gallery_file("run-a-s40-sample/nope.png")["not_found"]
    assert not ss.resolve_gallery_file("")["ok"]
    # 创建后两棵根都有分组
    assert len(ss.list_gallery()["groups"]) == 2


# ---------- M3 HTTP 端点 ----------

def test_history_and_checkpoints_http(env):
    fa, ft, es, ss = env
    make_run(fa.repo_root, steps=(40,))
    client = TestClient(app_module.app)
    res = client.get("/api/history")
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] and body["runs"][0]["run"] == "demo-run"
    assert isinstance(body["tasks"], list)
    res = client.get("/api/runs/demo-run/checkpoints")
    assert res.status_code == 200 and len(res.json()["checkpoints"]) == 1
    assert client.get("/api/runs/ghost/checkpoints").status_code == 404


def test_export_http(env, monkeypatch):
    fa, ft, es, ss = env
    run_dir = make_run(fa.repo_root, steps=(40,))
    client = TestClient(app_module.app)
    res = client.get("/api/export/formats")
    assert res.status_code == 200 and not res.json()["probed"]
    res = client.post("/api/export/start", json={"run": "demo-run",
                                                 "format": "comfy"})
    assert res.status_code == 200 and res.json()["ok"]
    # 引擎缺失 -> 503
    monkeypatch.setattr(app_module, "export_service",
                        ExportService(FakeAdapter(fa.repo_root, has_kogine=False), ft))
    assert client.post("/api/export/start", json={"run": "demo-run",
                                                  "format": "comfy"}).status_code == 503
    # 下载白名单
    exp = run_dir / "checkpoints" / "step_00000040" / "export"
    exp.mkdir()
    f = exp / "demo-run-comfy.safetensors"
    f.write_bytes(b"safetensors")
    monkeypatch.setattr(app_module, "export_service", es)
    res = client.get("/api/export-files", params={"path": str(f)})
    assert res.status_code == 200 and "attachment" in res.headers["content-disposition"]
    assert client.get("/api/export-files", params={"path": str(run_dir / "config.json")}
                      ).status_code == 403
    assert client.get("/api/export-files",
                      params={"path": "runs/moss/demo-run/export/../../runs/x"}
                      ).status_code == 403
    assert client.get("/api/export-files", params={"path": str(exp / "nope.safetensors")}
                      ).status_code == 404


def test_sample_grid_gallery_http(env):
    fa, ft, es, ss = env
    make_run(fa.repo_root, steps=(40,))
    client = TestClient(app_module.app)
    res = client.post("/api/sample/start", json={"run": "demo-run", "vram": 8})
    assert res.status_code == 200 and res.json()["ok"]
    ft.rows.clear()  # 释放 GPU 互斥,便于继续测 grid
    res = client.post("/api/grid/start", json={"run": "demo-run",
                                               "checkpoints": ["step_00000040"]})
    assert res.status_code == 200 and res.json()["ok"]
    g = sample_service.GALLERY_ROOT / "shot"
    g.mkdir(parents=True)
    (g / "0_1.png").write_bytes(b"\x89PNG\r\n\x1a\nx")
    res = client.get("/api/gallery")
    assert res.status_code == 200
    by = {x["name"]: x for x in res.json()["groups"]}
    assert by["shot"]["images"][0]["name"] == "0_1.png"
    assert by["demo-run-grid"]["kind"] == "grid"
    res = client.get("/api/gallery-files", params={"path": "shot/0_1.png"})
    assert res.status_code == 200 and res.content.startswith(b"\x89PNG")
    assert client.get("/api/gallery-files",
                      params={"path": "shot/../../evil.png"}).status_code == 403
    assert client.get("/api/gallery-files", params={"path": "shot/nope.png"}
                      ).status_code == 404
