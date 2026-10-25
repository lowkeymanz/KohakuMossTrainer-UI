"""高级设置链路测试(默认值解析 + overrides 持久化 + 双侧键校验)。

覆盖 rev2 §4:
- 资产根定位:结构探测(checkout)/ 产物通道(bundle 绝对路径 docstring);
- 默认值组合:script globals -> base -> data -> vram 分层覆盖与 layer_source,
  ast 字面量 + 数字表达式(1024*1024)+ 导入型枚举(profiles ORDER/TUNER MODES);
- check_override_keys:未知键拒绝、RENAMED 放行提示、ENCODER 字段等价提示;
- 端到端:new_train 校验+持久化 -> start 缺省读取 -> argv --set;
- 空覆盖清除残留;GET /api/config/defaults 与 /api/config/overrides 路由;
- 真实引擎资产金样(KNOWN 144 / 面板 43 / STEPS=4000):引擎改布局即红,兜底。
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app as app_module
from config_service import ConfigService
from defaults_service import (DefaultsService, GROUP_LABELS, PANEL,
                              override_path, save_overrides)
from train_service import TrainService
from test_contract_cli import FULL_TRAIN_FORM, NewTrainRecorder
from test_m3_export_sample import FakeAdapter, FakeTasks

KNOBS_TEXT = '''
from kohakumoss.encoders.knob_names import RENAMED_SAMPLE
from kohakumoss.memory.profiles import ORDER
from kohakumoss.precision.config import COMPUTE
from kohakumoss_kernels.tuning.tuner import MODES

KNOWN = frozenset("""
STEPS LR DIM TARGET_AREA MEMORY_PROFILE CAPTION ASPECT_RANGE SAVE_FORMAT
EXPORT_FORMATS SCHEDULE RESUME PRECISION ALGO SAMPLE_SIZE OPTIMIZER
""".split())
CHOICES = {
    "MEMORY_PROFILE": (*ORDER, "auto", "manual"),
    "TUNE": MODES,
    "PRECISION": COMPUTE + ("auto",),
}
ALGOS = ("lora", "full")
EXTRA_RENAMED = {"DTYPE": "renamed to PRECISION"}
RENAMED = {"SIZE": "the train script's name is SAMPLE_SIZE", **EXTRA_RENAMED}
'''

NAMES_TEXT = '''
from kohakumoss.offload.encoders.vae import VaePlan
from kohakumoss.types.plans import TextEncoderPlan

RENAMED_SAMPLE = {"SAMPLE_DTYPE": "PRECISION"}
SAMPLE_NAMES = ("WIDTH",)


def _f(cls):
    return list(cls.__annotations__)


ENCODER_FIELDS = {
    "TEXT_": ("TEXT_ENCODE", tuple(_f(TextEncoderPlan))),
    "VAE_": ("VAE_ENCODE", tuple(_f(VaePlan))),
}
'''


FACTORY_TEXT = '''
OPTIMIZERS = Registry("optimizer")
INTERNAL = {"cpu_adamw": "cpu", "host_state_adamw": "stream", "master_adamw": "gpu"}


@OPTIMIZERS.register("adamw")
def build_adamw(groups, device, **_):
    pass


@OPTIMIZERS.register("muon")
def build_muon(groups, device, **_):
    pass


@OPTIMIZERS.register("cpu_adamw")
def build_cpu(groups, device, **_):
    pass


@OPTIMIZERS.register("prodigy")
def build_prodigy(groups, device, **_):
    pass
'''


def _w(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def build_engine_repo(root: Path) -> Path:
    """tmp 假引擎资产(checkout 形态):分层配置 + knobs 元数据最小集。"""
    _w(root, "scripts/train/moss.py",
       'RUN_NAME = "x"\nSTEPS = 1000\nEXPORT_FORMATS = None\n')
    _w(root, "configs/moss/_base/base_lora.py",
       'STEPS = 4000\nLR = 1e-4\nDIM = 32\n'
       'TARGET_AREA = 1024 * 1024\n'
       'MEMORY_PROFILE = "balanced"\nSAVE_FORMAT = "both"\n'
       'SCHEDULE = {"mode": "cosine", "warmup": 100}\n')
    _w(root, "configs/moss/algo/algo_lora.py", 'DORA = "default"\n')
    _w(root, "configs/moss/data/data_kohya.py",
       'CAPTION = {"template": "tags"}\nASPECT_RANGE = (0.33, 3.0)\n')
    _w(root, "configs/moss/vram/vram_auto.py", 'MEMORY_PROFILE = "auto"\n')
    _w(root, "src/kohakumoss/__init__.py", "")
    _w(root, "src/kohakumoss/training/knobs.py", KNOBS_TEXT)
    _w(root, "src/kohakumoss/training/optim/optimizers/factory.py", FACTORY_TEXT)
    _w(root, "src/kohakumoss/memory/profiles.py",
       'ORDER = ("speed", "balanced", "low", "minimal")\n')
    _w(root, "src/kohakumoss/precision/config.py", 'COMPUTE = ("bf16", "fp16")\n')
    _w(root, "src/kohakumoss/encoders/knob_names.py", NAMES_TEXT)
    _w(root, "src/kohakumoss/types/plans.py",
       'class TextEncoderPlan:\n    memory: str = "ring"\n    ahead: int = 4\n')
    _w(root, "src/kohakumoss/offload/encoders/vae.py",
       'class VaePlan:\n    memory: str = "offload"\n')
    _w(root, "src/kohakumoss_kernels/tuning/tuner.py", 'MODES = ("table", "off")\n')
    (root / "configs" / "user").mkdir(parents=True, exist_ok=True)
    return root


class RootOnly:
    """DefaultsService 只依赖 adapter.repo_root(+可选 venv_dir 探测)。"""

    def __init__(self, repo_root):
        self.repo_root = Path(repo_root)


# ---------- 资产根定位 ----------

def test_locate_checkout_structural(tmp_path):
    build_engine_repo(tmp_path)
    root, sot, _ = DefaultsService(RootOnly(tmp_path)).locate_asset_root()
    assert root == tmp_path and sot == "checkout"


def test_locate_bundle_via_docstring(tmp_path):
    """产物通道:生成配置 docstring 打印 bundle 绝对路径 -> 反推 bundle 资产根。"""
    bundle = tmp_path / "sp" / "kohakumoss" / "bundle"
    _w(bundle, "scripts/train/moss.py", 'STEPS = 1000\n')
    _w(bundle, "configs/moss/_base/base_lora.py", 'STEPS = 4000\n')
    _w(bundle.parent, "training/knobs.py", KNOBS_TEXT)  # site-packages 包目录
    repo = tmp_path / "repo"
    printed = (bundle / "scripts" / "train" / "moss.py").as_posix()
    _w(repo, "configs/user/gen.py",
       f'"""doc.\n    kogine run {printed} --config configs/user/gen.py\n"""\n')
    root, sot, note = DefaultsService(RootOnly(repo)).locate_asset_root()
    assert root == bundle and sot == "bundle" and "gen.py" in note


def test_locate_unresolvable(tmp_path):
    root, sot, _ = DefaultsService(RootOnly(tmp_path)).locate_asset_root()
    assert root is None and sot == "fallback"


# ---------- 默认值组合与元数据 ----------

def test_get_defaults_layers_and_types(tmp_path):
    build_engine_repo(tmp_path)
    res = DefaultsService(RootOnly(tmp_path)).get_defaults()
    assert res["ok"], res
    assert res["source_of_truth"] == "checkout"
    kn = res["knobs"]
    # 分层覆盖:script 1000 -> base 4000;layer_source 记录最后设置层
    assert kn["STEPS"]["default_text"] == "4000"
    assert kn["STEPS"]["layer_source"] == "_base/base_lora"
    assert kn["EXPORT_FORMATS"]["default_text"] == "None"  # script 层
    assert kn["EXPORT_FORMATS"]["layer_source"] == "script"
    # ast 数字表达式与容器类型
    assert kn["TARGET_AREA"]["default_text"] == "1048576"
    assert kn["TARGET_AREA"]["type"] == "int"
    assert kn["ASPECT_RANGE"]["type"] == "tuple"
    assert kn["ASPECT_RANGE"]["default_text"] == "[0.33, 3.0]"
    assert kn["SCHEDULE"]["type"] == "dict"
    # vram 层覆盖 base 的 MEMORY_PROFILE;导入型枚举取自 profiles.ORDER
    assert kn["MEMORY_PROFILE"]["default_text"] == "auto"
    assert kn["MEMORY_PROFILE"]["layer_source"] == "vram/vram_auto"
    assert kn["MEMORY_PROFILE"]["choices"] == \
        ["speed", "balanced", "low", "minimal", "auto", "manual"]
    assert kn["CAPTION"]["layer_source"] == "data/data_kohya"
    assert kn["ALGO"]["choices"] == ["lora", "full"]
    assert kn["PRECISION"]["choices"] == ["bf16", "fp16", "auto"]
    # OPTIMIZER 枚举来自 factory 注册表,INTERNAL(OPTIM_STATE 自动变体)剔除
    assert kn["OPTIMIZER"]["choices"] == ["adamw", "muon", "prodigy"]
    # 分组/面板一致性:panel == 面板映射 ∩ KNOWN,expert 不可编辑
    assert set(res["panel"]) == {n for n, v in kn.items() if v["editable"]}
    assert set(res["panel"]) <= set(PANEL) and "STEPS" not in res["panel"]
    assert kn["RESUME"]["group"] == "expert"
    assert not kn["RESUME"]["editable"]
    assert set(res["groups"]) == set(GROUP_LABELS)
    # encoder 字段与 RENAMED 元数据
    assert res["encoder_fields"]["TEXT_"]["fields"] == ["memory", "ahead"]
    assert "SAMPLE_DTYPE" in res["renamed"]
    assert res["valid_key_count"] == 15 + 2 + 1  # KNOWN 15 + TEXT 2 + VAE 1


def test_get_defaults_missing_optional_layer(tmp_path):
    build_engine_repo(tmp_path)
    res = DefaultsService(RootOnly(tmp_path)).get_defaults(vram="8g")
    assert res["ok"] and res["missing_layers"] == ["vram/vram_8g"]


def test_check_override_keys(tmp_path):
    build_engine_repo(tmp_path)
    errors, warnings = DefaultsService(RootOnly(tmp_path)).check_override_keys({
        "LR": "1e-4", "NOPE": "1", "SIZE": "x", "TEXT_memory": "ring",
        "TEXT_zzz": "1", "SAMPLE_DTYPE": "x"})
    assert errors == ["NOPE", "TEXT_zzz"]
    joined = " | ".join(warnings)
    assert "SIZE" in joined and "TEXT_memory" in joined
    assert "SAMPLE_DTYPE" in joined  # knob_names 的 RENAMED_SAMPLE 并入


def test_check_override_keys_graceful_without_assets(tmp_path):
    errors, warnings = DefaultsService(RootOnly(tmp_path)).check_override_keys(
        {"NOPE": "1"})
    assert errors == [] and warnings == []  # 元数据不可用:放行,引擎最终把关


# ---------- 端到端:生成 -> 持久化 -> 启动 ----------

def test_new_train_persists_and_start_applies(tmp_path):
    build_engine_repo(tmp_path)
    fa = NewTrainRecorder(tmp_path)
    cs = ConfigService(fa, datasets_dir=tmp_path / "ds")
    res = cs.new_train({**FULL_TRAIN_FORM, "overrides": {"LR": "5e-5"}})
    assert res["ok"], res
    assert res["deferred_overrides"] == {"LR": "5e-5"}
    path = override_path("my-lora")
    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8")) == {"LR": "5e-5"}
    # start 未显式传 overrides -> 读持久化文件,argv 带 --set LR=5e-5
    ft = FakeTasks()
    ts = TrainService(fa, ft, cs)
    r2 = ts.start("my-lora")
    assert r2["ok"], r2
    assert "LR=5e-5" in r2["task"]["argv"]
    assert r2["override_count"] == 1
    # 显式 overrides 优先于持久化
    ft.rows.clear()
    r3 = ts.start("my-lora", {"DIM": "16"})
    assert "DIM=16" in r3["task"]["argv"]
    assert "LR=5e-5" not in r3["task"]["argv"]


def test_new_train_rejects_unknown_key(tmp_path):
    build_engine_repo(tmp_path)
    fa = NewTrainRecorder(tmp_path)
    cs = ConfigService(fa, datasets_dir=tmp_path / "ds")
    res = cs.new_train({**FULL_TRAIN_FORM, "overrides": {"NOPE": "1"}})
    assert res["ok"] is False and "未知键" in res["error"]
    assert fa.train_argv is None  # 校验在调引擎之前


def test_start_rejects_unknown_key(tmp_path):
    build_engine_repo(tmp_path)
    cfg = tmp_path / "configs" / "user" / "demo.py"
    cfg.write_text('RUN_NAME = "demo-run"\n', encoding="utf-8")
    ts = TrainService(FakeAdapter(tmp_path), FakeTasks(),
                      ConfigService(FakeAdapter(tmp_path),
                                    datasets_dir=tmp_path / "ds"))
    res = ts.start("demo", {"NOPE": "1"})
    assert res["ok"] is False and "未知键" in res["error"]


def test_start_warns_on_renamed_key(tmp_path):
    build_engine_repo(tmp_path)
    cfg = tmp_path / "configs" / "user" / "demo.py"
    cfg.write_text('RUN_NAME = "demo-run"\n', encoding="utf-8")
    ts = TrainService(FakeAdapter(tmp_path), FakeTasks(),
                      ConfigService(FakeAdapter(tmp_path),
                                    datasets_dir=tmp_path / "ds"))
    res = ts.start("demo", {"SIZE": "512"})
    assert res["ok"]
    assert any("SIZE" in w for w in res["override_warnings"])


def test_empty_overrides_clears_stale_file(tmp_path):
    save_overrides("demo", {"LR": "1e-4"})
    assert override_path("demo").is_file()
    save_overrides("demo", {})
    assert not override_path("demo").exists()


# ---------- 路由契约 ----------

def test_api_defaults_and_overrides_routes(tmp_path, monkeypatch):
    build_engine_repo(tmp_path)
    monkeypatch.setattr(app_module, "defaults_service",
                        DefaultsService(RootOnly(tmp_path)))
    save_overrides("demo", {"LR": "5e-5", "DIM": "16"})
    client = TestClient(app_module.app)
    r = client.get("/api/config/defaults?source=kohya&algo=lora&vram=auto")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] and body["knobs"]["STEPS"]["default_text"] == "4000"
    assert len(body["panel"]) == len(set(body["panel"]))
    r2 = client.get("/api/config/overrides", params={"name": "demo"})
    assert r2.status_code == 200
    assert r2.json()["count"] == 2
    assert r2.json()["overrides"] == {"LR": "5e-5", "DIM": "16"}


# ---------- 真实引擎资产金样(布局漂移兜底) ----------

def test_real_engine_defaults_golden():
    from adapter import EngineAdapter
    ds = DefaultsService(EngineAdapter())
    root, sot, _ = ds.locate_asset_root()
    if root is None:
        pytest.skip("无法定位引擎资产根,跳过金样")
    res = ds.get_defaults()
    assert res["ok"], res.get("error")
    # 当前契约:144 旋钮 / 面板 43 项 / base_lora 步数 4000
    assert len(res["knobs"]) == 144, f"KNOWN 数量漂移:{len(res['knobs'])}"
    assert len(res["panel"]) == 43
    assert res["knobs"]["STEPS"]["default_text"] == "4000"
    assert res["knobs"]["MEMORY_PROFILE"]["choices"][:4] == \
        ["speed", "balanced", "low", "minimal"]
    assert all(res["knobs"][n]["editable"] and
               res["knobs"][n]["group"] in GROUP_LABELS and
               res["knobs"][n]["group"] != "expert"
               for n in res["panel"])
