"""config_service 单元测试:argv 构造 + list 解析器。

全部用假 adapter(记录 argv、返回预制输出),**不触碰引擎与真实 configs/user**。
"""

import json
import subprocess
from pathlib import Path

from config_service import (
    ConfigService,
    append_datasets_to_config,
    apply_trigger_to_config,
    build_from_argv,
    build_sample_argv,
    build_train_argv,
    parse_config_list,
    set_sample_size_in_config,
    vram_options,
)

FIXTURE = Path(__file__).parent / "fixtures" / "config_list.txt"


class FakeAdapter:
    """记录 argv 并返回预制 CompletedProcess 的假适配器。"""

    def __init__(self, repo_root: Path, stdout: str = "", returncode: int = 0,
                 has_exe: bool = True) -> None:
        self.repo_root = repo_root
        self.calls: list[list[str]] = []
        self.stdout = stdout
        self.returncode = returncode
        self.has_exe = has_exe

    def find_executable(self, name: str):
        return Path(f"/venv/bin/{name}") if self.has_exe else None

    def _run(self, argv, timeout: float = 30):
        self.calls.append([str(a) for a in argv])
        return subprocess.CompletedProcess(argv, self.returncode,
                                           stdout=self.stdout, stderr="")


# ---------- new train argv 构造 ----------

def test_train_argv_full_mapping():
    argv, err = build_train_argv("cfg-exe", {
        "weights": "D:/models", "data": ["D:/ds"], "name": "run1",
        "algo": "loha", "source": "kohya", "steps": 1600,
        "prompt": ["a cat", "a dog"], "vram": "8", "precision": "bf16",
    })
    assert err is None
    assert argv == ["cfg-exe", "new", "train", "--weights", "D:/models",
                    "--data", "D:/ds", "--name", "run1", "--algo", "loha",
                    "--source", "kohya", "--steps", "1600",
                    "--prompt", "a cat", "--prompt", "a dog",
                    "--vram", "8", "--precision", "bf16"]


def test_train_argv_minimal_and_defaults():
    """只给必填项:可选 flag 一律不出现(vram 留空 = 引擎自动检测)。"""
    argv, err = build_train_argv("cfg-exe", {"weights": "w", "data": "d", "name": "n"})
    assert err is None
    assert "--algo" not in argv and "--steps" not in argv and "--vram" not in argv
    assert "--precision" not in argv and "--force" not in argv


def test_train_argv_vram_auto_and_steps_zero():
    argv, err = build_train_argv("exe", {"weights": "w", "data": "d", "name": "n",
                                         "vram": "auto", "steps": 0, "gpus": 1})
    assert err is None
    assert argv[argv.index("--vram") + 1] == "auto"
    assert "--steps" not in argv and "--gpus" not in argv  # 0/1 为引擎默认,不传


def test_train_argv_force():
    argv, err = build_train_argv("exe", {"weights": "w", "data": "d", "name": "n",
                                         "force": True})
    assert err is None and argv[-1] == "--force"


def test_train_argv_overrides_not_in_argv():
    """引擎 new train 没有 --set:overrides 绝不进 argv(由服务层回显 deferred)。"""
    argv, err = build_train_argv("exe", {"weights": "w", "data": "d", "name": "n",
                                         "overrides": {"LR": "2e-4", "DIM": "32"}})
    assert err is None
    assert "--set" not in argv and "LR=2e-4" not in argv


def test_train_argv_validation_errors():
    assert build_train_argv("e", {"data": "d", "name": "n"})[1]  # 缺 weights
    assert build_train_argv("e", {"weights": "w", "name": "n"})[1]  # 缺 data
    assert build_train_argv("e", {"weights": "w", "data": "d"})[1]  # 缺 name
    # 多数据集:引擎 --data 为单值,必须报错而不是静默丢弃
    assert "单个" in build_train_argv("e", {"weights": "w", "data": ["a", "b"],
                                           "name": "n"})[1]
    # 名字含路径分隔符/穿越
    assert build_train_argv("e", {"weights": "w", "data": "d", "name": "../evil"})[1]
    assert build_train_argv("e", {"weights": "w", "data": "d", "name": "a/b"})[1]


# ---------- new sample / --from argv ----------

def test_sample_argv():
    argv, err = build_sample_argv("exe", {
        "weights": "w", "name": "s1", "adapter": ["a.safetensors:1.0", "b.safetensors"],
        "prompt": "a cat", "vram": 8, "precision": "fp16", "samples": "out/dir",
    })
    assert err is None
    assert argv.count("--adapter") == 2
    assert argv[argv.index("--prompt") + 1] == "a cat"
    assert "--vram" in argv and "--samples" in argv


def test_sample_argv_vram_required():
    """采样引擎不接受 auto 显存。"""
    assert build_sample_argv("exe", {"weights": "w", "vram": "auto"})[1]
    assert build_sample_argv("exe", {"weights": "w"})[1]


def test_from_argv():
    argv, err = build_from_argv("exe", "lora_24g_bf16", "configs/user/x.py", force=True)
    assert err is None
    assert argv == ["exe", "new", "--from", "lora_24g_bf16", "configs/user/x.py", "--force"]
    assert build_from_argv("exe", "", "o")[1]
    assert build_from_argv("exe", "r", "")[1]


# ---------- list 解析器 ----------

def test_parse_config_list_real_output():
    parsed = parse_config_list(FIXTURE.read_text(encoding="utf-8"))
    assert parsed["parsed"] is True
    assert "lora_24g_bf16" in parsed["recipes"]
    assert parsed["presets"]["moss"]["algo"] == ["dora", "full", "loha", "lokr", "lora", "oft"]
    assert parsed["presets"]["moss"]["data"] == ["folder", "folder_nl", "jsonl", "kohya",
                                                 "moss_shard", "parquet", "webdataset"]
    assert parsed["presets"]["sample"]["precision"] == ["bf16", "fp16"]
    assert parsed["catalog_more"] == 240
    speed = parsed["memory_profiles"][0]
    assert speed["name"] == "speed" and speed["lora_gib"] == 35.0
    assert speed["optimizer"] == "gpu"
    assert parsed["memory_profiles"][3]["name"] == "minimal"


def test_vram_options_derived_from_presets():
    parsed = parse_config_list(FIXTURE.read_text(encoding="utf-8"))
    # 6/8/12/16/24 + auto;24g_full 层不出现(UI 只传 GiB)
    assert vram_options(parsed["presets"]) == [6, 8, 12, 16, 24, "auto"]


def test_parse_config_list_garbage_degrades():
    """乱文本:parsed=False、不抛异常、原文由调用方保留。"""
    parsed = parse_config_list("!!!totally not the expected format@@@\n随机中文行")
    assert parsed["parsed"] is False
    assert parsed["recipes"] == [] and parsed["memory_profiles"] == []


def test_parse_config_list_partial():
    """只有部分行也能出部分结构(上游格式变动时不整体崩)。"""
    parsed = parse_config_list("recipes          a, b\n")
    assert parsed["parsed"] is True and parsed["recipes"] == ["a", "b"]


# ---------- 服务层行为 ----------

def test_list_presets_structured(tmp_path):
    svc = ConfigService(FakeAdapter(tmp_path, stdout=FIXTURE.read_text(encoding="utf-8")))
    res = svc.list_presets()
    assert res["ok"] is True and res["parsed"] is True
    assert res["vram_options"] == [6, 8, 12, 16, 24, "auto"]
    assert svc.adapter.calls[-1] == [str(Path("/venv/bin/kohakumoss-config")), "list"]


def test_list_presets_all_flag(tmp_path):
    svc = ConfigService(FakeAdapter(tmp_path, stdout="recipes  a\n"))
    svc.list_presets(everything=True)
    assert svc.adapter.calls[-1][-1] == "--all"


def test_list_presets_cached_until_fresh(tmp_path):
    """presets 缓存:同参数只跑一次 config list;fresh=True 或换 --all 键才重跑。"""
    svc = ConfigService(FakeAdapter(tmp_path, stdout="recipes  a\n"))
    svc.list_presets()
    n = len(svc.adapter.calls)
    svc.list_presets()
    assert len(svc.adapter.calls) == n
    svc.list_presets(fresh=True)
    assert len(svc.adapter.calls) == n + 1
    svc.list_presets(everything=True)
    assert len(svc.adapter.calls) == n + 2


def test_list_presets_engine_missing(tmp_path):
    svc = ConfigService(FakeAdapter(tmp_path, has_exe=False))
    res = svc.list_presets()
    assert res["ok"] is False and res["engine_missing"] is True


def test_new_train_success_reads_back_file(tmp_path):
    """生成成功:exit=0 后读回 configs/user/<name>.py 与 _data.py 原文。"""
    (tmp_path / "configs" / "user").mkdir(parents=True)
    (tmp_path / "configs" / "user" / "demo.py").write_text("RUN_NAME = \"demo\"\n", encoding="utf-8")
    (tmp_path / "configs" / "user" / "demo_data.py").write_text("X = 1\n", encoding="utf-8")
    svc = ConfigService(FakeAdapter(tmp_path))
    res = svc.new_train({"weights": "w", "data": "d", "name": "demo", "force": True,
                         "overrides": {"LR": "2e-4"}})
    assert res["ok"] is True
    assert res["content"] == "RUN_NAME = \"demo\"\n"
    assert res["data_content"] == "X = 1\n"
    assert res["path"] == "configs/user/demo.py"
    assert res["deferred_overrides"] == {"LR": "2e-4"}  # 不进 argv,只回显
    # 服务层显式传 --out 到 configs/user(bundle 安装时引擎默认目录会退化为 cwd)
    call = svc.adapter.calls[-1]
    assert call[call.index("--out") + 1].replace("\\", "/").endswith("configs/user/demo.py")


def test_new_train_conflict_without_force(tmp_path):
    """同名已存在且未 force:exists=True 且不调引擎。"""
    (tmp_path / "configs" / "user").mkdir(parents=True)
    (tmp_path / "configs" / "user" / "demo.py").write_text("x", encoding="utf-8")
    adapter = FakeAdapter(tmp_path)
    svc = ConfigService(adapter)
    res = svc.new_train({"weights": "w", "data": "d", "name": "demo"})
    assert res["ok"] is False and res["exists"] is True
    assert adapter.calls == []
    # force 后真正调引擎
    res = svc.new_train({"weights": "w", "data": "d", "name": "demo", "force": True})
    assert adapter.calls and "--force" in adapter.calls[0]


def test_new_train_engine_failure(tmp_path):
    adapter = FakeAdapter(tmp_path, returncode=2, stdout="")
    svc = ConfigService(adapter)
    res = svc.new_train({"weights": "w", "data": "d", "name": "n"})
    assert res["ok"] is False and res["exit_code"] == 2


# ---------- 采样 prompt 每条自定义分辨率(SAMPLE_SIZE 配置文件注入) ----------

def _make_train_svc(tmp_path, monkeypatch, content: str = "x"):
    """生成成功路径的通用夹具:预置目标文件 + 拦截 save_overrides 防写真实 state。"""
    import config_service as cs
    saved: dict = {}
    monkeypatch.setattr(cs, "save_overrides",
                        lambda name, ov: saved.__setitem__(name, dict(ov)))
    (tmp_path / "configs" / "user").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs" / "user" / "demo.py").write_text(content, encoding="utf-8")
    return ConfigService(FakeAdapter(tmp_path)), saved


def test_set_sample_size_replaces_existing_line():
    text = "SAMPLE_PROMPTS = ['a']\nSAMPLE_SIZE = (1024, 1024)\nSAMPLE_EVERY = 500\n"
    new, err = set_sample_size_in_config(text, [[1024, 1024], [768, 1024]])
    assert err is None
    assert "SAMPLE_SIZE = [(1024, 1024), (768, 1024)]" in new
    assert "(1024, 1024)\n" not in new  # 旧行被整行替换


def test_set_sample_size_inserts_after_single_line_prompts():
    text = "SAMPLE_PROMPTS = ['a', 'b']\nSAMPLE_EVERY = 500\n"
    new, err = set_sample_size_in_config(text, [[640, 640]])
    assert err is None
    assert new.splitlines()[1] == "SAMPLE_SIZE = [(640, 640)]"


def test_set_sample_size_appends_when_multiline_prompts():
    """SAMPLE_PROMPTS 为多行字面量:不在中间插入,退化为文件末尾追加。"""
    text = "SAMPLE_PROMPTS = [\n    'a',\n]\nSAMPLE_EVERY = 500\n"
    new, err = set_sample_size_in_config(text, [[640, 640]])
    assert err is None
    assert new.rstrip().endswith("SAMPLE_SIZE = [(640, 640)]")


def test_set_sample_size_rejects_invalid():
    _, err = set_sample_size_in_config("x", [[0, 512]])
    assert err and "非法" in err


def test_new_train_prompt_sizes_injects_sample_size(tmp_path, monkeypatch):
    """任一分辨率非默认 1024x1024:生成后把 SAMPLE_SIZE 平行列表写进配置文件
    (--set 通道对无注解旋钮按默认值推断 tuple[int,...],平行列表必然强转失败,
    实测只能走文本注入),并触发 config check。"""
    svc, saved = _make_train_svc(tmp_path, monkeypatch)
    res = svc.new_train({"weights": "w", "data": "d", "name": "demo", "force": True,
                         "prompt": ["a cat", "a dog"],
                         "prompt_sizes": [[1024, 1024], [768, 1024]]})
    assert res["ok"] is True
    assert res["sample_size_injected"] == [[1024, 1024], [768, 1024]]
    assert "SAMPLE_SIZE = [(1024, 1024), (768, 1024)]" in res["content"]
    assert res["check"]["ok"] is True
    # 不进 overrides/--set 通道(save_overrides 全量替换,旧值不残留)
    assert "SAMPLE_SIZE" not in (res.get("deferred_overrides") or {})
    assert "SAMPLE_SIZE" not in saved["demo"]


def test_new_train_prompt_sizes_all_default_not_injected(tmp_path, monkeypatch):
    """全默认 1024x1024:配置文件保持引擎原文,不注入、不跑注入后 check。"""
    svc, saved = _make_train_svc(tmp_path, monkeypatch)
    res = svc.new_train({"weights": "w", "data": "d", "name": "demo", "force": True,
                         "prompt": ["a cat"], "prompt_sizes": [[1024, 1024]]})
    assert res["ok"] is True
    assert "SAMPLE_SIZE" not in res["content"]
    assert "check" not in res
    assert "SAMPLE_SIZE" not in saved["demo"]


def test_new_train_prompt_sizes_manual_sample_size_wins(tmp_path, monkeypatch):
    """高级面板手动设了 SAMPLE_SIZE:手动值走 --set 优先,不做文本注入并给出提示。"""
    svc, _ = _make_train_svc(tmp_path, monkeypatch)
    res = svc.new_train({"weights": "w", "data": "d", "name": "demo", "force": True,
                         "prompt": ["a cat"], "prompt_sizes": [[768, 1024]],
                         "overrides": {"SAMPLE_SIZE": "(512, 512)"}})
    assert res["ok"] is True
    assert res["deferred_overrides"]["SAMPLE_SIZE"] == "(512, 512)"
    assert any("SAMPLE_SIZE" in w for w in res["override_warnings"])
    assert "SAMPLE_SIZE" not in res["content"]


def test_new_train_prompt_sizes_mismatch_rejected(tmp_path, monkeypatch):
    """尺寸条数与 prompt 条数不一致 / 尺寸非法:生成前即拒绝,不调引擎。"""
    svc, _ = _make_train_svc(tmp_path, monkeypatch)
    res = svc.new_train({"weights": "w", "data": "d", "name": "demo", "force": True,
                         "prompt": ["a cat", "a dog"], "prompt_sizes": [[1024, 1024]]})
    assert res["ok"] is False and "不一致" in res["error"]
    res = svc.new_train({"weights": "w", "data": "d", "name": "demo", "force": True,
                         "prompt": ["a cat"], "prompt_sizes": [[0, 1024]]})
    assert res["ok"] is False and "非法" in res["error"]
    assert svc.adapter.calls == []


def test_check_config_exit_codes(tmp_path):
    """exit=0 -> ok;exit=1 -> ok=False 但 error=None(配置不合法是数据)。"""
    (tmp_path / "configs" / "user").mkdir(parents=True)
    cfg = tmp_path / "configs" / "user" / "demo.py"
    cfg.write_text("x", encoding="utf-8")
    svc = ConfigService(FakeAdapter(tmp_path, stdout="configs/user/demo.py: ok\n"))
    res = svc.check_config("demo")
    assert res["ok"] is True and res["error"] is None
    assert res["stdout"].endswith("ok\n")
    bad = ConfigService(FakeAdapter(tmp_path, stdout="demo.py: invalid\n  - bad\n",
                                    returncode=1))
    res = bad.check_config("demo.py")
    assert res["ok"] is False and res["error"] is None and res["exit_code"] == 1


def test_check_config_missing_file(tmp_path):
    svc = ConfigService(FakeAdapter(tmp_path))
    res = svc.check_config("nope")
    assert res["ok"] is False and "不存在" in res["error"]


def test_read_user_config_whitelist(tmp_path):
    (tmp_path / "configs" / "user").mkdir(parents=True)
    (tmp_path / "configs" / "user" / "ok.py").write_text("A = 1\n", encoding="utf-8")
    (tmp_path / "configs" / "evil.py").write_text("secret", encoding="utf-8")
    svc = ConfigService(FakeAdapter(tmp_path))
    assert svc.read_user_config("ok")["content"] == "A = 1\n"
    assert svc.read_user_config("ok.py")["ok"] is True
    for evil in ("../evil", "..\\evil", "a/b", "", "nope"):
        assert svc.read_user_config(evil)["ok"] is False, evil
    assert svc.read_user_config("nope")["not_found"] is True


def test_detect_weights_comfy_layout(tmp_path):
    """项目内 weights/models 三子目录齐全 -> 命中 comfy 布局并返回绝对路径。"""
    for sub in ("diffusion_models", "text_encoders", "vae"):
        (tmp_path / "weights" / "models" / sub).mkdir(parents=True)
    svc = ConfigService(FakeAdapter(tmp_path))
    res = svc.detect_weights()
    assert res["ok"] and res["found"] and res["layout"] == "comfy"
    assert res["relative"] == "weights/models"
    assert res["path"] == str((tmp_path / "weights" / "models").resolve())


def test_detect_weights_hf_layout(tmp_path):
    (tmp_path / "weights" / "transformer").mkdir(parents=True)
    (tmp_path / "weights" / "vae").mkdir(parents=True)
    svc = ConfigService(FakeAdapter(tmp_path))
    res = svc.detect_weights()
    assert res["found"] and res["layout"] == "hf" and res["relative"] == "weights"


def test_detect_weights_none(tmp_path):
    """无集成权重:found=False(前端保持 placeholder 手填)。"""
    (tmp_path / "weights").mkdir()  # 空目录不算命中
    svc = ConfigService(FakeAdapter(tmp_path))
    res = svc.detect_weights()
    assert res["ok"] and res["found"] is False and res["path"] is None


def test_list_user_configs(tmp_path):
    d = tmp_path / "configs" / "user"
    d.mkdir(parents=True)
    (d / "b.py").write_text("x")
    (d / "a.py").write_text("x")
    (d / "a_data.py").write_text("x")  # 引擎配套数据配置:不列为用户实体
    (d / "note.txt").write_text("x")
    svc = ConfigService(FakeAdapter(tmp_path))
    assert svc.list_user_configs()["configs"] == ["a", "b"]
    empty = ConfigService(FakeAdapter(tmp_path / "other"))
    assert empty.list_user_configs()["configs"] == []


def test_data_companion_not_a_user_entity(tmp_path):
    """*_data.py 是内部配套文件:读/删/路径解析一律拒绝,只认主配置名。"""
    d = tmp_path / "configs" / "user"
    d.mkdir(parents=True)
    (d / "demo.py").write_text("RUN_NAME = 'demo'\n", encoding="utf-8")
    (d / "demo_data.py").write_text("X = 1\n", encoding="utf-8")
    svc = ConfigService(FakeAdapter(tmp_path))
    for name in ("demo_data", "demo_data.py"):
        r = svc.read_user_config(name)
        assert not r["ok"] and "_data" in r["error"]
        rd = svc.delete_user_config(name)
        assert not rd["ok"] and "_data" in rd["error"]
    # 主配置与配套文件都还在(拒绝即不产生副作用)
    assert (d / "demo.py").is_file() and (d / "demo_data.py").is_file()
    # 按主名删除才成对清理
    assert svc.delete_user_config("demo")["ok"]
    assert not (d / "demo.py").exists() and not (d / "demo_data.py").exists()


def test_train_name_rejects_data_suffix():
    """LoRA 名称以 _data 结尾会与引擎配套文件命名冲突,生成期即拒绝。"""
    argv, err = build_train_argv("exe", {"weights": "w", "data": "d", "name": "my_data"})
    assert argv == [] and "_data" in err


# ---------- M1.5:datasets(库多选)-> argv / DATASETS 追加 ----------

# 引擎真实生成的 DATASETS 块结构(M1.5 实测 probe 摘录)
TRAIN_CFG_TEXT = '''"""demo: lora."""
from kohakuengine import use_config
use_config("data_kohya.py")

# ===================== DATA =====================
DATASETS = [
    {
        "source": "folder",
        "path": "D:/KohakuMossTrainer/ui/state/datasets/a",
        "txt_as": "tags",
    },
]
# RESOLUTION_MODE = "resize"
STEPS = 200
RUN_NAME = "demo"
'''


def test_train_argv_datasets_first_entry_only():
    """datasets 多选:data 已被服务层解析为多个目录时,--data 只吃第一个,不报错。"""
    argv, err = build_train_argv("exe", {"weights": "w", "name": "n",
                                         "datasets": ["a", "b"],
                                         "data": ["D:/ds/a", "D:/ds/b"]})
    assert err is None
    assert argv.count("--data") == 1
    assert argv[argv.index("--data") + 1] == "D:/ds/a"


def test_train_argv_multi_data_without_datasets_still_errors():
    """兼容:手填 data 多行(无 datasets 键)仍保持旧报错。"""
    assert "单个" in build_train_argv("exe", {"weights": "w", "name": "n",
                                             "data": ["a", "b"]})[1]


def test_append_datasets_to_config_structure():
    new, err = append_datasets_to_config(TRAIN_CFG_TEXT, ["D:/x/ui/state/datasets/b"])
    assert err is None
    compile(new, "<cfg>", "exec")  # 追加后仍是合法 Python
    assert new.count('"source": "folder"') == 2  # 复制首条目全部旋钮
    assert '"path": "D:/x/ui/state/datasets/b"' in new
    # 新条目插在列表闭合 ] 之前,原内容不损坏
    assert new.index("datasets/b") < new.index("\n]")
    assert "STEPS = 200" in new
    # 反斜杠路径归一为正斜杠
    new2, err2 = append_datasets_to_config(TRAIN_CFG_TEXT, [r"D:\ds\b"])
    assert err2 is None and '"path": "D:/ds/b"' in new2


def test_append_datasets_failure_cases():
    assert append_datasets_to_config("RUN_NAME = 1\n", ["p"])[1]
    assert append_datasets_to_config("DATASETS = [\n]\n", ["p"])[1]  # 无条目可复制
    no_path = 'DATASETS = [\n    {"source": "folder"},\n]\n'
    assert "path" in append_datasets_to_config(no_path, ["p"])[1]
    assert append_datasets_to_config(TRAIN_CFG_TEXT, ['D:/we"ird'])[1]  # 引号拒绝


class ScriptedAdapter(FakeAdapter):
    """按子命令返回不同退出码:new train 成功、check 失败(或反之)。"""

    def __init__(self, repo_root, gen_rc=0, check_rc=0, **kw):
        super().__init__(repo_root, **kw)
        self.gen_rc, self.check_rc = gen_rc, check_rc

    def _run(self, argv, timeout: float = 30):
        self.calls.append([str(a) for a in argv])
        cmd = str(argv[1]) if len(argv) > 1 else ""
        rc = self.gen_rc if cmd == "new" else self.check_rc
        return subprocess.CompletedProcess(argv, rc, stdout="out", stderr="engine says no" if rc else "")


def _make_datasets(tmp_path, names):
    root = tmp_path / "ui_state" / "datasets"
    for n in names:
        (root / n).mkdir(parents=True)
    return root


def test_new_train_datasets_single_no_append(tmp_path):
    """datasets 单选:第一个目录走 --data,默认 --source folder,不追加、不内部 check。"""
    (tmp_path / "configs" / "user").mkdir(parents=True)
    ds = _make_datasets(tmp_path, ["a"])
    svc = ConfigService(FakeAdapter(tmp_path), datasets_dir=ds)
    res = svc.new_train({"weights": "w", "datasets": ["a"], "name": "demo"})
    assert res["ok"] is True
    assert res["datasets_used"] == ["a"]
    call = svc.adapter.calls[-1]
    assert call[call.index("--data") + 1] == str(ds / "a")
    assert call[call.index("--source") + 1] == "folder"
    assert len(svc.adapter.calls) == 1  # 没有 check


def test_new_train_datasets_forces_folder_source(tmp_path):
    """M2 实测:库目录永远是平铺布局,kohya source 训练期 catalog scan 直接
    ValueError;即使表单显式传 kohya 也必须强制 --source folder。"""
    (tmp_path / "configs" / "user").mkdir(parents=True)
    ds = _make_datasets(tmp_path, ["a"])
    svc = ConfigService(FakeAdapter(tmp_path), datasets_dir=ds)
    res = svc.new_train({"weights": "w", "datasets": ["a"], "name": "demo",
                         "source": "kohya"})
    assert res["ok"] is True
    call = svc.adapter.calls[-1]
    assert call[call.index("--source") + 1] == "folder"


def test_new_train_datasets_multi_appends_and_checks(tmp_path):
    """多选:生成后把其余库追加进 DATASETS 并跑 check;check 过 -> ok + datasets_appended。"""
    ud = tmp_path / "configs" / "user"
    ud.mkdir(parents=True)
    (ud / "demo.py").write_text(TRAIN_CFG_TEXT.replace(
        "datasets/a", f"{(tmp_path / 'ui_state' / 'datasets' / 'a').as_posix()}"), encoding="utf-8")
    ds = _make_datasets(tmp_path, ["a", "b"])
    svc = ConfigService(FakeAdapter(tmp_path), datasets_dir=ds)
    res = svc.new_train({"weights": "w", "datasets": ["a", "b"], "name": "demo", "force": True})
    assert res["ok"] is True, res.get("error")
    assert res["datasets_used"] == ["a", "b"]
    assert res["datasets_appended"] == ["b"]
    assert res["check"]["ok"] is True
    final = (ud / "demo.py").read_text(encoding="utf-8")
    assert f"{(ds / 'b').as_posix()}" in final
    compile(final, "<cfg>", "exec")
    assert res["content"] == final  # 读回的是追加后的全文


def test_new_train_datasets_check_failure_returns_error(tmp_path):
    """追加后 check 不过:ok=False 且附引擎输出。"""
    ud = tmp_path / "configs" / "user"
    ud.mkdir(parents=True)
    (ud / "demo.py").write_text(TRAIN_CFG_TEXT, encoding="utf-8")
    ds = _make_datasets(tmp_path, ["a", "b"])
    svc = ConfigService(ScriptedAdapter(tmp_path, gen_rc=0, check_rc=1), datasets_dir=ds)
    res = svc.new_train({"weights": "w", "datasets": ["a", "b"], "name": "demo", "force": True})
    assert res["ok"] is False and "check" in res["error"]
    assert res["check"]["ok"] is False and res["check"]["stderr"] == "engine says no"


def test_new_train_datasets_unknown_name_rejected(tmp_path):
    """库中不存在的名字:不调引擎,直接报错。"""
    ds = _make_datasets(tmp_path, ["a"])
    adapter = FakeAdapter(tmp_path)
    svc = ConfigService(adapter, datasets_dir=ds)
    res = svc.new_train({"weights": "w", "datasets": ["ghost"], "name": "demo"})
    assert res["ok"] is False and "不在库中" in res["error"]
    assert adapter.calls == []
    # 非法名(穿越)同样拒绝
    assert svc.new_train({"weights": "w", "datasets": ["../evil"], "name": "demo"})["ok"] is False
    assert adapter.calls == []


def test_new_train_legacy_data_path_still_works(tmp_path):
    """兼容:旧 data 路径字段(无 datasets)行为不变。"""
    (tmp_path / "configs" / "user").mkdir(parents=True)
    svc = ConfigService(FakeAdapter(tmp_path))
    res = svc.new_train({"weights": "w", "data": "D:/ds", "name": "demo"})
    assert res["ok"] is True and "datasets_used" not in res
    call = svc.adapter.calls[-1]
    assert call[call.index("--data") + 1] == "D:/ds"
    assert "--source" not in call  # 旧路径不注入 folder 默认


# ---------- 触发词(kohya tag word 等价:CAPTION tags 模板 prefix 注入) ----------

CAPTION_CFG = TRAIN_CFG_TEXT + """CAPTION = {
    "template": "tags",
    "keep": ["artist"],
    "shuffle": True,
}
"""
NL_CFG = TRAIN_CFG_TEXT + 'CAPTION = {\n    "template": "nl",\n}\n'


def test_apply_trigger_injects_prefix_into_existing_caption():
    new, err = apply_trigger_to_config(CAPTION_CFG, "skz", None)
    assert err is None
    compile(new, "<cfg>", "exec")
    assert '"prefix": "skz, "' in new
    assert new.index('"prefix"') < new.index('"template"')
    # 二次注入:替换而非叠加
    again, _ = apply_trigger_to_config(new, "skz2", None)
    assert again.count('"prefix"') == 1 and '"skz2, "' in again


def test_apply_trigger_rejects_non_tags_and_bad_input():
    assert "tags" in apply_trigger_to_config(NL_CFG, "skz", None)[1]
    assert apply_trigger_to_config(CAPTION_CFG, 'a"b', None)[1]
    assert apply_trigger_to_config(TRAIN_CFG_TEXT, "skz", None)[1]  # 无 CAPTION 也没格式层 dict
    assert "tags" in apply_trigger_to_config(TRAIN_CFG_TEXT, "skz", {"template": "nl"})[1]


def test_apply_trigger_appends_layer_caption():
    """生成文件无 CAPTION:整份追加数据格式层 dict + prefix,不丢 keep/tag_dropout 旋钮。"""
    layer = {"template": "tags", "keep": ["artist"], "tag_dropout": 0.1, "shuffle": True}
    new, err = apply_trigger_to_config(TRAIN_CFG_TEXT, "skz", layer)
    assert err is None
    compile(new, "<cfg>", "exec")
    assert '"prefix": "skz, "' in new and '"tag_dropout": 0.1' in new


def test_new_train_trigger_prepends_prompts_and_injects(tmp_path):
    """触发词端到端:--prompt 前缀注入 + 生成后 CAPTION prefix 改写 + check。"""
    ud = tmp_path / "configs" / "user"
    ud.mkdir(parents=True)
    layer_dir = tmp_path / "configs" / "moss" / "data"
    layer_dir.mkdir(parents=True)
    (layer_dir / "data_kohya.py").write_text(
        'CAPTION = {"template": "tags", "keep": ["artist"], "shuffle": True}\n',
        encoding="utf-8")
    (ud / "demo.py").write_text(TRAIN_CFG_TEXT, encoding="utf-8")
    svc = ConfigService(FakeAdapter(tmp_path))
    res = svc.new_train({"weights": "w", "data": "d", "name": "demo",
                         "prompt": ["a cat"], "trigger": "skz", "force": True})
    assert res["ok"] is True, res.get("error")
    assert res["trigger_applied"] == "skz"
    call = svc.adapter.calls[0]  # new train argv:采样 prompt 带触发词前缀
    assert call[call.index("--prompt") + 1] == "skz, a cat"
    final = (ud / "demo.py").read_text(encoding="utf-8")
    assert '"prefix": "skz, "' in final
    compile(final, "<cfg>", "exec")
    assert res["check"]["ok"] is True
    assert res["content"] == final  # 读回的是注入后的全文


def test_new_train_trigger_unsupported_source_errors(tmp_path):
    """folder_nl(nl 模板):注入失败 -> ok=False,文件保持原样不半损坏。"""
    ud = tmp_path / "configs" / "user"
    ud.mkdir(parents=True)
    layer_dir = tmp_path / "configs" / "moss" / "data"
    layer_dir.mkdir(parents=True)
    (layer_dir / "data_folder_nl.py").write_text(
        'CAPTION = {"template": "nl"}\n', encoding="utf-8")
    (ud / "demo.py").write_text(TRAIN_CFG_TEXT, encoding="utf-8")
    svc = ConfigService(FakeAdapter(tmp_path))
    res = svc.new_train({"weights": "w", "data": "d", "name": "demo",
                         "source": "folder_nl", "trigger": "skz", "force": True})
    assert res["ok"] is False and "tags" in res["error"]
    assert (ud / "demo.py").read_text(encoding="utf-8") == TRAIN_CFG_TEXT
