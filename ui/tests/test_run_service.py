"""run_service 单元测试:RUN_NAME 解析、样图白名单防穿越、extract_json。"""

from run_service import parse_run_name, resolve_sample_file
from train_service import extract_json


def test_parse_run_name(tmp_path):
    text = 'use_config("../x.py")\nRUN_NAME = "my-run"\nSTEPS = 30\n'
    assert parse_run_name(text, "fallback") == "my-run"
    assert parse_run_name("STEPS = 30", "fallback") == "fallback"
    assert parse_run_name("", "fallback") == "fallback"


def _mk(root, rel):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x89PNG fake")
    return p


def test_sample_whitelist_ok(tmp_path):
    img = _mk(tmp_path, "runs/moss/demo/samples/step_00000010/000.png")
    r = resolve_sample_file(tmp_path, str(img))
    assert r["ok"] and r["path"] == img.resolve()
    # 相对路径亦可
    assert resolve_sample_file(tmp_path, "runs/moss/demo/samples/step_00000010/000.png")["ok"]


def test_sample_whitelist_traversal(tmp_path):
    _mk(tmp_path, "runs/moss/demo/samples/step_00000010/000.png")
    secret = tmp_path / "configs" / "user"
    secret.mkdir(parents=True)
    (secret / "evil.png").write_bytes(b"x")
    # 穿越:借 samples 目录跳出 runs/moss 白名单
    r = resolve_sample_file(tmp_path,
                            "runs/moss/demo/samples/../../../../configs/user/evil.png")
    assert not r["ok"] and r.get("forbidden")
    # 在 runs/moss 内但不在 <run>/samples 下也拒绝
    r = resolve_sample_file(tmp_path,
                            "runs/moss/demo/samples/../checkpoints/x.png")
    assert not r["ok"] and r.get("forbidden")
    # 绝对路径指向库外
    r = resolve_sample_file(tmp_path, str(secret / "evil.png"))
    assert not r["ok"] and r.get("forbidden")


def test_sample_whitelist_scope_and_ext(tmp_path):
    # samples 之外(checkpoints)拒绝
    ck = _mk(tmp_path, "runs/moss/demo/checkpoints/step_00000010/x.png")
    r = resolve_sample_file(tmp_path, str(ck))
    assert not r["ok"] and r.get("forbidden")
    # 非图片扩展名拒绝
    bad = _mk(tmp_path, "runs/moss/demo/samples/step_00000010/secret.pt")
    r = resolve_sample_file(tmp_path, str(bad))
    assert not r["ok"] and not r.get("forbidden")
    # 不存在 404 语义
    r = resolve_sample_file(tmp_path, str(tmp_path / "runs/moss/demo/samples/a/none.png"))
    assert not r["ok"] and r.get("not_found")
    # 空路径
    assert not resolve_sample_file(tmp_path, "")["ok"]


def test_extract_json_with_log_prefix_and_braces_in_strings():
    text = ('2026-10-24 01:00:00 INFO scripts.inspect_data: {"catalog": {"images": 6, '
            '"note": "brace { inside string"}, "plan": {"buckets": [1, 2]},\n'
            ' "captions": ["a } b"]}\n2026-10-24 01:00:01 INFO done\n')
    obj = extract_json(text)
    assert obj is not None
    assert obj["catalog"]["images"] == 6
    assert obj["captions"] == ["a } b"]
    assert extract_json("no json here") is None
    assert extract_json('{"unbalanced": ') is None
