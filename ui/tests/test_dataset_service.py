"""dataset_service 单元测试:建库/删库/上传过滤/caption 读写/items 统计/文件白名单。

全部在 tmp_path 里操作,**不触碰真实 ui/state/datasets**。
"""

from pathlib import Path

from dataset_service import DatasetService

# 1x1 红色 PNG(与 API 测试共用)
PNG = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
       b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\xcf"
       b"\xc0\x00\x00\x00\x03\x00\x01\x00\x00\x00\x00IEND\xaeB`\x82")


def svc(tmp_path) -> DatasetService:
    return DatasetService(tmp_path / "datasets")


# ---------- 建库 / 名称校验 ----------

def test_create_and_list(tmp_path):
    s = svc(tmp_path)
    assert s.create("demo")["ok"] is True
    assert s.create("中文库-1_x")["ok"] is True
    res = s.list_datasets()
    assert res["ok"] and [d["name"] for d in res["datasets"]] == ["demo", "中文库-1_x"]
    assert res["datasets"][0] == {"name": "demo", "image_count": 0,
                                  "caption_count": 0, "created": res["datasets"][0]["created"]}


def test_create_rejects_bad_names(tmp_path):
    s = svc(tmp_path)
    for bad in ("", "   ", "../evil", "..", "a/b", "a\\b", "a.b", "带 空格",
                "x" * 65, "emoji😀", "C:\\abs"):
        assert s.create(bad)["ok"] is False, bad


def test_create_duplicate(tmp_path):
    s = svc(tmp_path)
    assert s.create("demo")["ok"] is True
    assert s.create("demo")["ok"] is False


def test_delete_requires_confirm(tmp_path):
    s = svc(tmp_path)
    s.create("demo")
    assert s.delete("demo")["ok"] is False  # 未确认
    assert (tmp_path / "datasets" / "demo").is_dir()
    assert s.delete("demo", confirm=True)["ok"] is True
    assert not (tmp_path / "datasets" / "demo").exists()
    assert s.delete("ghost", confirm=True)["not_found"] is True


# ---------- 上传过滤 ----------

def test_upload_accepts_images_and_txt_rejects_rest(tmp_path):
    s = svc(tmp_path)
    s.create("demo")
    res = s.upload("demo", [
        ("a.png", PNG), ("b.jpg", PNG), ("c.txt", b"caption"),
        ("evil.exe", b"MZ"), ("no_ext", b"x"), ("d.txtx", b"x"),
        ("sub/../../escape.png", b"x"), ("", b"x"),
    ])
    assert res["ok"] is True
    assert sorted(res["saved"]) == ["a.png", "b.jpg", "c.txt"]
    assert {r["name"] for r in res["rejected"]} == {"evil.exe", "no_ext", "d.txtx",
                                                    "sub/../../escape.png", ""}
    d = tmp_path / "datasets" / "demo"
    assert (d / "a.png").read_bytes() == PNG
    assert not (d / "evil.exe").exists()
    assert not (d / "escape.png").exists()


def test_upload_overwrite_flag(tmp_path):
    s = svc(tmp_path)
    s.create("demo")
    s.upload("demo", [("a.png", PNG)])
    res = s.upload("demo", [("a.png", PNG), ("new.webp", PNG)])
    assert res["overwritten"] == ["a.png"]
    assert res["saved"] == ["a.png", "new.webp"]


def test_upload_missing_dataset(tmp_path):
    s = svc(tmp_path)
    res = s.upload("ghost", [("a.png", PNG)])
    assert res["ok"] is False and res["not_found"] is True


# ---------- items / caption ----------

def test_items_and_caption_roundtrip(tmp_path):
    s = svc(tmp_path)
    s.create("demo")
    s.upload("demo", [("a.png", PNG), ("b.png", PNG), ("a.txt", "红色像素".encode("utf-8"))])
    res = s.items("demo")
    assert res["ok"] is True
    assert [i["image"] for i in res["items"]] == ["a.png", "b.png"]
    assert res["items"][0]["caption"] == "红色像素"
    assert res["items"][1]["caption"] is None
    # /api/files 用的 path 是绝对路径且指向图片
    assert Path(res["items"][0]["path"]).is_file()
    # 保存 caption(UTF-8)
    save = s.save_caption("demo", "b.png", "蓝色 像素\n第二行")
    assert save["ok"] is True
    txt = tmp_path / "datasets" / "demo" / "b.txt"
    assert txt.read_text(encoding="utf-8") == "蓝色 像素\n第二行"
    assert s.items("demo")["items"][1]["caption"].startswith("蓝色")
    # caption 覆盖率进 list_datasets
    listing = s.list_datasets()["datasets"][0]
    assert listing == {"name": "demo", "image_count": 2, "caption_count": 2,
                       "created": listing["created"]}


def test_save_caption_validation(tmp_path):
    s = svc(tmp_path)
    s.create("demo")
    assert s.save_caption("demo", "ghost.png", "x")["not_found"] is True
    assert s.save_caption("demo", "evil.exe", "x")["ok"] is False
    assert s.save_caption("demo", "../a.png", "x")["ok"] is False
    assert s.save_caption("ghost", "a.png", "x")["not_found"] is True


# ---------- /api/files 白名单 ----------

def test_resolve_file_whitelist(tmp_path):
    s = svc(tmp_path)
    s.create("demo")
    s.upload("demo", [("a.png", PNG), ("secret.txt", b"no")])
    base = tmp_path / "datasets"
    # 相对库根
    assert s.resolve_file("demo/a.png")["ok"] is True
    # 绝对路径(前端 items 返回的形态)
    assert s.resolve_file(str((base / "demo" / "a.png").resolve()))["ok"] is True
    # 穿越:相对/绝对/.. 一律拒绝
    (tmp_path / "outside.png").write_bytes(PNG)
    for evil in ("../outside.png", "..\\..\\outside.png", "demo/../../outside.png",
                 str(tmp_path / "outside.png"), ""):
        res = s.resolve_file(evil)
        assert res["ok"] is False, evil
    # 只放行图片扩展名
    assert s.resolve_file("demo/secret.txt")["ok"] is False
    assert s.resolve_file("demo/ghost.png")["not_found"] is True


def test_dataset_dir_for_config(tmp_path):
    s = svc(tmp_path)
    assert s.dataset_dir("ghost", must_exist=True)[1]
    s.create("demo")
    target, err = s.dataset_dir("demo", must_exist=True)
    assert err is None and target.name == "demo"
    assert s.dataset_dir("../evil")[1]
