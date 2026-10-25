"""/api/datasets 与 /api/files 端点测试:TestClient + tmp 库目录,不触碰真实 ui/state。

数据集端点是纯文件管理,**不需要引擎就绪**(不 mock adapter)。
"""

import base64

import pytest
from fastapi.testclient import TestClient

import app as app_module
from dataset_service import DatasetService

# 内存生成的最小合法 PNG(1x1)
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "dataset_service", DatasetService(tmp_path / "datasets"))
    return TestClient(app_module.app)


def _create(client, name):
    res = client.post("/api/datasets", json={"name": name})
    assert res.status_code == 200 and res.json()["ok"] is True


def test_create_list_delete_flow(client):
    _create(client, "demo")
    _create(client, "中文库")
    body = client.get("/api/datasets").json()
    assert body["ok"] and {d["name"] for d in body["datasets"]} == {"demo", "中文库"}
    # 非法名:业务结果 200 + ok=False
    assert client.post("/api/datasets", json={"name": "../evil"}).json()["ok"] is False
    # 删除必须确认
    assert client.delete("/api/datasets/demo").status_code == 400
    assert client.delete("/api/datasets/demo", params={"confirm": "true"}).status_code == 200
    assert client.delete("/api/datasets/demo", params={"confirm": "true"}).status_code == 404


def test_upload_items_caption_and_files_endpoints(client):
    _create(client, "demo")
    res = client.post("/api/datasets/demo/files", files=[
        ("files", ("a.png", PNG, "image/png")),
        ("files", ("a.txt", "一只红点".encode("utf-8"), "text/plain")),
        ("files", ("b.png", PNG, "image/png")),
        ("files", ("evil.exe", b"MZ\x90\x00", "application/octet-stream")),
    ])
    assert res.status_code == 200
    body = res.json()
    assert sorted(body["saved"]) == ["a.png", "a.txt", "b.png"]
    assert [r["name"] for r in body["rejected"]] == ["evil.exe"]
    # list 统计:image_count=2 / caption_count=1
    listing = client.get("/api/datasets").json()["datasets"][0]
    assert listing["image_count"] == 2 and listing["caption_count"] == 1
    # items
    items = client.get("/api/datasets/demo/items").json()["items"]
    assert [i["image"] for i in items] == ["a.png", "b.png"]
    assert items[0]["caption"] == "一只红点" and items[1]["caption"] is None
    # /api/files 取图 200 + 内联图片字节
    got = client.get("/api/files", params={"path": items[0]["path"]})
    assert got.status_code == 200
    assert got.content == PNG
    assert got.headers["content-type"].startswith("image/")
    # PUT caption
    res = client.put("/api/datasets/demo/caption", json={"image": "b.png", "text": "蓝点"})
    assert res.status_code == 200 and res.json()["ok"] is True
    items = client.get("/api/datasets/demo/items").json()["items"]
    assert items[1]["caption"] == "蓝点"
    # 上传到不存在的库 -> 404
    assert client.post("/api/datasets/ghost/files",
                       files=[("files", ("a.png", PNG, "image/png"))]).status_code == 404


def test_files_whitelist_traversal(client):
    _create(client, "demo")
    client.post("/api/datasets/demo/files", files=[("files", ("a.png", PNG, "image/png"))])
    # 编码/非编码的 .. 穿越必须 4xx
    for evil in ("../requirements.txt", "..%2f..%2frequirements.txt",
                 "demo/../../ui/backend/app.py", "..\\..\\configs"):
        res = client.get("/api/files", params={"path": evil})
        assert 400 <= res.status_code < 500, evil
    # 非图片扩展名拒绝
    assert client.get("/api/files", params={"path": "demo/../../ui/.gitignore"}).status_code >= 400
    # 不存在的图片 404
    assert client.get("/api/files", params={"path": "demo/ghost.png"}).status_code == 404
    # 空 path 400
    assert client.get("/api/files", params={"path": ""}).status_code == 400


def test_caption_endpoint_validation(client):
    _create(client, "demo")
    assert client.put("/api/datasets/demo/caption",
                      json={"image": "no.png", "text": "x"}).status_code == 404
    assert client.put("/api/datasets/demo/caption",
                      json={"image": "../a.png", "text": "x"}).status_code == 400
    assert client.get("/api/datasets/ghost/items").status_code == 404
