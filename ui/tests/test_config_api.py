"""/api/config 端点测试:FastAPI TestClient + 假 adapter,不触碰引擎。

通过 monkeypatch 替换 app 模块级 config_service 单例。
"""

import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app as app_module
from config_service import ConfigService

FIXTURE = Path(__file__).parent / "fixtures" / "config_list.txt"


class FakeAdapter:
    def __init__(self, repo_root: Path, stdout: str = "", returncode: int = 0,
                 has_exe: bool = True) -> None:
        self.repo_root = repo_root
        self.stdout = stdout
        self.returncode = returncode
        self.has_exe = has_exe

    def find_executable(self, name: str):
        return Path("cfg-exe") if self.has_exe else None

    def _run(self, argv, timeout: float = 30):
        return subprocess.CompletedProcess(argv, self.returncode,
                                           stdout=self.stdout, stderr="")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    (tmp_path / "configs" / "user").mkdir(parents=True)
    (tmp_path / "configs" / "user" / "demo.py").write_text("RUN_NAME = \"demo\"\n", encoding="utf-8")
    svc = ConfigService(FakeAdapter(tmp_path, stdout=FIXTURE.read_text(encoding="utf-8")))
    monkeypatch.setattr(app_module, "config_service", svc)
    return TestClient(app_module.app)


def test_presets_endpoint(client):
    res = client.get("/api/config/presets")
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] and body["parsed"]
    assert "lora" in body["presets"]["moss"]["algo"]
    assert body["vram_options"] == [6, 8, 12, 16, 24, "auto"]


def test_presets_engine_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "config_service",
                        ConfigService(FakeAdapter(tmp_path, has_exe=False)))
    res = TestClient(app_module.app).get("/api/config/presets")
    assert res.status_code == 503


def test_train_endpoint_ok(client):
    res = client.post("/api/config/train", json={
        "weights": "w", "data": ["d"], "name": "fresh"})
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] and body["path"] == "configs/user/fresh.py"


def test_train_endpoint_conflict_409(client):
    res = client.post("/api/config/train", json={
        "weights": "w", "data": "d", "name": "demo"})
    assert res.status_code == 409
    assert res.json()["detail"]["exists"] is True


def test_train_endpoint_validation_200(client):
    """表单校验失败是业务结果:200 + ok=False。"""
    res = client.post("/api/config/train", json={"weights": "", "data": "d", "name": "x"})
    assert res.status_code == 200
    assert res.json()["ok"] is False


def test_check_endpoint(client):
    res = client.post("/api/config/check", json={"name": "demo"})
    assert res.status_code == 200
    assert res.json()["ok"] is True


def test_file_endpoint_whitelist(client):
    res = client.get("/api/config/file", params={"name": "demo"})
    assert res.status_code == 200
    assert "RUN_NAME" in res.json()["content"]
    # 路径穿越必须 400
    for evil in ("../evil", "..\\..\\requirements", "a/b"):
        assert client.get("/api/config/file", params={"name": evil}).status_code == 400
    assert client.get("/api/config/file", params={"name": "ghost"}).status_code == 404


def test_list_endpoint(client, tmp_path):
    """配套 _data.py 不列为用户实体:一个配置名 = 一对文件,只出主名。"""
    (tmp_path / "configs" / "user" / "demo_data.py").write_text("X = 1\n", encoding="utf-8")
    res = client.get("/api/config/list")
    assert res.status_code == 200
    assert res.json()["configs"] == ["demo"]


def test_data_companion_endpoints_400(client, tmp_path):
    """读/删配套 _data 名一律 400,防止把内部文件当实体操作。"""
    (tmp_path / "configs" / "user" / "demo_data.py").write_text("X = 1\n", encoding="utf-8")
    r = client.get("/api/config/file", params={"name": "demo_data"})
    assert r.status_code == 400 and "_data" in r.json()["detail"]
    r = client.delete("/api/config/file", params={"name": "demo_data"})
    assert r.status_code == 400 and "_data" in r.json()["detail"]
    assert (tmp_path / "configs" / "user" / "demo_data.py").is_file()


def test_delete_endpoint(client, tmp_path):
    """删除主配置 + 配套 _data.py + 持久化高级设置;404/400 与读接口同款白名单。"""
    user = tmp_path / "configs" / "user"
    (user / "demo_data.py").write_text("X = 1\n", encoding="utf-8")
    ov_dir = tmp_path / "overrides"  # conftest 已把 OVERRIDES_DIR 隔离到这里
    ov_dir.mkdir(parents=True, exist_ok=True)
    (ov_dir / "demo.json").write_text('{"LR": "1e-4"}', encoding="utf-8")

    res = client.delete("/api/config/file", params={"name": "demo"})
    assert res.status_code == 200
    body = res.json()
    assert body["ok"]
    assert sorted(body["deleted"]) == ["configs/user/demo.py", "configs/user/demo_data.py"]
    assert not (user / "demo.py").exists()
    assert not (user / "demo_data.py").exists()
    assert not (ov_dir / "demo.json").exists()

    assert client.delete("/api/config/file", params={"name": "ghost"}).status_code == 404
    assert client.delete("/api/config/file", params={"name": "../evil"}).status_code == 400
