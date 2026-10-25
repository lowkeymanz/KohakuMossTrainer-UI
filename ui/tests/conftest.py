import sys
from pathlib import Path

import pytest

# 让测试可以直接 import backend 模块(扁平模式)
_BACKEND = Path(__file__).resolve().parents[1] / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


@pytest.fixture(autouse=True)
def _isolate_overrides_dir(tmp_path, monkeypatch):
    """全局隔离 overrides 持久化目录:任何测试都不读写真实 ui/state/overrides。"""
    import defaults_service
    monkeypatch.setattr(defaults_service, "OVERRIDES_DIR", tmp_path / "overrides")
