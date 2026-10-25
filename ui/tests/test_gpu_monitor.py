"""GpuMonitor 优雅降级测试:不依赖真实 GPU/NVML。"""

import sys
import types

import pytest

from gpu_monitor import GpuMonitor


class _Util:
    gpu = 42
    memory = 7


class _Mem:
    # 真实 pynvml 的 nvmlDeviceGetMemoryInfo 返回字节;此处用字节值(4096/8192 MiB)
    # 才能守住后端的 bytes -> MiB 换算(见 test_snapshot 断言 mem_used_mb == 4096)。
    used = 4096 * 1024 * 1024
    total = 8192 * 1024 * 1024


def make_fake_nvml(count=1, fail_fields=()):
    """构造假 pynvml:字段查询可指定抛异常,验证单项容错。"""
    mod = types.ModuleType("pynvml")
    mod.NVML_TEMPERATURE_GPU = 0

    def nvmlInit():
        return None

    def nvmlDeviceGetCount():
        return count

    def nvmlDeviceGetHandleByIndex(i):
        return f"handle-{i}"

    def _getter(name, value):
        def fn(handle, *a):
            if name in fail_fields:
                raise RuntimeError(name)
            return value
        return fn

    mod.nvmlInit = nvmlInit
    mod.nvmlDeviceGetCount = nvmlDeviceGetCount
    mod.nvmlDeviceGetHandleByIndex = nvmlDeviceGetHandleByIndex
    mod.nvmlDeviceGetMemoryInfo = _getter("mem", _Mem())
    mod.nvmlDeviceGetUtilizationRates = _getter("util", _Util())
    mod.nvmlDeviceGetTemperature = _getter("temp", 61)
    mod.nvmlDeviceGetPowerUsage = _getter("power", 12345)  # 毫瓦
    mod.nvmlDeviceGetName = _getter("name", b"Fake RTX 4060")
    mod.nvmlShutdown = lambda: None
    return mod


@pytest.fixture()
def clean_pynvml():
    saved = sys.modules.get("pynvml")
    yield
    if saved is not None:
        sys.modules["pynvml"] = saved
    else:
        sys.modules.pop("pynvml", None)


def test_snapshot_with_fake_nvml(clean_pynvml):
    sys.modules["pynvml"] = make_fake_nvml(count=2)
    m = GpuMonitor()
    assert m.available and m.device_count == 2
    snap = m.snapshot()
    assert len(snap) == 2
    g = snap[0]
    assert g["name"] == "Fake RTX 4060"
    assert g["util_gpu"] == 42
    assert g["mem_used_mb"] == 4096
    assert g["temp_c"] == 61
    assert g["power_w"] == pytest.approx(12.345)


def test_field_failure_degrades_to_none(clean_pynvml):
    """单项查询抛异常:该字段 None,整帧仍可用。"""
    sys.modules["pynvml"] = make_fake_nvml(fail_fields=("temp", "power"))
    m = GpuMonitor()
    assert m.available
    g = m.snapshot()[0]
    assert g["temp_c"] is None and g["power_w"] is None
    assert g["util_gpu"] == 42


def test_no_devices_not_available(clean_pynvml):
    sys.modules["pynvml"] = make_fake_nvml(count=0)
    m = GpuMonitor()
    assert m.available is False
    assert m.snapshot() is None


def test_import_failure_graceful(clean_pynvml):
    """pynvml 不可导入(无包/无驱动 dll):available=False,snapshot 恒 None。

    sys.modules 置 None 使 import pynvml 直接抛 ImportError(标准禁用手法)。
    """
    sys.modules["pynvml"] = None
    m = GpuMonitor()
    assert m.available is False
    assert "pynvml" in (m.error or "")
    assert m.snapshot() is None


def test_nvml_init_failure_graceful(clean_pynvml):
    mod = make_fake_nvml()
    def boom():
        raise RuntimeError("NVML_ERROR_DRIVER_NOT_LOADED")
    mod.nvmlInit = boom
    sys.modules["pynvml"] = mod
    m = GpuMonitor()
    assert m.available is False
    assert "NVML" in (m.error or "")
    assert m.snapshot() is None
