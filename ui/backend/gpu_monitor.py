"""GPU 采样(M2):pynvml 轮询利用率/显存/温度/功耗,WS 推送。

- UI 唯一自行采集的数据,与引擎无关,坏了不影响训练;
- 无 NVIDIA GPU / 驱动缺失 / pynvml 导入失败 => available=False,
  snapshot() 恒返回 None,前端隐藏 GPU 面板(优雅降级);
- 每个字段独立 try/except:单项查询失败置 None,不拖垮整帧。
"""

from __future__ import annotations


class GpuMonitor:
    """pynvml 封装;构造时完成初始化探测,运行期只读采样。"""

    def __init__(self) -> None:
        self.available = False
        self.error: str | None = None
        self.device_count = 0
        self._nvml = None
        self._handles: list = []
        try:
            import pynvml as nvml  # noqa: F401  (M0 已装;deprecation 警告忽略)
        except Exception as err:  # ImportError 或 dll 加载失败
            self.error = f"pynvml 不可用:{type(err).__name__}: {err}"
            return
        try:
            nvml.nvmlInit()
            self.device_count = nvml.nvmlDeviceGetCount()
            self._handles = [nvml.nvmlDeviceGetHandleByIndex(i)
                             for i in range(self.device_count)]
            self._nvml = nvml
            self.available = self.device_count > 0
            if not self.available:
                self.error = "未检测到 NVIDIA GPU"
        except Exception as err:
            self.error = f"NVML 初始化失败:{type(err).__name__}: {err}"
            self._handles = []
            self._nvml = None

    def _field(self, fn, *args) -> float | int | str | None:
        try:
            value = fn(*args)
            return value
        except Exception:
            return None

    def snapshot(self) -> list[dict] | None:
        """每卡一帧:{index,name,util_gpu,mem_used_mb,mem_total_mb,temp_c,power_w};
        不可用时返回 None。

        单位:pynvml 的 nvmlDeviceGetMemoryInfo 返回字节(bytes),而字段名/前端
        约定为 MiB(前端再 /1024 显示 GiB),故此处 bytes -> MiB 换算(/ 1024^2);
        功耗 NVML 返回毫瓦,同样换算为瓦。
        """
        if not self.available or self._nvml is None:
            return None
        nvml = self._nvml
        out: list[dict] = []
        for i, h in enumerate(self._handles):
            mem = self._field(nvml.nvmlDeviceGetMemoryInfo, h)
            gpu_util = self._field(nvml.nvmlDeviceGetUtilizationRates, h)
            temp = self._field(nvml.nvmlDeviceGetTemperature, h, nvml.NVML_TEMPERATURE_GPU)
            power = self._field(nvml.nvmlDeviceGetPowerUsage, h)
            name = self._field(nvml.nvmlDeviceGetName, h)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            used = getattr(mem, "used", None)
            total = getattr(mem, "total", None)
            out.append({
                "index": i,
                "name": name,
                "util_gpu": getattr(gpu_util, "gpu", None),
                "util_mem": getattr(gpu_util, "memory", None),
                "mem_used_mb": (used / (1024 * 1024)) if isinstance(used, (int, float)) else None,
                "mem_total_mb": (total / (1024 * 1024)) if isinstance(total, (int, float)) else None,
                "temp_c": temp,
                "power_w": (power / 1000.0) if isinstance(power, (int, float)) else None,
            })
        return out

    def close(self) -> None:
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
