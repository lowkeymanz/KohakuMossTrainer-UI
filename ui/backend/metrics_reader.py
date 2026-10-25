"""运行产物增量读取(M2):metrics.jsonl / log.txt / samples/ 目录。

契约(已按上游 src/kohakumoss/logging/sinks.py 核实):
- metrics.jsonl 每行 {"step": N, "time": <unix>, "loss", "grad_norm", "lr", ...};
  validation 事件以 val_loss / val_loss@<sigma> 键合并进同一步的行;
- 非 RESUME 重跑会把旧 metrics.jsonl/log.txt 移名为 <stem>.<n>(fresh_run),
  RESUME 会把晚于恢复点的行移到旁文件后重写前缀(resume_at)——两种情况
  当前文件都会"变短",本模块检测到缩短即重置偏移并按全新内容重读,
  同时置 run_reset 标志(前端据此清空曲线重建,避免新旧 run 点混叠);
- 最后一行可能写了一半:按字节偏移只消费完整行,残行留待下次;
- 同一文件内按 step 去重(resume 重写前缀时重读到旧行不重复推送);
- run_dir 在训练开始后才创建,attach 到不存在目录时静默等待。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# 样图墙允许的产物图片扩展名(引擎采样输出 .png;保留常见格式兼容)
SAMPLE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
# 引擎在 samples/ 根下生成的汇总对比表(行=步数,列=提示词),每采样一轮
# 整体重写覆盖;它是历次预览的拼图而非单张样图,且重写瞬间被浏览器抓到会
# 显示半截文件。样图墙只展示单张预览,过滤掉此汇总图。
SAMPLE_GRID = "grid.png"


def parse_metric_line(line: str) -> dict | None:
    """解析一行 metrics.jsonl;非法/半行返回 None。"""
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except ValueError:
        return None
    return obj if isinstance(obj, dict) else None


class RunWatcher:
    """对一个 run 目录的增量读取器;poll_* 返回自上次读取以来的新内容。"""

    def __init__(self) -> None:
        self.run_dir: Path | None = None
        self._metrics_offset = 0
        self._log_offset = 0
        self._fallback_offset = 0
        self._on_log_txt = False
        self._log_emitted: set[str] = set()
        self._seen_steps: set[int] = set()
        self._samples_seen: set[str] = set()
        self._config: dict | None = None
        self._config_read = False

    # ---------- 附着 ----------

    def attach(self, run_dir: str | os.PathLike | None) -> None:
        """附着到 run 目录;目录变化(含 None->路径)即全量重置。"""
        target = Path(run_dir) if run_dir else None
        if target != self.run_dir:
            self.run_dir = target
            self.reset()

    def reset(self) -> None:
        self._metrics_offset = 0
        self._log_offset = 0
        self._fallback_offset = 0
        self._on_log_txt = False
        self._log_emitted = set()
        self._seen_steps = set()
        self._samples_seen = set()
        self._config = None
        self._config_read = False

    # ---------- 内部:增量读 ----------

    def _read_new_lines(self, path: Path, offset_attr: str) -> tuple[list[str], bool]:
        """从记录的字节偏移读新完整行;返回 (行列表, 是否发生轮转/截断)。

        文件大小 < 偏移 => 文件被轮转或重写(fresh_run / resume_at),
        重置偏移从头重读(调用方按 run_reset 清屏,行内 step 去重防重复)。
        """
        if self.run_dir is None:
            return [], False
        try:
            if not path.is_file():
                return [], False
            size = path.stat().st_size
        except OSError:
            return [], False
        offset = getattr(self, offset_attr)
        rotated = size < offset
        if rotated:
            offset = 0
        try:
            with open(path, "rb") as f:
                f.seek(offset)
                data = f.read()
        except OSError:
            return [], False
        # 只消费完整行:残行(无换行结尾)留在原地,下次再读
        cut = data.rfind(b"\n")
        if cut < 0:
            return [], rotated
        consumed = data[: cut + 1]
        setattr(self, offset_attr, offset + len(consumed))
        text = consumed.decode("utf-8", errors="replace")
        # splitlines 兼容 \r\n(Windows 文本模式追加会转 CRLF,二进制切片残留 \r)
        lines = [ln for ln in text.splitlines() if ln.strip()]
        return lines, rotated

    # ---------- 公开轮询 ----------

    def poll_metrics(self) -> tuple[list[dict], bool]:
        """新 metrics 行(按 step 去重,step 缺失的行原样保留);含轮转标志。"""
        if self.run_dir is None:
            return [], False
        lines, rotated = self._read_new_lines(self.run_dir / "metrics.jsonl",
                                              "_metrics_offset")
        if rotated:
            # 轮转/重写后按全新内容处理:清去重集,整文件重新推送
            # (前端收到 run_reset 事件清屏,再按重推行重建曲线)
            self._seen_steps = set()
        out: list[dict] = []
        for ln in lines:
            row = parse_metric_line(ln)
            if row is None:
                continue
            step = row.get("step")
            if isinstance(step, int):
                if step in self._seen_steps:
                    continue
                self._seen_steps.add(step)
            out.append(row)
        return out, rotated

    def poll_log(self, fallback_path: str | os.PathLike | None = None) -> list[str]:
        """日志新行:任务原始 stdout/stderr 与 run 目录 log.txt 每次都读并合并。

        关键:引擎 logger 的文件 handler 只把 INFO 行写进 log.txt,而进程崩溃时
        的 Python traceback、logging 初始化前的 C++ 警告等只进任务原始输出落盘
        文件(ui/state/logs/<id>.log)。若 log.txt 一出现就停读原始输出(旧实现的
        bug),崩溃栈永远进不了 UI。两个源都按各自偏移增量读,再按内容去重:
        原始输出是 log.txt 的超集(log.txt 的行会被去重掉),log.txt 若含原始输出
        没有的行也能补齐。log.txt 轮转(fresh_run)时清空去重集整文件重推。"""
        lines: list[str] = []
        if fallback_path:
            fb, _ = self._read_new_lines(Path(fallback_path), "_fallback_offset")
            lines.extend(fb)
        if self.run_dir is not None and (self.run_dir / "log.txt").is_file():
            self._on_log_txt = True
            lt, rotated = self._read_new_lines(self.run_dir / "log.txt",
                                               "_log_offset")
            if rotated:
                self._log_emitted = set()  # fresh_run 轮转后整文件重推
            lines.extend(lt)
        out: list[str] = []
        for ln in lines:
            key = ln.strip()
            if not key or key in self._log_emitted:
                continue
            self._log_emitted.add(key)
            if len(self._log_emitted) > 20000:  # 防长任务内存无界增长
                self._log_emitted = set()
            out.append(ln)
        return out

    def poll_samples(self) -> list[str]:
        """samples/ 下新出现的图片文件(绝对路径 posix 字符串,稳定排序)。

        过滤掉 samples/ 根下的汇总对比表 grid.png(见 `SAMPLE_GRID`),
        样图墙只展示 step_X/ 目录内的单张预览。
        """
        if self.run_dir is None:
            return []
        root = self.run_dir / "samples"
        try:
            if not root.is_dir():
                return []
            found = sorted((p for p in root.rglob("*")
                            if p.is_file() and p.suffix.lower() in SAMPLE_EXTS
                            and p.name != SAMPLE_GRID),
                           key=lambda p: str(p).lower())
        except OSError:
            return []
        out: list[str] = []
        for p in found:
            key = str(p).lower()
            if key in self._samples_seen:
                continue
            self._samples_seen.add(key)
            out.append(str(p).replace("\\", "/"))
        return out

    # ---------- run 的 config.json(最终生效旋钮) ----------

    def read_config(self) -> dict | None:
        """读 run 目录 config.json(成功一次后缓存);不存在/损坏返回 None。"""
        if self.run_dir is None or self._config_read:
            return self._config
        try:
            text = (self.run_dir / "config.json").read_text(encoding="utf-8")
            obj = json.loads(text)
            self._config = obj if isinstance(obj, dict) else None
        except (OSError, ValueError):
            self._config = None
        if self._config is not None:
            self._config_read = True
        return self._config
