"""run 产物访问(M2):runs/moss/<RUN_NAME> 定位、RUN_NAME 解析、样图白名单。

- 只读上游产物目录,不写不删(删除 run 交给文件管理器,需求文档 §6.10);
- RUN_NAME 从引擎生成的 configs/user/<name>.py 文本正则提取(不 import 引擎);
- 样图白名单:仅 runs/moss/<run>/samples/ 子树内的图片文件,
  resolve 后复核前缀,防路径穿越(沿用 datasets 白名单思路)。
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

# 采样产物图片扩展名(与 metrics_reader.SAMPLE_EXTS 一致)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# 配置文件文本里的 RUN_NAME = "..."(引擎 starter 以 json.dumps 字面量写出)
_RUN_NAME_RE = re.compile(r'^RUN_NAME\s*=\s*"([^"]+)"', re.M)


def runs_root(repo_root: str | os.PathLike) -> Path:
    """训练产物根目录(OUTPUT_DIR 默认值 runs/moss,相对仓库根)。"""
    return Path(repo_root) / "runs" / "moss"


def parse_run_name(config_text: str, fallback: str) -> str:
    """从配置 .py 原文提取 RUN_NAME;缺省回退到配置名(fallback)。"""
    m = _RUN_NAME_RE.search(config_text or "")
    if m and m.group(1).strip():
        return m.group(1).strip()
    return fallback


def resolve_run_dir(repo_root: str | os.PathLike, run_name: str) -> Path:
    """run 名 -> run 目录(名字来自配置解析,这里仅做基本防穿越)。"""
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", (run_name or "").strip()) or "run"
    return runs_root(repo_root) / safe


def resolve_sample_file(repo_root: str | os.PathLike, raw: str) -> dict:
    """把请求路径解析为某 run 的 samples/ 下的图片(供 GET /api/run-files)。

    白名单 = 仓库根 runs/moss/<任意 run>/samples/ 子树:相对路径按仓库根解析,
    绝对路径也允许,但 resolve 后必须仍在某 run 的 samples/ 内;只放行图片扩展名。
    """
    raw = (raw or "").strip()
    if not raw:
        return {"ok": False, "error": "path 不能为空"}
    root = Path(repo_root).resolve()
    base = runs_root(root)
    p = Path(raw)
    if not p.is_absolute():
        p = root / p
    try:
        rp = p.resolve()
    except OSError as err:
        return {"ok": False, "error": f"路径解析失败:{err}"}
    # 必须位于 runs/moss/<run>/samples/ 内(samples 本身不算)
    try:
        rel = rp.relative_to(base)
    except ValueError:
        return {"ok": False, "forbidden": True,
                "error": "路径越出 runs/moss/ 白名单,已拒绝"}
    parts = rel.parts
    if len(parts) < 3 or parts[1] != "samples":
        return {"ok": False, "forbidden": True,
                "error": "仅允许访问 runs/moss/<run>/samples/ 下的文件"}
    if rp.suffix.lower() not in IMAGE_EXTS:
        return {"ok": False, "error": "仅允许图片文件(png/jpg/jpeg/webp/bmp)"}
    if not rp.is_file():
        return {"ok": False, "not_found": True, "error": f"文件不存在:{raw}"}
    return {"ok": True, "path": rp}


def read_run_config(run_dir: str | os.PathLike) -> dict | None:
    """读 run 目录 config.json(最终生效旋钮);不存在/损坏返回 None。"""
    try:
        text = (Path(run_dir) / "config.json").read_text(encoding="utf-8")
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except (OSError, ValueError):
        return None


# ---------- M3:checkpoint 浏览 / run 摘要 / 导出产物 ----------

# checkpoint 四件套(io.py 实测契约):trainable 必含,ema 仅启用 EMA 时有,
# optim+state 齐备才算"完整可续训";state.json 最后写,存在即标记该步落盘完成。
CKPT_REQUIRED = ("trainable.safetensors", "optim.pt", "state.json")
CKPT_OPTIONAL = ("ema.safetensors",)

_STEP_DIR_RE = re.compile(r"^step_(\d+)$")


def _file_info(path: Path) -> dict | None:
    """单文件摘要 {name, size, mtime};不存在返回 None。"""
    try:
        st = path.stat()
    except OSError:
        return None
    if not path.is_file():
        return None
    return {"name": path.name, "size": st.st_size,
            "mtime": int(st.st_mtime)}


def checkpoint_dirs(run_dir: str | os.PathLike) -> list[Path]:
    """run 目录下 step_* checkpoint 目录,按步数升序。"""
    ckpt_root = Path(run_dir) / "checkpoints"
    found: list[tuple[int, Path]] = []
    try:
        for d in ckpt_root.iterdir():
            m = _STEP_DIR_RE.match(d.name)
            if m and d.is_dir():
                found.append((int(m.group(1)), d))
    except OSError:
        return []
    return [d for _, d in sorted(found)]


def find_checkpoint_dir(run_dir: str | os.PathLike, ref) -> Path | None:
    """checkpoint 引用 -> 目录:接受目录名(step_00000040)、纯步数(40)或
    None(最新可用步,即引擎 usable 判定的最后一个);找不到返回 None。"""
    dirs = checkpoint_dirs(run_dir)
    if not dirs:
        return None
    if ref is None or str(ref).strip() == "":
        for d in reversed(dirs):  # 步数降序取第一个 usable
            if (d / "state.json").is_file():
                return d
        return dirs[-1]
    ref = str(ref).strip()
    for d in dirs:
        if d.name == ref:
            return d
    if ref.isdigit():
        for d in dirs:
            m = _STEP_DIR_RE.match(d.name)
            if m and int(m.group(1)) == int(ref):
                return d
    return None


def list_checkpoints(repo_root: str | os.PathLike, run_name: str) -> dict:
    """某 run 的全部 checkpoint:四件套存在性/大小 + usable/complete + export 产物。

    usable = state.json + trainable(引擎恢复判定);
    complete = trainable + optim + state(UI 展示的"完整"角标)。
    """
    run_dir = resolve_run_dir(repo_root, run_name)
    if not run_dir.is_dir():
        return {"ok": False, "not_found": True, "run": run_name,
                "checkpoints": [], "error": f"run 不存在:runs/moss/{run_name}"}
    out: list[dict] = []
    for d in checkpoint_dirs(run_dir):
        m = _STEP_DIR_RE.match(d.name)
        files = {}
        for name in (*CKPT_REQUIRED, *CKPT_OPTIONAL):
            info = _file_info(d / name)
            if info:
                files[name] = info
        exports = _export_listing(d / "export")
        out.append({
            "dir": d.name,
            "step": int(m.group(1)) if m else None,
            "files": files,
            "has_trainable": "trainable.safetensors" in files,
            "has_ema": "ema.safetensors" in files,
            "usable": (d / "state.json").is_file()
                      and (d / "trainable.safetensors").is_file(),
            "complete": all((d / n).is_file() for n in CKPT_REQUIRED),
            "exports": exports,
            "path": d.as_posix(),
        })
    return {"ok": True, "run": run_name, "run_dir": run_dir.as_posix(),
            "checkpoints": out, "error": None}


def _export_listing(export_dir: Path) -> list[dict]:
    """checkpoint 的 export/ 子目录产物:文件 + diffusers 文件夹(记子项)。"""
    out: list[dict] = []
    try:
        entries = sorted(export_dir.iterdir(), key=lambda p: p.name.lower())
    except OSError:
        return out
    for p in entries:
        if p.is_file():
            info = _file_info(p)
            if info:
                info["path"] = p.as_posix()
                out.append(info)
        elif p.is_dir():  # diffusers 布局:文件夹本身作为一项
            info = {"name": p.name, "size": None, "folder": True,
                    "mtime": int(p.stat().st_mtime), "path": p.as_posix()}
            out.append(info)
    return out


def list_export_files(repo_root: str | os.PathLike, run_name: str) -> list[dict]:
    """某 run 的全部导出产物:run/export/ + 各 checkpoint 的 export/ 子目录。"""
    run_dir = resolve_run_dir(repo_root, run_name)
    found: list[dict] = []
    for info in _export_listing(run_dir / "export"):
        found.append({**info, "checkpoint": None})
    for d in checkpoint_dirs(run_dir):
        for info in _export_listing(d / "export"):
            found.append({**info, "checkpoint": d.name})
    return found


def list_runs(repo_root: str | os.PathLike) -> dict:
    """扫描 runs/moss/:每个 run 的摘要(配置快照关键旋钮、checkpoint 数、
    导出文件、采样图数、最后修改时间),按修改时间倒序。"""
    base = runs_root(repo_root)
    if not base.is_dir():
        return {"ok": True, "runs": [], "error": None}
    out: list[dict] = []
    try:
        entries = [p for p in base.iterdir() if p.is_dir()]
    except OSError as err:
        return {"ok": False, "runs": [], "error": f"列 runs/moss 失败:{err}"}
    for d in entries:
        cfg = read_run_config(d) or {}
        steps = [int(m.group(1)) for p in checkpoint_dirs(d)
                 if (m := _STEP_DIR_RE.match(p.name))]
        try:
            n_samples = sum(1 for p in (d / "samples").iterdir()
                            if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        except OSError:
            n_samples = 0
        exports = list_export_files(repo_root, d.name)
        try:
            mtime = int(d.stat().st_mtime)
        except OSError:
            mtime = 0
        out.append({
            "run": d.name,
            "run_dir": d.as_posix(),
            "algo": cfg.get("ALGO"),
            "steps": cfg.get("STEPS"),
            "weights": cfg.get("WEIGHTS"),
            "sample_every": cfg.get("SAMPLE_EVERY"),
            "checkpoint_count": len(steps),
            "last_step": max(steps) if steps else None,
            "export_count": len(exports),
            "sample_count": n_samples,
            "mtime": mtime,
        })
    out.sort(key=lambda r: r["mtime"], reverse=True)
    return {"ok": True, "runs": out, "error": None}


def resolve_export_file(repo_root: str | os.PathLike, raw: str) -> dict:
    """把请求路径解析为导出产物文件(供 GET /api/export-files 下载)。

    白名单 = runs/moss/<run>/export/ 与 runs/moss/<run>/checkpoints/step_*/export/
    两棵子树;resolve 后复核前缀,防路径穿越。导出文件扩展名多样
    (safetensors/模型分片等),不按图片白名单过滤,但必须是普通文件。
    """
    raw = (raw or "").strip()
    if not raw:
        return {"ok": False, "error": "path 不能为空"}
    root = Path(repo_root).resolve()
    base = runs_root(root)
    p = Path(raw)
    if not p.is_absolute():
        p = root / p
    try:
        rp = p.resolve()
    except OSError as err:
        return {"ok": False, "error": f"路径解析失败:{err}"}
    try:
        rel = rp.relative_to(base)
    except ValueError:
        return {"ok": False, "forbidden": True,
                "error": "路径越出 runs/moss/ 白名单,已拒绝"}
    parts = rel.parts
    ok_subtree = (
        (len(parts) >= 3 and parts[1] == "export")
        or (len(parts) >= 4 and _STEP_DIR_RE.match(parts[2]) and parts[3] == "export")
    )
    if not ok_subtree:
        return {"ok": False, "forbidden": True,
                "error": "仅允许 runs/moss/<run>/export/ 或 "
                         "checkpoints/step_*/export/ 下的导出产物"}
    if not rp.is_file():
        return {"ok": False, "not_found": True, "error": f"文件不存在:{raw}"}
    return {"ok": True, "path": rp}
