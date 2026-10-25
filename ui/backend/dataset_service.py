"""数据集库服务(M1.5):UI 自管的本地数据集目录,aitoolkit 式建库/上传/预览/编辑 caption。

- 库目录固定 ui/state/datasets/<名称>/(state/ 在 ui/.gitignore 内,不碰上游、不写仓库其他位置);
- 纯文件管理,不调引擎、不 import 引擎模块(铁律);
- 名称严格白名单(字母/数字/下划线/连字符/中文,1-64 字符),所有路径 resolve 后校验
  仍在库根内,防路径穿越;
- 图片(png/jpg/jpeg/webp/bmp)与同名 .txt caption 配对;其余扩展名一律拒绝并报告;
- 所有公开方法返回结构化 dict(ok/error/...),异常全部捕获,绝不抛出。
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
import re

# 库根:ui/state/datasets(与 adapter.state_dir 同层)
DATASETS_ROOT = Path(__file__).resolve().parents[1] / "state" / "datasets"

# 收编的扩展名:图片 + .txt caption(同名配对)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
CAPTION_EXT = ".txt"

# 数据集名白名单:字母/数字/下划线/连字符/中文,长度 1-64;
# 不含点与任何分隔符 => 天然排除 "." ".." "/" "\" 与隐藏文件
_NAME_RE = re.compile(r"^[A-Za-z0-9_\u4e00-\u9fff-]{1,64}$")


class DatasetService:
    """数据集库的文件管理封装;所有公开方法返回结构化 dict,绝不抛异常。"""

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self.base_dir = Path(base_dir) if base_dir else DATASETS_ROOT

    # ---------- 内部工具 ----------

    @staticmethod
    def _invalid_name(name: str) -> str | None:
        """名称校验;合法返回 None,非法返回原因。"""
        name = (name or "").strip()
        if not name:
            return "数据集名称不能为空"
        if not _NAME_RE.fullmatch(name):
            return ("数据集名称只允许字母/数字/下划线/连字符/中文,长度 1-64,"
                    "不允许点号与路径分隔符")
        return None

    def _safe_dir(self, name: str) -> tuple[Path | None, str | None]:
        """name -> 库内目录的安全解析(白名单 + resolve 前缀复核,防穿越)。"""
        err = self._invalid_name(name)
        if err:
            return None, err
        base = self.base_dir.resolve()
        target = (base / name.strip()).resolve()
        if not target.is_relative_to(base):
            return None, "数据集路径越出库目录,已拒绝"
        return target, None

    @staticmethod
    def _images_in(dataset_dir: Path) -> list[Path]:
        """库内全部图片文件(递归),按名排序。"""
        try:
            return sorted((p for p in dataset_dir.rglob("*")
                           if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
                          key=lambda p: str(p).lower())
        except OSError:
            return []

    @staticmethod
    def _caption_of(image: Path) -> str | None:
        """同名 .txt caption 文本;不存在返回 None。"""
        txt = image.with_suffix(CAPTION_EXT)
        if not txt.is_file():
            return None
        try:
            return txt.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def dataset_dir(self, name: str, must_exist: bool = False) -> tuple[Path | None, str | None]:
        """库名 -> 库内目录(供 config_service 解析 datasets 选择)。"""
        target, err = self._safe_dir(name)
        if err:
            return None, err
        if must_exist and not target.is_dir():
            return None, f"数据集 {name} 不在库中(请先在数据集页建库并上传)"
        return target, None

    # ---------- 库级操作 ----------

    def list_datasets(self) -> dict:
        """扫描库目录:每项 {name, image_count, caption_count(同名 .txt 覆盖), created}。"""
        base = self.base_dir
        if not base.is_dir():
            return {"ok": True, "datasets": [], "error": None}
        out: list[dict] = []
        try:
            entries = sorted((p for p in base.iterdir() if p.is_dir()),
                             key=lambda p: p.name.lower())
        except OSError as err:
            return {"ok": False, "datasets": [], "error": f"列库目录失败:{err}"}
        for d in entries:
            images = self._images_in(d)
            captioned = sum(1 for p in images if p.with_suffix(CAPTION_EXT).is_file())
            try:
                created = datetime.fromtimestamp(d.stat().st_ctime).strftime("%Y-%m-%d %H:%M")
            except OSError:
                created = ""
            out.append({"name": d.name, "image_count": len(images),
                        "caption_count": captioned, "created": created})
        return {"ok": True, "datasets": out, "error": None}

    def create(self, name: str) -> dict:
        """建库;名称非法 400 语义(ok=False),已存在报错。"""
        target, err = self._safe_dir(name)
        if err:
            return {"ok": False, "error": err}
        if target.exists():
            return {"ok": False, "error": f"数据集 {target.name} 已存在"}
        try:
            target.mkdir(parents=True)
        except OSError as e:
            return {"ok": False, "error": f"建库失败:{e}"}
        return {"ok": True, "name": target.name, "error": None}

    def delete(self, name: str, confirm: bool = False) -> dict:
        """删库(连图片与 caption 一并删除);必须显式 confirm。"""
        if not confirm:
            return {"ok": False, "error": "删除数据集需要确认参数 confirm=true"}
        target, err = self._safe_dir(name)
        if err:
            return {"ok": False, "error": err}
        if not target.is_dir():
            return {"ok": False, "not_found": True, "error": f"数据集 {name} 不存在"}
        try:
            shutil.rmtree(target)
        except OSError as e:
            return {"ok": False, "error": f"删除失败:{e}"}
        return {"ok": True, "name": target.name, "error": None}

    # ---------- 文件操作 ----------

    def upload(self, name: str, files: list[tuple[str, bytes]]) -> dict:
        """保存上传文件:只收图片与 .txt;其余拒绝并在 rejected 中报告。

        files 为 (原始文件名, 字节) 列表;文件名含路径分隔符的一律拒绝(浏览器
        multipart 本应只带裸文件名,出现分隔符视为构造的恶意输入)。
        同名覆盖并记入 overwritten。
        """
        target, err = self._safe_dir(name)
        if err:
            return {"ok": False, "error": err}
        if not target.is_dir():
            return {"ok": False, "not_found": True,
                    "error": f"数据集 {name} 不存在(请先建库)"}
        saved: list[str] = []
        overwritten: list[str] = []
        rejected: list[dict] = []
        for raw_name, data in files:
            fname = (raw_name or "").strip()
            if not fname or "/" in fname or "\\" in fname or fname != Path(fname).name:
                rejected.append({"name": fname, "reason": "文件名含路径分隔符或为空,已拒绝"})
                continue
            ext = Path(fname).suffix.lower()
            if ext not in IMAGE_EXTS and ext != CAPTION_EXT:
                rejected.append({"name": fname,
                                 "reason": f"不支持的扩展名 {ext or '(无)'}(仅图片/txt)"})
                continue
            dest = target / fname
            if not dest.resolve().is_relative_to(target.resolve()):
                rejected.append({"name": fname, "reason": "路径越出数据集目录,已拒绝"})
                continue
            existed = dest.is_file()
            try:
                dest.write_bytes(data)
            except OSError as e:
                rejected.append({"name": fname, "reason": f"写入失败:{e}"})
                continue
            saved.append(fname)
            if existed:
                overwritten.append(fname)
        return {"ok": True, "name": target.name, "saved": saved,
                "overwritten": overwritten, "rejected": rejected, "error": None}

    def items(self, name: str) -> dict:
        """列出库内图片(文件名 + 绝对路径供 /api/files 用)+ 各自 caption 文本。"""
        target, err = self._safe_dir(name)
        if err:
            return {"ok": False, "items": [], "error": err}
        if not target.is_dir():
            return {"ok": False, "items": [], "not_found": True,
                    "error": f"数据集 {name} 不存在"}
        out = []
        for img in self._images_in(target):
            out.append({"image": img.name,
                        "path": str(img.resolve()).replace("\\", "/"),
                        "caption": self._caption_of(img)})
        return {"ok": True, "name": target.name, "items": out, "error": None}

    def save_caption(self, name: str, image: str, text: str) -> dict:
        """写图片同名 .txt(UTF-8);image 必须是库内已存在的图片文件名。"""
        target, err = self._safe_dir(name)
        if err:
            return {"ok": False, "error": err}
        if not target.is_dir():
            return {"ok": False, "not_found": True, "error": f"数据集 {name} 不存在"}
        fname = (image or "").strip()
        if (not fname or "/" in fname or "\\" in fname or fname != Path(fname).name
                or Path(fname).suffix.lower() not in IMAGE_EXTS):
            return {"ok": False, "error": "image 必须是库内图片的裸文件名(图片扩展名)"}
        img = target / fname
        if not img.is_file():
            return {"ok": False, "not_found": True, "error": f"图片不存在:{fname}"}
        txt = img.with_suffix(CAPTION_EXT)
        try:
            txt.write_text(text or "", encoding="utf-8")
        except OSError as e:
            return {"ok": False, "error": f"写入 caption 失败:{e}"}
        return {"ok": True, "name": target.name, "image": fname,
                "caption_file": txt.name, "error": None}

    # ---------- /api/files 白名单 ----------

    def resolve_file(self, raw: str) -> dict:
        """把请求路径解析为库内图片文件(供 GET /api/files)。

        白名单 = 仅 ui/state/datasets/ 前缀:相对路径按库根解析,绝对路径也允许,
        但 resolve 后必须仍在库根内;只放行图片扩展名。
        """
        raw = (raw or "").strip()
        if not raw:
            return {"ok": False, "error": "path 不能为空"}
        base = self.base_dir.resolve()
        p = Path(raw)
        if not p.is_absolute():
            p = base / p
        try:
            rp = p.resolve()
        except OSError as e:
            return {"ok": False, "error": f"路径解析失败:{e}"}
        if not rp.is_relative_to(base):
            return {"ok": False, "forbidden": True,
                    "error": "数据集路径不合法(越出库目录),已拒绝"}
        if rp.suffix.lower() not in IMAGE_EXTS:
            return {"ok": False, "error": "仅允许图片文件(png/jpg/jpeg/webp/bmp)"}
        if not rp.is_file():
            return {"ok": False, "not_found": True, "error": f"文件不存在:{raw}"}
        return {"ok": True, "path": rp}
