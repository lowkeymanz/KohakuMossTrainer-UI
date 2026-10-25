"""训练参数默认值服务(高级设置面板数据源)。

- 默认值权威来源 = **运行引擎的真实资产根**(rev2 §2 L1):定位走"产物通道"——
  从引擎生成配置(configs/user/*.py)docstring 打印的 `kogine run <脚本> --config`
  反推脚本路径;绝对路径(含 site-packages/.../bundle)→ wheel 形态资产根,
  仓库相对路径 → checkout 形态资产根(仓库 configs/+scripts/+src/)。无产物可反推
  时按引擎 validate.repo_root 的同款结构规则探测,再回退仓库路径,响应里以
  `source_of_truth` 标注(bundle / checkout / fallback)。
- 解析全部为**只读文本解析**(ast/正则,绝不 import 引擎模块;铁律 2 允许的
  只读耦合,引擎改文件布局时由契约测试金样兜底)。
- 组合顺序与引擎 compose() 一致:script globals → base_lora/base_full →
  algo_* → data_<source> → vram_* → precision_*;后层覆盖前层,layer_source
  记录最后设置该旋钮的层文件。
- 旋钮全集/枚举/改名表来自 knobs.py 的 KNOWN/CHOICES/RENAMED(ast 字面量);
  导入型枚举(MEMORY_PROFILE/TUNE/PRECISION)按 rev2 §2 L2 从 profiles.py /
  kohakumoss_kernels tuner.py / precision config.py 文本取。
- override 合法键 = KNOWN ∪ ENCODER_FIELDS(前缀+字段),RENAMED 旧名放行并提示
  (不得严于引擎,引擎侧最终报 unknown knob)。
- 结果 TTL 缓存(同 presets 的 PRESETS_TTL 模式)。
"""

from __future__ import annotations

import ast
import json
import re
import time
from pathlib import Path

# defaults 解析结果缓存时长(秒);纯文本解析代价低,但页面刷新频繁
DEFAULTS_TTL = 60.0

TRAIN_SCRIPT_REL = "scripts/train/moss.py"
# 配置 docstring 打印的权威运行命令(train_service 同款契约,此处独立小正则避免循环 import)
_RUN_CMD_RE = re.compile(r"kogine run (\S*scripts/train/moss\.py)")
_USE_CONFIG_RE = re.compile(r'^use_config\("([^"]+)"\)')
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# ---------- 高级设置分区(rev2 §3.2:6 组实列 43 项,其余归"专家"只读) ----------

GROUP_LABELS = {
    "lora": "LoRA 结构",
    "optim": "优化与调度",
    "sample": "采样与验证",
    "data": "数据与缓存",
    "memory": "显存与并行",
    "export": "导出与日志",
    "expert": "专家(只读清单 + 自由键值)",
}

PANEL: dict[str, str] = {
    # LoRA 结构(7)
    "DIM": "lora", "ALPHA": "lora", "PRESET": "lora", "DROPOUT": "lora",
    "DORA": "lora", "DORA_DETACH": "lora", "ALGO_KWARGS": "lora",
    # 优化与调度(8)
    "OPTIMIZER": "optim", "LR": "optim", "BETAS": "optim",
    "WEIGHT_DECAY": "optim", "WD_RULES": "optim", "SCHEDULE": "optim",
    "GRAD_CLIP": "optim", "EMA": "optim",
    # 采样与验证(8)
    "SAMPLE_EVERY": "sample", "SAMPLE_STEPS": "sample", "SAMPLE_CFG": "sample",
    "SAMPLE_SHIFT": "sample", "SAMPLE_NEGATIVE": "sample", "SAMPLE_SIZE": "sample",
    "SAMPLE_SEEDS": "sample", "VALIDATE_EVERY": "sample",
    # 数据与缓存(10)
    "CAPTION": "data", "TARGET_AREA": "data", "RESOLUTION_MODE": "data",
    "ASPECT_RANGE": "data", "BATCHING": "data", "MAX_TOKENS": "data",
    "TEXT_DROP": "data", "LATENT_CACHE": "data", "TEXT_CACHE": "data",
    "CACHE_DTYPE": "data",
    # 显存与并行(5)
    "MEMORY_PROFILE": "memory", "MEMORY_OVERRIDES": "memory",
    "GRAD_ACC": "memory", "PARALLEL": "memory", "DEVICES": "memory",
    # 导出与日志(5)
    "SAVE_EVERY": "export", "SAVE_MINUTES": "export", "KEEP_LAST": "export",
    "SAVE_FORMAT": "export", "LOGGER": "export",
}

KNOB_NOTES = {
    "MEMORY_PROFILE": "随显存档位变化(auto 时非 balanced)",
    "CAPTION": "触发词在生成期写进 prefix;启动期 --set CAPTION 覆盖时以覆盖为准",
    "RESUME": "不进面板:训练页勾选「续训」自动 --set RESUME=latest",
    "EXPORT_FORMATS": "旧拼写:请改用 SAVE_FORMAT",
    "STEPS": "基础区已提供(引擎默认 4000;冒烟建议 200 步,需大于 warmup)",
    "ALGO": "基础区已提供(引擎默认 lora)",
    "PRECISION": "基础区已提供(留空=按 GPU 自动检测;bf16 为引擎默认档)",
    "WEIGHTS": "基础区已提供",
    "DATASETS": "基础区已提供(数据集多选)",
    "RUN_NAME": "基础区已提供(LoRA 名称)",
    "SAMPLE_PROMPTS": "基础区已提供(采样 prompt)",
    "SAMPLE_SIZE": "采样 prompt 行的分辨率框会自动写进配置文件(平行列表);此处手动覆盖只支持单一 (高, 宽)",
}

# ---------- overrides 持久化(rev2 §4.3:B1 方案) ----------

OVERRIDES_DIR = Path(__file__).resolve().parents[1] / "state" / "overrides"


def _safe_name(name: str) -> str | None:
    name = (name or "").strip()
    if name.endswith(".py"):
        name = name[:-3]
    return name if name and _NAME_RE.fullmatch(name) else None


def override_path(name: str) -> Path | None:
    stem = _safe_name(name)
    return OVERRIDES_DIR / f"{stem}.json" if stem else None


def save_overrides(name: str, overrides: dict) -> None:
    """按配置名持久化 deferred overrides;空覆盖删除旧文件(防重生成后残留)。"""
    path = override_path(name)
    if path is None:
        return
    try:
        if overrides:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({str(k): str(v) for k, v in overrides.items()},
                                       ensure_ascii=False, indent=1), encoding="utf-8")
        else:
            path.unlink(missing_ok=True)
    except OSError:
        pass  # 持久化失败不阻断生成(启动侧读不到即空)


def load_overrides(name: str) -> dict:
    path = override_path(name)
    if path is None or not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


# ---------- ast 文本解析工具 ----------

def _literal(node: ast.AST):
    """字面量求值;支持引擎层文件常见的数字乘表达式(1024*1024)与 256<<20;
    失败返回 None(调用方回退 ast.unparse 文本)。"""
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        pass
    if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Mult, ast.Div, ast.Add, ast.Sub, ast.LShift, ast.RShift)):
        left, right = _literal(node.left), _literal(node.right)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            ops = {ast.Mult: lambda: left * right, ast.Div: lambda: left / right,
                   ast.Add: lambda: left + right, ast.Sub: lambda: left - right,
                   ast.LShift: lambda: int(left) << int(right),
                   ast.RShift: lambda: int(left) >> int(right)}
            for typ, fn in ops.items():
                if isinstance(node.op, typ):
                    return fn()
    return None


def parse_module_globals(path: Path) -> dict[str, tuple[object, str]] | None:
    """模块级 UPPER_CASE 赋值 -> {名: (值或 None, 原文文本)};文件坏/缺失返回 None。"""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return None
    out: dict[str, tuple[object, str]] = {}
    for node in tree.body:
        targets: list[ast.expr] = []
        value: ast.expr | None = None
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets, value = [node.target], node.value
        if value is None:
            continue
        for t in targets:
            if isinstance(t, ast.Name) and t.id.isupper() and not t.id.startswith("_"):
                out[t.id] = (_literal(value), ast.unparse(value))
    return out


def _find_assign(tree: ast.Module, name: str) -> ast.expr | None:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return node.value
    return None


def _class_fields(path: Path, class_name: str) -> list[str] | None:
    """dataclass 字段名(类体 AnnAssign 目标);失败 None。"""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError):
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return [t.target.id for t in node.body
                    if isinstance(t, ast.AnnAssign) and isinstance(t.target, ast.Name)
                    and t.target.id.islower()]
    return None


def _knob_type(value: object, text: str) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, dict):
        return "dict"
    if isinstance(value, list):
        return "list"
    if isinstance(value, tuple):
        return "tuple"
    if value is None:
        return "none"
    return "text"


def _display(value: object, text: str) -> str:
    """面板回填与 diff 用的字符串形态(与 --set 下发值一致)。"""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(list(value) if isinstance(value, tuple) else value,
                          ensure_ascii=False)
    if value is None:
        return "None"
    return text


class DefaultsService:
    """运行引擎默认值/旋钮元数据的解析门面;所有公开方法返回结构化 dict。"""

    def __init__(self, adapter) -> None:
        self.adapter = adapter
        self._cache: dict[tuple, tuple[float, dict]] = {}

    # ---------- 资产根定位(产物通道优先,结构探测兜底) ----------

    @staticmethod
    def _has_assets(root: Path) -> bool:
        return (root / "configs").is_dir() and (root / "scripts").is_dir()

    def _package_dirs(self, root: Path, kind: str) -> tuple[Path, Path]:
        """(kohakumoss 包目录, kohakumoss_kernels 包目录)。"""
        repo = self.adapter.repo_root
        if kind == "checkout":
            return (repo / "src" / "kohakumoss", repo / "src" / "kohakumoss_kernels")
        # bundle 形态:root = .../site-packages/kohakumoss/bundle
        site = root.parents[1] if kind == "bundle" else repo / "src"
        return (site / "kohakumoss", site / "kohakumoss_kernels")

    def locate_asset_root(self) -> tuple[Path | None, str, str | None]:
        """返回 (资产根, source_of_truth, 说明)。"""
        repo = self.adapter.repo_root
        # 1) 产物通道:任一已生成配置的 docstring 运行命令
        user_dir = repo / "configs" / "user"
        try:
            candidates = sorted(user_dir.glob("*.py"))
        except OSError:
            candidates = []
        for cfg in candidates:
            try:
                text = cfg.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            m = _RUN_CMD_RE.search(text)
            if not m:
                continue
            raw = m.group(1).replace("\\", "/")
            if re.match(r"^[A-Za-z]:/", raw) or raw.startswith("/"):
                root = Path(raw).parents[2]  # 去掉 scripts/train/moss.py
                if self._has_assets(root):
                    return root, "bundle", f"由 {cfg.name} docstring 运行命令反推"
            else:
                if self._has_assets(repo):
                    return repo, "checkout", f"由 {cfg.name} docstring 运行命令反推(仓库相对路径)"
        # 2) 结构探测(与引擎 validate.repo_root 同款规则)
        if (repo / "src" / "kohakumoss" / "__init__.py").is_file() and self._has_assets(repo):
            return repo, "checkout", "src/ 存在:引擎按仓库源码运行(UI engine_env 前置 PYTHONPATH)"
        bundle = self._probe_bundle()
        if bundle is not None:
            return bundle, "bundle", "按 venv site-packages bundle 布局探测"
        # 3) 回退仓库路径(可能过时,响应已标注)
        if self._has_assets(repo):
            return repo, "fallback", "无法定位运行引擎资产根,回退仓库路径(可能与引擎实际版本分叉)"
        return None, "fallback", "仓库与 venv 均无 configs/scripts,无法解析默认值"

    def _probe_bundle(self) -> Path | None:
        venv = getattr(self.adapter, "venv_dir", None)
        if venv is None:
            return None
        try:
            hits = list(Path(venv).glob(
                "Lib/site-packages/kohakumoss/bundle"))
            hits += list(Path(venv).glob(
                "lib/python*/site-packages/kohakumoss/bundle"))
        except OSError:
            return None
        for b in hits:
            if self._has_assets(b):
                return b
        return None

    # ---------- knobs.py 元数据 ----------

    def _knob_meta(self, package: Path, kernels: Path) -> dict | None:
        """缓存键 "_meta:<package>":KNOWN/CHOICES/RENAMED/ENCODER_FIELDS/ALGOS。"""
        now = time.monotonic()
        hit = self._cache.get(("_meta", str(package)))
        if hit and now - hit[0] < DEFAULTS_TTL:
            return hit[1]
        knobs_file = package / "training" / "knobs.py"
        try:
            tree = ast.parse(knobs_file.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError):
            return None
        meta: dict = {}
        known = _find_assign(tree, "KNOWN")
        if (isinstance(known, ast.Call) and known.args
                and isinstance(known.args[0], ast.Call)
                and isinstance(known.args[0].func, ast.Attribute)
                and known.args[0].func.attr == "split"
                and isinstance(known.args[0].func.value, ast.Constant)):
            meta["KNOWN"] = frozenset(str(known.args[0].func.value.value).split())
        choices_node = _find_assign(tree, "CHOICES")
        choices: dict[str, list] = {}
        if isinstance(choices_node, ast.Dict):
            for k, v in zip(choices_node.keys, choices_node.values):
                if not isinstance(k, ast.Constant):
                    continue
                val = _literal(v)
                if isinstance(val, (tuple, list)):
                    choices[str(k.value)] = [str(x) for x in val]
                elif str(k.value) == "MEMORY_PROFILE":
                    choices["MEMORY_PROFILE"] = self._profile_choices(package)
                elif str(k.value) == "TUNE":
                    choices["TUNE"] = self._tune_choices(kernels)
        # OPTIMIZER 不在 knobs.py CHOICES 里,引擎运行时按 factory 注册表校验
        if "OPTIMIZER" not in choices:
            choices["OPTIMIZER"] = self._optimizer_choices(package)
        meta["CHOICES"] = choices
        algos = _literal(_find_assign(tree, "ALGOS"))
        if isinstance(algos, tuple):
            meta["ALGOS"] = [str(a) for a in algos]
        # RENAMED:字面量条目 + knob_names 的 RENAMED_SAMPLE / SAMPLE_NAMES 展开
        renamed: dict[str, str] = {}
        node = _find_assign(tree, "RENAMED")
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and isinstance(v, ast.Constant):
                    renamed[str(k.value)] = str(v.value)
        names_file = package / "encoders" / "knob_names.py"
        encoder_fields: dict[str, dict] = {}
        try:
            ntree = ast.parse(names_file.read_text(encoding="utf-8", errors="replace"))
            rs = _literal(_find_assign(ntree, "RENAMED_SAMPLE"))
            if isinstance(rs, dict):
                for k, v in rs.items():
                    renamed.setdefault(str(k), str(v))
            sample_names = _literal(_find_assign(ntree, "SAMPLE_NAMES"))
            if isinstance(sample_names, tuple):
                for n in sample_names:
                    renamed.setdefault(f"SAMPLE_{n}",
                                       f"the train script's name is SAMPLE_{n}")
            ef = _find_assign(ntree, "ENCODER_FIELDS")
            if isinstance(ef, ast.Dict):
                for k, v in zip(ef.keys, ef.values):
                    if isinstance(k, ast.Constant) and isinstance(v, ast.Tuple) and len(v.elts) == 2:
                        knob = v.elts[0]
                        fields = self._encoder_fields(package, v.elts[1])
                        if isinstance(knob, ast.Constant) and fields is not None:
                            encoder_fields[str(k.value)] = {"knob": str(knob.value),
                                                            "fields": fields}
        except (OSError, SyntaxError):
            pass
        meta["RENAMED"] = renamed
        meta["ENCODER_FIELDS"] = encoder_fields
        self._cache[("_meta", str(package))] = (now, meta)
        return meta

    def _encoder_fields(self, package: Path, node: ast.expr) -> list[str] | None:
        """ENCODER_FIELDS 值里的字段来源:dataclass 定义文件按 import 路径解析。
        (TextEncoderPlan -> types/plans.py,VaePlan -> offload/encoders/vae.py,
        实测锚点;找不到类则返回 None 由前缀正则降级。)"""
        text = ast.unparse(node)
        if "TextEncoderPlan" in text:
            return _class_fields(package / "types" / "plans.py", "TextEncoderPlan")
        if "VaePlan" in text:
            return _class_fields(package / "offload" / "encoders" / "vae.py", "VaePlan")
        return None

    def _profile_choices(self, package: Path) -> list[str]:
        """MEMORY_PROFILE 枚举:profiles.py 的 ORDER 字面量 + auto/manual
        (catalog 长尾名经 /api/config/presets 的 memory_profiles 提供)。"""
        try:
            tree = ast.parse((package / "memory" / "profiles.py")
                             .read_text(encoding="utf-8", errors="replace"))
            order = _literal(_find_assign(tree, "ORDER"))
            if isinstance(order, tuple):
                return [str(o) for o in order] + ["auto", "manual"]
        except (OSError, SyntaxError):
            pass
        return []

    def _tune_choices(self, kernels: Path) -> list[str]:
        try:
            tree = ast.parse((kernels / "tuning" / "tuner.py")
                             .read_text(encoding="utf-8", errors="replace"))
            modes = _literal(_find_assign(tree, "MODES"))
            if isinstance(modes, tuple):
                return [str(m) for m in modes]
        except (OSError, SyntaxError):
            pass
        return []

    def _optimizer_choices(self, package: Path) -> list[str]:
        """OPTIMIZER 枚举:引擎运行时白名单是 factory.py 的 @OPTIMIZERS.register
        注册表(knobs.py 校验循环),不是 CHOICES 字典。INTERNAL 里的名字
        (cpu_adamw/master_adamw/host_state_adamw)是 OPTIM_STATE 自动选择的
        变体,factory 文档明确"never named by OPTIMIZER",故剔除。"""
        try:
            tree = ast.parse((package / "training" / "optim" / "optimizers" / "factory.py")
                             .read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError):
            return []
        internal = _literal(_find_assign(tree, "INTERNAL")) or {}
        names: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            for dec in node.decorator_list:
                if (isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute)
                        and dec.func.attr == "register"
                        and isinstance(dec.func.value, ast.Name)
                        and dec.func.value.id == "OPTIMIZERS"
                        and dec.args and isinstance(dec.args[0], ast.Constant)):
                    n = str(dec.args[0].value)
                    if n not in internal and n not in names:
                        names.append(n)
        return names

    def _precision_choices(self, package: Path) -> list[str]:
        try:
            tree = ast.parse((package / "precision" / "config.py")
                             .read_text(encoding="utf-8", errors="replace"))
            compute = _literal(_find_assign(tree, "COMPUTE"))
            if isinstance(compute, tuple):
                return [str(c) for c in compute] + ["auto"]
        except (OSError, SyntaxError):
            pass
        return ["bf16", "fp16", "auto"]

    # ---------- 默认值组合 ----------

    def _layer_files(self, root: Path, source: str, algo: str, vram: str,
                     precision: str) -> list[tuple[str, Path]]:
        base = "base_full" if algo == "full" else "base_lora"
        out: list[tuple[str, Path]] = [
            ("script", root / TRAIN_SCRIPT_REL),
            (f"_base/{base}", root / "configs" / "moss" / "_base" / f"{base}.py"),
        ]
        moss = root / "configs" / "moss"
        if algo and _NAME_RE.fullmatch(algo):
            out.append((f"algo/algo_{algo}", moss / "algo" / f"algo_{algo}.py"))
        if source and _NAME_RE.fullmatch(source):
            out.append((f"data/data_{source}", moss / "data" / f"data_{source}.py"))
        vram_name = "auto" if vram in ("", "auto") else str(vram)
        if vram_name not in ("",) and re.fullmatch(r"(auto|\d+(\.\d+)?g?)", vram_name):
            name = vram_name if vram_name == "auto" or vram_name.endswith("g") \
                else f"{vram_name}g"
            out.append((f"vram/vram_{name}", moss / "vram" / f"vram_{name}.py"))
        if precision and _NAME_RE.fullmatch(precision):
            out.append((f"precision/precision_{precision}",
                        moss / "precision" / f"precision_{precision}.py"))
        return out

    def get_defaults(self, source: str = "kohya", algo: str = "lora",
                     vram: str = "auto", precision: str = "") -> dict:
        """组合默认值 + 旋钮元数据;TTL 缓存。"""
        key = (source, algo, vram, precision)
        now = time.monotonic()
        hit = self._cache.get(key)
        if hit and now - hit[0] < DEFAULTS_TTL:
            return hit[1]
        result = self._get_defaults_uncached(source, algo, vram, precision)
        self._cache[key] = (now, result)
        return result

    def _get_defaults_uncached(self, source: str, algo: str, vram: str,
                               precision: str) -> dict:
        root, sot, note = self.locate_asset_root()
        if root is None:
            return {"ok": False, "error": note, "source_of_truth": sot}
        package, kernels = self._package_dirs(root, sot)
        meta = self._knob_meta(package, kernels)
        if not meta or "KNOWN" not in meta:
            return {"ok": False, "error": f"无法解析旋钮元数据({package})",
                    "source_of_truth": sot}
        merged: dict[str, tuple[object, str]] = {}
        layer_source: dict[str, str] = {}
        missing_layers: list[str] = []
        for label, path in self._layer_files(root, source, algo, vram, precision):
            parsed = parse_module_globals(path)
            if parsed is None:
                if label != "script":
                    missing_layers.append(label)  # 可选层缺失(如未选精度层)不算故障
                continue
            for k, v in parsed.items():
                merged[k] = v
                layer_source[k] = label
        choices = dict(meta.get("CHOICES") or {})
        if meta.get("ALGOS"):
            choices["ALGO"] = meta["ALGOS"]
        choices["PRECISION"] = self._precision_choices(package)
        knobs: dict[str, dict] = {}
        for name in sorted(meta["KNOWN"]):
            value, text = merged.get(name, (None, ""))
            group = PANEL.get(name, "expert")
            knobs[name] = {
                "default": value if isinstance(value, (str, int, float, bool)) or value is None
                else list(value) if isinstance(value, tuple) else value,
                "default_text": _display(value, text) if name in merged else "",
                "type": _knob_type(value, text) if name in merged else "unknown",
                "choices": choices.get(name),
                "group": group,
                "editable": group != "expert",
                "layer_source": layer_source.get(name),
                "note": KNOB_NOTES.get(name),
            }
        return {
            "ok": True, "error": None,
            "source_of_truth": sot, "locate_note": note,
            "asset_root": str(root).replace("\\", "/"),
            "query": {"source": source, "algo": algo, "vram": vram,
                      "precision": precision},
            "missing_layers": missing_layers,
            "groups": GROUP_LABELS,
            "panel": [n for n in knobs if n in PANEL],
            "knobs": knobs,
            "renamed": meta.get("RENAMED") or {},
            "encoder_fields": meta.get("ENCODER_FIELDS") or {},
            "valid_key_count": len(meta["KNOWN"]) + sum(
                len(e["fields"]) for e in (meta.get("ENCODER_FIELDS") or {}).values()),
        }

    # ---------- override 键校验(new_train 与 start 双侧共用) ----------

    def check_override_keys(self, overrides: dict) -> tuple[list[str], list[str]]:
        """返回 (非法键错误列表, RENAMED 提示列表)。元数据不可用时返回空表(不拦)。"""
        if not overrides:
            return [], []
        root, sot, _ = self.locate_asset_root()
        if root is None:
            return [], []
        package, kernels = self._package_dirs(root, sot)
        meta = self._knob_meta(package, kernels)
        if not meta or "KNOWN" not in meta:
            return [], []
        known = meta["KNOWN"]
        renamed = meta.get("RENAMED") or {}
        encoder = meta.get("ENCODER_FIELDS") or {}
        errors: list[str] = []
        warnings: list[str] = []
        for key in overrides:
            key = str(key)
            if key in known:
                continue
            hit = False
            for prefix, spec in encoder.items():
                if key.startswith(prefix) and key[len(prefix):].lower() in spec["fields"]:
                    warnings.append(f"{key}:等价于 {spec['knob']}['{key[len(prefix):].lower()}']")
                    hit = True
                    break
            if hit:
                continue
            if key in renamed:
                warnings.append(f"{key}:旧名({renamed[key]}),已放行;引擎可能拒绝")
                continue
            errors.append(key)
        return errors, warnings
