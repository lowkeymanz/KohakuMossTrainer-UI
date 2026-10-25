"""配置闭环服务(M1):封装 kohakumoss-config 的 list / new train / new sample / --from / check。

- 一切经 EngineAdapter._run() 子进程,不 import 引擎模块;
- flag 名称与参数严格对齐上游 src/kohakumoss/cli/config/main.py:
  `new train` 实际支持 --weights --data --source --algo --gpus --name --steps
  --prompt(可重复) --vram --vram-tolerance --precision --out --force,
  **没有 --set**(那是 kogine run 的覆盖机制,训练启动时生效,M2 接入);
- `--data` 上游为单值,多数据集目录在生成期不支持(需编辑生成文件的 DATASETS);
- 所有方法返回结构化 dict(ok/error/stdout/...),异常全部捕获,绝不抛出;
- 预设/算法/显存档位一律运行时从 `config list` 解析,UI 不硬编码清单。
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import time
from pathlib import Path

# 兼容两种启动方式:包模式与扁平模式(测试用)
try:
    from .dataset_service import DATASETS_ROOT, DatasetService
    from .defaults_service import DefaultsService, override_path, save_overrides
except ImportError:  # pragma: no cover
    from dataset_service import DATASETS_ROOT, DatasetService
    from defaults_service import DefaultsService, override_path, save_overrides

# 引擎 CLI 生成的配置目录(引擎自己写;UI 可删除,不提供直接改写)
USER_DIR = "configs/user"

# config 命令都要先加载 torch(探测硬件/组合旋钮),30s 默认超时不够用
LIST_TIMEOUT = 300
GEN_TIMEOUT = 600
CHECK_TIMEOUT = 300
# config list 解析结果缓存时长(秒);见 list_presets
PRESETS_TTL = 60.0

# 配置文件名白名单:字母数字开头,允许 . _ -,防路径穿越
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


# ---------- `config list` 纯文本解析 ----------

def _split_names(text: str) -> list[str]:
    return [t.strip() for t in text.split(",") if t.strip()]


def parse_config_list(text: str) -> dict:
    """把 `kohakumoss-config list [--all]` 的纯文本解析为结构化 dict。

    上游输出格式(已按 main.py list_presets 与真实输出核实):
      recipes          a, b, c
      <domain>/<group 补位到 10> name, name, ...
      (空行) memory profiles (...) 标题 + Markdown 表格
      (可选) <N> catalog profiles more: ...
    解析失败时 parsed=False 并保留原文,不抛异常(降级展示)。
    """
    out: dict = {"recipes": [], "presets": {}, "memory_profiles": [],
                 "catalog_more": None, "parsed": False}
    in_table = False
    header: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        m = re.match(r"^recipes\s+(.+)$", s)
        if m:
            out["recipes"] = _split_names(m.group(1))
            out["parsed"] = True
            continue
        m = re.match(r"^([A-Za-z_]+)/([A-Za-z_]+)\s+(.+)$", s)
        if m:
            domain, group, names = m.groups()
            out["presets"].setdefault(domain, {})[group] = _split_names(names)
            out["parsed"] = True
            continue
        if s.startswith("memory profiles"):
            in_table = True
            continue
        m = re.match(r"^(\d+) catalog profiles more", s)
        if m:
            out["catalog_more"] = int(m.group(1))
            continue
        if in_table and s.startswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if not header:
                header = cells
                continue
            if all(re.fullmatch(r":?-+:?", c) for c in cells):
                continue  # 分隔行
            row = dict(zip(header, cells))
            name = row.get("profile", "").strip("`")
            if not name:
                continue
            profile = {"name": name}
            for key, col in (("checkpoint", "checkpoint"), ("weights", "weights"),
                             ("act_offload", "act_offload"), ("optimizer", "optimizer"),
                             ("grad_offload", "grad_offload")):
                profile[key] = row.get(col, "")
            for key, col in (("lora_gib", "LoRA GiB"), ("full_ft_gib", "full FT GiB")):
                raw = row.get(col, "")
                try:
                    profile[key] = float(raw)
                except ValueError:
                    profile[key] = raw
            out["memory_profiles"].append(profile)
            out["parsed"] = True
    return out


def vram_options(presets: dict) -> list:
    """从 moss/vram 预设名推导 UI 可传的 --vram GiB 档位(不硬编码清单);
    *_full 层由引擎按算法自动选择,UI 只传 GiB 数字(需求文档 §5 页面 C)。"""
    names = (presets.get("moss") or {}).get("vram") or []
    sizes: list = []
    auto = False
    for n in names:
        if n == "auto":
            auto = True
            continue
        m = re.fullmatch(r"(\d+)g(_full)?", n)
        if m and not m.group(2):
            sizes.append(int(m.group(1)))
    sizes.sort()
    return sizes + (["auto"] if auto else [])


# ---------- 参数数组构造(与上游 flag 一一对应,单测覆盖) ----------

def build_train_argv(exe: str | Path, form: dict) -> tuple[list[str], str | None]:
    """表单 -> `kohakumoss-config new train` 参数数组;返回 (argv, 校验错误)。"""
    weights = (form.get("weights") or "").strip()
    if not weights:
        return [], "权重目录(weights)不能为空"
    data = form.get("data")
    if isinstance(data, str):
        data = [data]
    data = [d.strip() for d in (data or []) if d and d.strip()]
    if not data:
        return [], "数据集目录(data)不能为空"
    if len(data) > 1 and not form.get("datasets"):
        return [], ("引擎 `new train` 仅接受单个 --data 数据集目录;"
                    "多数据集请在生成后编辑配置文件的 DATASETS 列表")
    # datasets(从库选,M1.5):--data 只吃第一个,其余由 new_train 追加进 DATASETS
    name = (form.get("name") or "").strip()
    if not name:
        return [], "LoRA 名称不能为空"
    if not _NAME_RE.fullmatch(name):
        return [], "LoRA 名称只允许字母/数字/._-,且不能含路径分隔符"
    if name.endswith("_data"):
        # 引擎会产出 <name>.py + <name>_data.py 一对文件,_data 后缀与
        # 配套文件命名规则冲突(主配置会被 UI 过滤成不可见),生成期即拒绝
        return [], "LoRA 名称不能以 _data 结尾(与引擎配套数据配置命名冲突)"

    argv: list[str] = [str(exe), "new", "train", "--weights", weights,
                       "--data", data[0], "--name", name]
    if form.get("algo"):
        argv += ["--algo", str(form["algo"])]
    if form.get("source"):
        argv += ["--source", str(form["source"])]
    steps = form.get("steps")
    if steps not in (None, "", 0, "0"):
        argv += ["--steps", str(int(steps))]
    prompts = form.get("prompt")
    if isinstance(prompts, str):
        prompts = [prompts]
    for p in prompts or []:
        if p and str(p).strip():
            argv += ["--prompt", str(p)]
    vram = form.get("vram")
    if vram not in (None, ""):
        argv += ["--vram", str(vram)]
    if form.get("precision"):
        argv += ["--precision", str(form["precision"])]
    gpus = form.get("gpus")
    if gpus not in (None, "", 1, "1"):
        argv += ["--gpus", str(int(gpus))]
    if form.get("force"):
        argv += ["--force"]
    # 注意:上游 new train 没有 --set;overrides 由 new_train 作为
    # deferred_overrides 返回(训练启动时经 kogine --set 生效),不进 argv。
    return argv, None


def build_sample_argv(exe: str | Path, form: dict) -> tuple[list[str], str | None]:
    """表单 -> `kohakumoss-config new sample` 参数数组。"""
    weights = (form.get("weights") or "").strip()
    if not weights:
        return [], "权重路径(weights)不能为空"
    name = (form.get("name") or "sample").strip()
    if not _NAME_RE.fullmatch(name):
        return [], "配置文件名只允许字母/数字/._-"
    argv: list[str] = [str(exe), "new", "sample", "--weights", weights, "--name", name]
    for a in form.get("adapter") or []:
        if a and str(a).strip():
            argv += ["--adapter", str(a).strip()]
    if form.get("samples"):
        argv += ["--samples", str(form["samples"])]
    prompts = form.get("prompt")
    if isinstance(prompts, str):
        prompts = [prompts]
    for p in prompts or []:
        if p and str(p).strip():
            argv += ["--prompt", str(p)]
    vram = form.get("vram")
    if vram in (None, "", "auto"):
        return [], "采样必须给出 --vram GiB 数字(引擎不支持 auto)"
    argv += ["--vram", str(vram)]
    if form.get("precision"):
        argv += ["--precision", str(form["precision"])]
    if form.get("force"):
        argv += ["--force"]
    return argv, None


def build_from_argv(exe: str | Path, base: str, out: str, force: bool = False) -> tuple[list[str], str | None]:
    """`kohakumoss-config new --from <recipe> <out>` 参数数组。"""
    if not base or not base.strip():
        return [], "--from 预设名(base)不能为空"
    if not out or not out.strip():
        return [], "输出路径(out)不能为空"
    argv = [str(exe), "new", "--from", base.strip(), out.strip()]
    if force:
        argv.append("--force")
    return argv, None


# ---------- 多数据集:向生成文件的 DATASETS 列表追加条目(M1.5) ----------

def append_datasets_to_config(text: str, extra_paths: list[str]) -> tuple[str | None, str | None]:
    """在引擎生成的 `<name>.py` 的 DATASETS 列表末尾追加数据集(最小文本插入)。

    引擎 `new train` 的 --data 为单值,实测(M1.5)DATASETS 列表生成在主配置文件
    (不在 _data.py,_data.py 仅 use_config 主文件 + 缓存编码器)。追加方式:
    复制首个条目块(保留引擎写入的全部旋钮),仅替换其中的 "path" 值,插到列表
    闭合 `]` 之前。解析失败返回错误,绝不产出半损坏的文件。
    返回 (新文本, 错误)。
    """
    m = re.search(r"^DATASETS = \[\n", text, re.M)
    if not m:
        return None, "生成的配置中未找到 DATASETS 列表,无法追加数据集"
    i = text.find("{", m.end())
    if i < 0:
        return None, "DATASETS 列表中没有条目可复制"
    depth = 0
    j = -1
    for k in range(i, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                j = k
                break
    if j < 0:
        return None, "DATASETS 首条目花括号不配平,拒绝改写"
    close = re.search(r"^\]", text[m.end():], re.M)
    if not close:
        return None, "未找到 DATASETS 列表的闭合 ],拒绝改写"
    close_at = m.end() + close.start()
    line_start = text.rfind("\n", 0, i) + 1
    indent = text[line_start:i]
    first_entry = text[i:j + 1]
    additions = ""
    for p in extra_paths:
        norm = str(p).replace("\\", "/")
        if '"' in norm:
            return None, f"数据集路径含引号,无法安全写入:{norm}"
        entry = re.sub(r'"path":\s*"[^"]*"', lambda _m: f'"path": "{norm}"',
                       first_entry, count=1)
        if entry == first_entry:
            return None, "DATASETS 首条目缺少 path 字段,无法追加"
        additions += indent + entry + ",\n"
    return text[:close_at] + additions + text[close_at:], None


# ---------- 触发词:tags caption 模板 prefix(kohya tag word 的等价机制) ----------

# 引擎无专门触发词旋钮(实测 src 无 trigger);但 tags 模板支持 prefix/suffix
# (kohakumoss_data/caption/template.py TagTemplate),prefix 会加到每条训练
# caption 开头,无需改动数据集 txt 文件。

_CAPTION_RE = re.compile(r"^CAPTION = \{(?=\n)", re.M)


def _layer_caption_dict(repo_root: Path, source: str) -> tuple[dict | None, str | None]:
    """读数据格式层 configs/moss/data/data_<source>.py 的 CAPTION dict
    (ast 解析纯文本,不 import 引擎)。返回 (dict, 错误)。"""
    if not source or not _NAME_RE.fullmatch(source):
        return None, f"数据格式名不合法,无法读取 caption 层:{source!r}"
    layer = repo_root / "configs" / "moss" / "data" / f"data_{source}.py"
    try:
        text = layer.read_text(encoding="utf-8")
    except OSError as err:
        return None, f"无法读取数据格式层 {layer.name}:{err}"
    try:
        tree = ast.parse(text)
    except SyntaxError as err:
        return None, f"数据格式层 {layer.name} 解析失败:{err}"
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "CAPTION" for t in node.targets
        ):
            try:
                return ast.literal_eval(node.value), None
            except (ValueError, TypeError) as err:
                return None, f"数据格式层 CAPTION 取值解析失败:{err}"
    return None, f"数据格式层 {layer.name} 中没有 CAPTION,无法注入触发词"


def apply_trigger_to_config(text: str, trigger: str,
                            layer_caption: dict | None) -> tuple[str | None, str | None]:
    """把触发词注入生成的配置文本,返回 (新文本, 错误)。

    生成文件已有 CAPTION 块 -> 原地插入/替换其 prefix(仅 tags 模板);
    没有(引擎默认把 CAPTION 留在数据格式层)-> 整份追加"格式层 CAPTION + prefix",
    避免只写 prefix 而丢掉 keep/tag_dropout 等格式层旋钮。绝不产出半损坏文件。
    """
    trigger = (trigger or "").strip()
    if not trigger:
        return text, None
    if '"' in trigger or "\n" in trigger or "\r" in trigger:
        return None, "触发词含引号或换行,无法安全写入配置"
    prefix = f"{trigger}, "
    m = _CAPTION_RE.search(text)
    if m:
        depth = 1  # m.end() 在 '{' 之后,首层 '{' 已计入
        j = -1
        for k in range(m.end(), len(text)):
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
                if depth == 0:
                    j = k
                    break
        if j < 0:
            return None, "CAPTION 块花括号不配平,拒绝改写"
        block = text[m.end():j]
        if '"template": "tags"' not in block:
            return None, ("当前配置的 caption 模板不是 tags,不支持触发词注入"
                          "(nl/json 等格式请把触发词直接写进数据集 caption)")
        if re.search(r'"prefix":\s*"[^"]*"', block):
            new_block = re.sub(r'"prefix":\s*"[^"]*"', f'"prefix": "{prefix}"',
                               block, count=1)
        else:
            new_block = f'\n    "prefix": "{prefix}",' + block
        return text[:m.end()] + new_block + text[j:], None
    if not isinstance(layer_caption, dict):
        return None, "无法注入触发词:未取到数据格式层的 CAPTION"
    if layer_caption.get("template") != "tags":
        return None, ("该数据格式的 caption 模板不是 tags,不支持触发词注入"
                      "(nl/json 等格式请把触发词直接写进数据集 caption)")
    merged = {**layer_caption, "prefix": prefix}
    rendered = _caption_literal(merged)
    if rendered is None:
        return None, ("数据格式层 CAPTION 含无法安全写回的值类型,不支持触发词注入"
                      "(可直接在生成后的配置文件里手动加 prefix)")
    return (text.rstrip("\n") + "\n\n# 触发词(UI 注入):tags 模板 prefix,自动加到每条训练 caption 开头\n"
            "CAPTION = " + rendered + "\n"), None


def _caption_literal(d: dict) -> str | None:
    """CAPTION dict -> 与引擎生成风格一致的双引号 Python 字面量文本;
    仅支持 str/bool/int/float/str-list(数据格式层实测范围),超范围返回 None。"""
    def val(v) -> str | None:
        if isinstance(v, bool):
            return "True" if v else "False"
        if isinstance(v, (int, float)):
            return repr(v)
        if isinstance(v, str):
            return json.dumps(v)
        if isinstance(v, (list, tuple)):
            parts = []
            for x in v:
                if not isinstance(x, str):
                    return None
                parts.append(json.dumps(x))
            return ("[" if isinstance(v, list) else "(") + ", ".join(parts) \
                + ("]" if isinstance(v, list) else ")" if len(parts) != 1 else ",)")
        return None

    lines = ["{"]
    for k, v in d.items():
        sv = val(v)
        if sv is None:
            return None
        lines.append(f"    {json.dumps(str(k))}: {sv},")
    return "\n".join(lines) + "\n}"


# ---------- 采样分辨率:向生成文件写 SAMPLE_SIZE 平行列表 ----------

_SAMPLE_SIZE_RE = re.compile(r"^SAMPLE_SIZE\s*=.*$", re.M)


def set_sample_size_in_config(text: str, sizes: list) -> tuple[str, str | None]:
    """把 SAMPLE_SIZE 改写为与 SAMPLE_PROMPTS 平行的 [(高, 宽), ...] 元组列表。

    引擎对无注解旋钮的 --set 强转按默认值 (1024,1024) 推断为 tuple[int, ...],
    平行列表在 --set 通道必然失败(实测 TypeError),只能走配置文件文本直写
    (配置文件字面量不经强转,训练期 sample_settings 以 tuple(SAMPLE_SIZE) 透传
    pipeline.generate 的 sizes 参数,支持 Size | Sequence[Size])。
    已存在 SAMPLE_SIZE 行则整行替换,否则插到 SAMPLE_PROMPTS 行之后(保持
    SAMPLING 区块聚合);两者都没有时追加到文件末尾。返回 (新文本, 错误)。
    """
    try:
        pairs = [(int(h), int(w)) for h, w in sizes]
        if any(h <= 0 or w <= 0 for h, w in pairs):
            raise ValueError("尺寸必须为正数")
    except (TypeError, ValueError):
        return text, "采样分辨率格式非法(应为 [高, 宽] 正数对)"
    literal = "SAMPLE_SIZE = [" + ", ".join(f"({h}, {w})" for h, w in pairs) + "]"
    if _SAMPLE_SIZE_RE.search(text):
        return _SAMPLE_SIZE_RE.sub(literal, text, count=1), None
    m = re.search(r"^SAMPLE_PROMPTS\s*=.*$", text, re.M)
    if m:
        line = m.group(0)
        opens = line.count("(") + line.count("[")
        closes = line.count(")") + line.count("]")
        if opens == closes:  # 单行完整赋值才在其后插入,防截断多行字面量
            return text[:m.end()] + "\n" + literal + text[m.end():], None
    sep = "" if text.endswith("\n") else "\n"
    return text + f"{sep}{literal}\n", None


# ---------- 服务类 ----------

class ConfigService:
    """config CLI 的 UI 侧封装;所有公开方法返回结构化 dict,绝不抛异常。"""

    def __init__(self, adapter, datasets_dir: str | Path | None = None) -> None:
        self.adapter = adapter
        # 数据集库根(默认 ui/state/datasets;测试可注入 tmp)
        self.datasets_dir = Path(datasets_dir) if datasets_dir else DATASETS_ROOT
        self._defaults_service = None

    # ---------- 内部工具 ----------

    def _defaults(self) -> DefaultsService:
        """惰性 DefaultsService(纯文本解析;FakeAdapter 环境解析失败自动降级)。"""
        if self._defaults_service is None:
            self._defaults_service = DefaultsService(self.adapter)
        return self._defaults_service

    def _exe(self):
        return self.adapter.find_executable("kohakumoss-config")

    @staticmethod
    def _engine_missing() -> dict:
        return {"ok": False, "engine_missing": True, "error":
                "kohakumoss-config 不存在(引擎未安装)", "stdout": "", "stderr": ""}

    @staticmethod
    def _result(proc: subprocess.CompletedProcess) -> dict:
        return {"ok": proc.returncode == 0, "exit_code": proc.returncode,
                "stdout": proc.stdout or "", "stderr": proc.stderr or ""}

    def _run(self, argv: list[str], timeout: float) -> dict:
        """执行短命令;超时/OSError 也归一为结构化结果。"""
        try:
            proc = self.adapter._run(argv, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"ok": False, "exit_code": None, "stdout": "",
                    "stderr": f"引擎命令超时({timeout}s):{argv[1:3]}"}
        except OSError as err:
            return {"ok": False, "exit_code": None, "stdout": "",
                    "stderr": f"无法执行引擎命令:{err}"}
        return self._result(proc)

    def user_dir(self) -> Path:
        return self.adapter.repo_root / USER_DIR

    def _safe_user_path(self, name: str) -> tuple[Path | None, str | None]:
        """name -> configs/user/<name>.py 的安全解析(严格白名单,防路径穿越)。

        拒绝 *_data 结尾的名字:new train 每次产出 <name>.py + <name>_data.py
        一对文件,后者是引擎数据阶段(inspect/cache)的内部配套配置(编码器
        指向离线缓存),不是用户可操作的实体;训练/读取/删除/校验一律只认
        主配置名,_data 路径由服务端从主名派生。
        """
        name = (name or "").strip()
        if name.endswith(".py"):
            stem = name[:-3]
        else:
            stem = name
        if not _NAME_RE.fullmatch(stem):
            return None, "配置名不合法(仅允许字母/数字/._-,禁止路径分隔符与 ..)"
        if stem.endswith("_data"):
            return None, ("_data 结尾的是引擎自动生成的配套数据配置,"
                          "请使用主配置名(去掉 _data 后缀)")
        target = (self.user_dir() / f"{stem}.py").resolve()
        base = self.user_dir().resolve()
        if target.parent != base:
            return None, "配置名不合法"
        return target, None

    # ---------- 公开接口 ----------

    def list_presets(self, everything: bool = False, fresh: bool = False) -> dict:
        """跑 `config list [ --all]` 并解析;解析失败降级返回原文(parsed=False)。

        结果按 everything 分别缓存(PRESETS_TTL 秒):config list 要冷启动
        torch,页面每次刷新都重跑代价极高;引擎升级后用 fresh=True 绕过。
        """
        now = time.monotonic()
        cache = getattr(self, "_presets_cache", None)
        if cache is None:
            cache = self._presets_cache = {}
        hit = cache.get("all" if everything else "std")
        if hit and not fresh and now - hit[0] < PRESETS_TTL:
            return hit[1]
        result = self._list_presets_uncached(everything)
        cache["all" if everything else "std"] = (now, result)
        return result

    def _list_presets_uncached(self, everything: bool) -> dict:
        exe = self._exe()
        if exe is None:
            return self._engine_missing()
        argv = [exe, "list"] + (["--all"] if everything else [])
        res = self._run(argv, LIST_TIMEOUT)
        if not res["ok"]:
            res["parsed"] = False
            res["error"] = res["stderr"] or f"config list 退出码 {res['exit_code']}"
            return res
        parsed = parse_config_list(res["stdout"])
        parsed["ok"] = True
        parsed["error"] = None
        parsed["stdout"] = res["stdout"]
        parsed["stderr"] = res["stderr"]
        parsed["exit_code"] = res["exit_code"]
        parsed["vram_options"] = vram_options(parsed["presets"])
        return parsed

    def new_train(self, form: dict) -> dict:
        """表单 -> `new train` -> 成功后读回 configs/user/<name>.py 原文。

        同名已存在且未 force:返回 exists=True(由路由层转 409)。
        form["overrides"](--set 键值)引擎在生成期不支持,原样回显为
        deferred_overrides,供训练启动(M2)经 kogine run --set 应用。
        显式传 --out:引擎的默认输出目录在 bundle 安装(wheel)时会退化为 cwd,
        固定指向 configs/user 保证落点确定(目录不存在时由引擎自行创建)。

        M1.5:form["datasets"]=[库名...] 从数据集库选择(替代手填 data 路径,
        旧 data 字段仍兼容)。第一个库目录走 --data;其余在生成后以最小文本插入
        追加进 DATASETS 列表(M1.5 实测:列表在主配置文件而非 _data.py),
        随后必须跑 config check,不通过则返回错误并附引擎输出。
        库目录是"根下平铺图片 + 同名 txt"布局,kohya source 只认 <repeats>_<name>/
        子目录(引擎实测:平铺布局训练期直接 ValueError),因此选了 datasets
        就强制 --source folder,忽略表单里的 source 值。
        """
        exe = self._exe()
        if exe is None:
            return self._engine_missing()
        # 高级设置合法键校验(引擎未知键直接拒绝;元数据不可用时自动放行)
        overrides = {str(k): str(v) for k, v in (form.get("overrides") or {}).items()}
        bad_keys, ov_warnings = self._defaults().check_override_keys(overrides)
        if bad_keys:
            return {"ok": False, "stdout": "", "stderr": "",
                    "error": "高级设置包含引擎未知键:" + ", ".join(bad_keys)}
        # datasets(库名) -> 库内目录解析;兼容:只传 data 路径时走旧逻辑
        names = [str(n).strip() for n in (form.get("datasets") or []) if str(n).strip()]
        resolved: list[Path] = []
        if names:
            ds = DatasetService(self.datasets_dir)
            for n in names:
                target, derr = ds.dataset_dir(n, must_exist=True)
                if derr:
                    return {"ok": False, "error": derr, "stdout": "", "stderr": ""}
                resolved.append(target)
            # 库目录永远是平铺布局(根下图片+同名 txt),kohya source 只认
            # <repeats>_<name>/ 子目录,训练期 catalog scan 直接 ValueError
            # (M2 实测),因此选了库数据集就强制 folder,覆盖下拉值。
            form = {**form, "data": [str(p) for p in resolved],
                    "source": "folder"}
        argv, err = build_train_argv(exe, form)
        if err:
            return {"ok": False, "error": err, "stdout": "", "stderr": ""}
        # 触发词:自动加到每条显式采样 prompt 前(采样也要用触发词激活 LoRA)
        trigger = str(form.get("trigger") or "").strip()
        if trigger:
            plist = form.get("prompt")
            if isinstance(plist, str):
                plist = [plist]
            plist = [str(p) for p in (plist or []) if str(p).strip()]
            if plist:
                form = {**form, "prompt": [f"{trigger}, {p}" for p in plist]}
                argv, err = build_train_argv(exe, form)
                if err:
                    return {"ok": False, "error": err, "stdout": "", "stderr": ""}
        # 采样分辨率:form["prompt_sizes"] 为与 prompt 平行的 [高, 宽] 列表。
        # 任一非默认(1024x1024)时,生成后把 SAMPLE_SIZE 平行列表直接写进配置文件
        # (实测:kogine --set 对无注解旋钮按默认值 (1024,1024) 推断为 tuple[int,...],
        #  平行列表强转 TypeError、非 strict 时透传字符串致训练期崩溃,--set 通道
        #  传不了 per-prompt 尺寸;配置文件文本不经强转,与 DATASETS/触发词注入同机制)。
        # 全默认不注入;高级面板手动设过 SAMPLE_SIZE 时以手动值(--set)为准并提示。
        sample_sizes: list | None = None
        raw_sizes = form.get("prompt_sizes") or []
        plist_final = [str(p) for p in (form.get("prompt") or []) if str(p).strip()]
        if plist_final and raw_sizes:
            if len(raw_sizes) != len(plist_final):
                return {"ok": False, "error": "采样分辨率数量与采样 prompt 条数不一致",
                        "stdout": "", "stderr": ""}
            try:
                sizes = [[int(s[0]), int(s[1])] for s in raw_sizes]
                if any(h <= 0 or w <= 0 for h, w in sizes):
                    raise ValueError("尺寸必须为正数")
            except (TypeError, ValueError, IndexError):
                return {"ok": False, "error": "采样分辨率格式非法(应为 [高, 宽] 正数对)",
                        "stdout": "", "stderr": ""}
            if any(s != [1024, 1024] for s in sizes):
                if "SAMPLE_SIZE" in overrides:
                    ov_warnings.append(
                        "高级设置已手动指定 SAMPLE_SIZE,采样 prompt 行的分辨率被忽略(以高级设置为准)")
                else:
                    sample_sizes = sizes
        name = form["name"].strip()
        target = self.user_dir() / f"{name}.py"
        if target.exists() and not form.get("force"):
            return {"ok": False, "exists": True, "path": str(target),
                    "error": f"已存在同名 LoRA 配置「{name}」",
                    "stdout": "", "stderr": ""}
        argv += ["--out", str(target)]
        res = self._run(argv, GEN_TIMEOUT)
        res["name"] = name
        if overrides:
            res["deferred_overrides"] = dict(overrides)
            # 仅面向用户确认结果,不暴露引擎命令与存储路径等实现细节
            res["note"] = "高级设置已保存,启动训练时将自动生效"
        if ov_warnings:
            res["override_warnings"] = ov_warnings
        if names:
            res["datasets_used"] = names
        if res["ok"] and (len(resolved) > 1 or trigger or sample_sizes):
            # 生成后文本注入:多数据集追加 DATASETS / 触发词 CAPTION prefix /
            # 采样分辨率 SAMPLE_SIZE;注入过就必须跑 config check(需求 §5 页面 B)
            try:
                text = target.read_text(encoding="utf-8")
            except OSError as read_err:
                res["ok"] = False
                res["error"] = f"配置已生成但读取失败:{read_err}"
            new_text = text
            if res["ok"] and len(resolved) > 1:
                new_text, aerr = append_datasets_to_config(
                    text, [str(p) for p in resolved[1:]])
                if aerr:
                    res["ok"] = False
                    res["error"] = aerr
            if res["ok"] and trigger:
                cap = None
                if not _CAPTION_RE.search(new_text):
                    src = str(form.get("source") or "kohya").strip()
                    cap, cerr = _layer_caption_dict(self.adapter.repo_root, src)
                    if cerr:
                        res["ok"] = False
                        res["error"] = cerr
                if res["ok"]:
                    new_text, terr = apply_trigger_to_config(new_text, trigger, cap)
                    if terr:
                        res["ok"] = False
                        res["error"] = terr
            if res["ok"] and sample_sizes:
                new_text, serr = set_sample_size_in_config(new_text, sample_sizes)
                if serr:
                    res["ok"] = False
                    res["error"] = serr
            if res["ok"]:
                try:
                    target.write_text(new_text, encoding="utf-8")
                except OSError as werr:
                    res["ok"] = False
                    res["error"] = f"配置改写写入失败:{werr}"
            if res["ok"]:
                if len(resolved) > 1:
                    res["datasets_appended"] = [p.name for p in resolved[1:]]
                if trigger:
                    res["trigger_applied"] = trigger
                if sample_sizes:
                    res["sample_size_injected"] = sample_sizes
                chk = self.check_config(name)
                res["check"] = {"ok": chk["ok"], "exit_code": chk["exit_code"],
                                "stdout": chk["stdout"], "stderr": chk["stderr"]}
                if not chk["ok"]:
                    res["ok"] = False
                    res["error"] = ("配置注入后 config check 未通过"
                                    f"(exit={chk['exit_code']}),详见 check 输出")
        if res["ok"]:
            # 持久化高级设置(空覆盖同步删除旧文件,防重生成后残留)
            save_overrides(name, overrides)
            content = None
            data_content = None
            try:
                if target.is_file():
                    content = target.read_text(encoding="utf-8")
                d = self.user_dir() / f"{name}_data.py"
                if d.is_file():
                    data_content = d.read_text(encoding="utf-8")
            except OSError as read_err:
                res["ok"] = False
                res["error"] = f"配置已生成但读回失败:{read_err}"
            res["path"] = f"{USER_DIR}/{name}.py"
            res["content"] = content
            res["data_content"] = data_content
        else:
            # 追加/check 失败已带具体 error,不被引擎输出覆盖
            res["error"] = res.get("error") or res["stderr"] or res["stdout"] or f"退出码 {res['exit_code']}"
        return res

    def new_sample(self, form: dict) -> dict:
        """表单 -> `new sample`;成功后读回生成的配置文件原文。"""
        exe = self._exe()
        if exe is None:
            return self._engine_missing()
        argv, err = build_sample_argv(exe, form)
        if err:
            return {"ok": False, "error": err, "stdout": "", "stderr": ""}
        name = (form.get("name") or "sample").strip()
        target = self.user_dir() / f"{name}.py"
        if target.exists() and not form.get("force"):
            return {"ok": False, "exists": True, "path": str(target),
                    "error": f"已存在同名 LoRA 配置「{name}」",
                    "stdout": "", "stderr": ""}
        argv += ["--out", str(target)]
        res = self._run(argv, GEN_TIMEOUT)
        res["name"] = name
        if res["ok"]:
            try:
                res["path"] = f"{USER_DIR}/{name}.py"
                res["content"] = target.read_text(encoding="utf-8") if target.is_file() else None
            except OSError as read_err:
                res["ok"] = False
                res["error"] = f"配置已生成但读回失败:{read_err}"
        else:
            res["error"] = res["stderr"] or res["stdout"] or f"退出码 {res['exit_code']}"
        return res

    def from_preset(self, base: str, out_name: str, force: bool = False) -> dict:
        """`new --from <recipe> configs/user/<out>.py`;成功后读回原文。"""
        exe = self._exe()
        if exe is None:
            return self._engine_missing()
        out_path, err = self._safe_user_path(out_name)
        if err:
            return {"ok": False, "error": err, "stdout": "", "stderr": ""}
        argv, err = build_from_argv(exe, base, str(out_path), force)
        if err:
            return {"ok": False, "error": err, "stdout": "", "stderr": ""}
        if out_path.exists() and not force:
            return {"ok": False, "exists": True, "path": str(out_path),
                    "error": "已存在同名配置", "stdout": "", "stderr": ""}
        res = self._run(argv, GEN_TIMEOUT)
        if res["ok"]:
            res["path"] = str(out_path.relative_to(self.adapter.repo_root)).replace("\\", "/")
            try:
                res["content"] = out_path.read_text(encoding="utf-8") if out_path.is_file() else None
            except OSError as read_err:
                res["ok"] = False
                res["error"] = f"配置已生成但读回失败:{read_err}"
        else:
            res["error"] = res["stderr"] or res["stdout"] or f"退出码 {res['exit_code']}"
        return res

    def check_config(self, name_or_path: str) -> dict:
        """`config check <cfg>`:返回 exit_code + 输出;ok = 退出码 0。

        裸名字解析为 configs/user/<name>.py;含分隔符的路径仅允许仓库内相对路径。
        """
        exe = self._exe()
        if exe is None:
            return self._engine_missing()
        raw = (name_or_path or "").strip()
        if not raw:
            return {"ok": False, "error": "check 目标为空", "stdout": "", "stderr": ""}
        if "\\" in raw or "/" in raw or Path(raw).is_absolute():
            candidate = (self.adapter.repo_root / raw).resolve()
            if not candidate.is_relative_to(self.adapter.repo_root.resolve()):
                return {"ok": False, "error": "check 路径越出仓库根,已拒绝",
                        "stdout": "", "stderr": ""}
            path = candidate
        else:
            path, err = self._safe_user_path(raw)
            if err:
                return {"ok": False, "error": err, "stdout": "", "stderr": ""}
        if not path.is_file():
            return {"ok": False, "error": f"配置文件不存在:{path}",
                    "stdout": "", "stderr": ""}
        res = self._run([exe, "check", str(path)], CHECK_TIMEOUT)
        res["path"] = str(path)
        # 引擎对"有问题"的配置退出码 1:这是数据而非故障,保留 ok=False 但无 error
        if not res["ok"] and res["exit_code"] == 1:
            res["error"] = None
        elif not res["ok"]:
            res["error"] = res["stderr"] or f"退出码 {res['exit_code']}"
        else:
            res["error"] = None
        return res

    def read_user_config(self, name: str) -> dict:
        """读回 configs/user/<name>.py 原文(严格白名单)。"""
        path, err = self._safe_user_path(name)
        if err:
            return {"ok": False, "error": err}
        if not path.is_file():
            return {"ok": False, "error": f"配置文件不存在:{USER_DIR}/{path.name}",
                    "not_found": True}
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as e:
            return {"ok": False, "error": f"读取失败:{e}"}
        return {"ok": True, "name": path.stem, "path": f"{USER_DIR}/{path.name}",
                "content": text}

    def list_user_configs(self) -> dict:
        """列出 configs/user/ 下已有配置名(不含扩展名)。

        过滤 *_data.py:new train 一个配置产出 <name>.py + <name>_data.py
        一对文件,_data 是数据阶段的内部配套,UI 层一个配置名 = 一对文件,
        只列主配置名,避免下拉把同一配置显示成两项或被选中跑错阶段。
        """
        d = self.user_dir()
        try:
            names = sorted(p.stem for p in d.glob("*.py")
                           if not p.stem.endswith("_data")) if d.is_dir() else []
        except OSError as e:
            return {"ok": False, "error": f"列目录失败:{e}", "configs": []}
        return {"ok": True, "configs": names, "error": None}

    def delete_user_config(self, name: str) -> dict:
        """删除 configs/user/<name>.py 及配套 <name>_data.py(引擎生成的数据配置),
        同时清理该配置名持久化的高级设置(ui/state/overrides/<name>.json)。"""
        path, err = self._safe_user_path(name)
        if err:
            return {"ok": False, "error": err}
        if not path.is_file():
            return {"ok": False, "error": f"配置文件不存在:{USER_DIR}/{path.name}",
                    "not_found": True}
        targets = [path]
        companion = path.parent / f"{path.stem}_data.py"
        if companion.is_file():
            targets.append(companion)
        deleted: list[str] = []
        try:
            for p in targets:
                p.unlink(missing_ok=True)
                deleted.append(f"{USER_DIR}/{p.name}")
        except OSError as e:
            return {"ok": False, "error": f"删除失败:{e}", "deleted": deleted}
        ov = override_path(path.stem)
        if ov is not None:
            try:
                ov.unlink(missing_ok=True)
            except OSError:
                pass  # 覆盖文件残留不影响删除结果
        return {"ok": True, "deleted": deleted, "error": None}

    # 项目内集成的权重目录候选(按优先级),相对仓库根
    WEIGHTS_CANDIDATES = ("weights/models", "weights", "models")

    def detect_weights(self) -> dict:
        """检测项目是否已集成模型权重(不跑引擎,纯目录结构判断)。
        ComfyUI 布局 = diffusion_models/+text_encoders/+vae/ 三子目录;
        HF 布局 = transformer/+vae/。命中返回绝对路径供前端预填。"""
        root = self.adapter.repo_root
        for rel in self.WEIGHTS_CANDIDATES:
            cand = root / rel
            if not cand.is_dir():
                continue
            has_comfy = all((cand / sub).is_dir() for sub in
                            ("diffusion_models", "text_encoders", "vae"))
            has_hf = (cand / "transformer").is_dir() and (cand / "vae").is_dir()
            if has_comfy or has_hf:
                return {"ok": True, "found": True,
                        "path": str(cand.resolve()),
                        "relative": rel,
                        "layout": "comfy" if has_comfy else "hf",
                        "error": None}
        return {"ok": True, "found": False, "path": None, "layout": None,
                "error": None}
