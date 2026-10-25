// 页面 C:训练配置向导(M1)——填表 -> new train 生成 -> check -> 预览引擎原文
// 算法/数据源/显存档位全部运行时来自 /api/config/presets,UI 不硬编码清单。
// M1.5:数据集字段改为从数据集库(/datasets)多选,不再手填目录路径。
// M2.5:基础区按引擎默认值补提示;"高级设置"分组面板默认值来自 /api/config/defaults
// (运行引擎资产文本解析),仅改动项 diff 成 overrides,随配置名持久化供训练启动应用。
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import type {
  CheckResult, DatasetSummary, DefaultsResult, InspectResult, KnobInfo, PresetsResult, TrainResult,
} from "../api";
import {
  fetchDatasets,
  fetchDefaults,
  fetchPresets,
  fetchUserConfigFile,
  fetchUserConfigs,
  fetchWeightsDetect,
  postCache,
  postCheck,
  postTrain,
  runInspect,
} from "../api";

interface OverrideRow {
  key: string;
  value: string;
}

// 高级设置面板:dict/list 用 JSON 文本域(引擎 literal_eval 可解析 JSON)
const JSON_TYPES = new Set(["dict", "list", "tuple"]);

// 主流训练器(kohya_ss/OneTrainer 等)高频调节参数,直接暴露在基础区;
// 默认值仍取运行引擎解析结果,仅改动项进 overrides(与高级面板同一 diff 机制)
const COMMON_KNOBS: { name: string; label: string; tip: string }[] = [
  { name: "LR", label: "学习率", tip: "LoRA 常用 5e-5 ~ 1e-4;训练不稳/抖动时调低(引擎旋钮 LR)" },
  { name: "DIM", label: "LoRA 秩 rank", tip: "容量档位:8 轻量冒烟 / 16-32 常用 / 64+ 高容量(引擎旋钮 DIM)" },
  { name: "ALPHA", label: "LoRA alpha", tip: "常取与 rank 相等或一半,控制适配强度(引擎旋钮 ALPHA)" },
  { name: "OPTIMIZER", label: "优化器", tip: "下拉选项来自引擎注册表;adamw 为默认。8bit 需 bitsandbytes、prodigy 需 prodigyopt、schedule_free 需 schedulefree、muon 需 fused_step=False,未装对应包启动会报错(引擎旋钮 OPTIMIZER)" },
  { name: "SCHEDULE", label: "学习率调度", tip: 'JSON dict,如 {"mode":"cosine","warmup":100,"end":-1};warmup 步数必须小于训练步数(引擎旋钮 SCHEDULE)' },
  { name: "BATCH_SIZE", label: "批大小", tip: "每卡每批样本数,仅在 BATCHING=count 时生效;默认 tokens 模式按 token 预算自动组批(引擎旋钮 BATCH_SIZE)" },
  { name: "GRAD_ACC", label: "梯度累积", tip: "等效批大小 = 批大小 × 累积步数,显存不足时优先加大它(引擎旋钮 GRAD_ACC)" },
  { name: "TARGET_AREA", label: "目标分辨率面积", tip: "像素总数,1048576≈1024×1024、262144≈512×512,宽高比自适应(引擎旋钮 TARGET_AREA)" },
  { name: "DROPOUT", label: "Dropout", tip: 'JSON dict,如 {"input":0,"rank":0,"module":0};防过拟合(引擎旋钮 DROPOUT)' },
  { name: "SAVE_EVERY", label: "保存间隔(步)", tip: "每隔多少步导出一个 checkpoint,0=只在结束时保存(引擎旋钮 SAVE_EVERY)" },
  { name: "SAMPLE_EVERY", label: "采样预览间隔(步)", tip: "训练中途出图预览的频率,0=关闭(引擎旋钮 SAMPLE_EVERY)" },
  { name: "SAMPLE_CFG", label: "采样 CFG", tip: "预览图引导强度,常用 2-6(引擎旋钮 SAMPLE_CFG)" },
];
const COMMON_NAMES = new Set(COMMON_KNOBS.map((c) => c.name));

function knobDefaultStr(k: KnobInfo): string {
  if (k.type === "bool") return k.default ? "true" : "false";
  if (k.default === null || k.default === undefined) return k.default_text ?? "";
  if (typeof k.default === "object") {
    try { return JSON.stringify(k.default); } catch { return k.default_text ?? ""; }
  }
  return String(k.default);
}

function jsonValid(s: string): boolean {
  try { JSON.parse(s); return true; } catch { return false; }
}

// 分辨率数字框:必须为正整数(宽/高各自独立填写,无分隔符解析)
function isPositiveInt(s: string): boolean {
  const n = Number(s.trim());
  return s.trim() !== "" && Number.isInteger(n) && n > 0;
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* 剪贴板不可用时静默降级 */
    }
  };
  return (
    <button className="copy-btn" onClick={copy} title="复制全文">
      {copied ? "已复制" : "复制"}
    </button>
  );
}

export default function ConfigPage() {

  const [presets, setPresets] = useState<PresetsResult | null>(null);
  const [presetsError, setPresetsError] = useState("");
  // presets 需启动引擎子进程(首次约 10s),拉取期间显示 loading 文案
  const [presetsLoading, setPresetsLoading] = useState(true);

  // 基础区(一级参数)
  const [weights, setWeights] = useState("");
  // 项目内集成权重检测结果(命中则预填并提示;用户手动改动后提示消失)
  const [weightsAuto, setWeightsAuto] = useState("");
  // M1.5:数据集从库中多选(替代原手填目录路径行)
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [selectedDs, setSelectedDs] = useState<string[]>([]);
  const [name, setName] = useState("");
  const [algo, setAlgo] = useState("");
  const [source, setSource] = useState("");
  const [steps, setSteps] = useState("4000");
  // 每条采样 prompt 附独立分辨率(宽/高两个数字框,默认 1024×1024,可改)
  const [prompts, setPrompts] = useState<{ text: string; w: string; h: string }[]>([]);
  const [trigger, setTrigger] = useState("");
  const [vram, setVram] = useState("");
  const [precision, setPrecision] = useState("");

  // 高级设置:分组面板改动项(旋钮名 -> 字符串值;只存"与默认不同"的项)
  const [panelEdits, setPanelEdits] = useState<Record<string, string>>({});
  // 专家组自由键值行(未进面板的 101 个旋钮的兜底覆盖通道)
  const [overrides, setOverrides] = useState<OverrideRow[]>([]);
  // 面板数据源:运行引擎资产解析(/api/config/defaults),随基础区组合参数刷新
  const [defaults, setDefaults] = useState<DefaultsResult | null>(null);
  const [defaultsError, setDefaultsError] = useState("");
  const defaultsSeq = useRef(0);

  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState<TrainResult | null>(null);
  const [checkResult, setCheckResult] = useState<CheckResult | null>(null);
  const [pageError, setPageError] = useState("");

  // 已有配置
  const [existing, setExisting] = useState<string[]>([]);
  const [preview, setPreview] = useState<{ name: string; content: string } | null>(null);
  // M2:数据检查(inspect 同步跑引擎,可能要数分钟)与缓存任务入口
  const [inspecting, setInspecting] = useState<string | null>(null);
  const [inspectRes, setInspectRes] = useState<{ cfg: string; body: InspectResult } | null>(null);
  const [cacheMsg, setCacheMsg] = useState<string | null>(null);

  const load = useCallback(() => {
    setPageError("");
    // 环境检测只在「环境与模型」页做;本页不等待任何探测。
    // 各接口独立回填、谁先回来谁先显示:数据集/配置列表是毫秒级,
    // 只有 presets 需要启动引擎子进程(首次约 10s),期间显示 loading,
    // 不再拖着其它字段一起等。引擎未就绪时各接口自行报错
    // (presetsError/defaultsError/生成失败提示),表单始终可填,
    // 失败在用户真正触发的动作上自然暴露。
    let rejected = 0;
    const markRejected = (e: unknown) => {
      if (++rejected === 3) setPageError(`无法连接 UI 后端:${String(e)}`);
    };
    fetchDatasets().then(setDatasets).catch(markRejected);
    fetchUserConfigs().then(setExisting).catch(markRejected);
    setPresetsLoading(true);
    fetchPresets()
      .then(setPresets)
      .catch((e) => { setPresetsError(String(e)); markRejected(e); })
      .finally(() => setPresetsLoading(false));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  // 项目内集成权重检测:命中则预填输入框(不覆盖用户已填内容)
  useEffect(() => {
    fetchWeightsDetect()
      .then((w) => {
        if (w.found && w.path) {
          setWeightsAuto(`已检测到项目集成的模型(${w.relative},${w.layout === "comfy" ? "ComfyUI" : "HF"} 布局),已自动填入`);
          setWeights((cur) => cur || w.path!);
        }
      })
      .catch(() => {
        /* 检测失败静默:保持 placeholder 手填模式 */
      });
  }, []);

  // 库数据集永远是平铺布局(根下图片+同名 txt),只有 folder 格式能扫到;
  // 后端生成时也会强制 folder,前端同步锁定下拉与高级设置默认值来源,保持一致。
  const effSource = selectedDs.length > 0 ? "folder" : source;

  // 高级设置面板数据:基础区组合参数(数据格式/算法/显存/精度)变化即重拉
  // (引擎不同层文件默认值不同,如 data_<source>.py 决定 CAPTION);序号守卫防旧响应覆盖。
  useEffect(() => {
    const seq = ++defaultsSeq.current;
    setDefaultsError("");
    fetchDefaults({
      source: effSource || "kohya",
      algo: algo || "lora",
      vram: vram || "auto",
      precision: precision || undefined,
    })
      .then((d) => {
        if (seq !== defaultsSeq.current) return;
        if (d.ok) setDefaults(d);
        else { setDefaults(null); setDefaultsError(d.error ?? "未知错误"); }
      })
      .catch((e) => { if (seq === defaultsSeq.current) setDefaultsError(String(e)); });
  }, [effSource, algo, vram, precision]);

  const algoOptions = presets?.presets?.moss?.algo ?? [];
  const sourceOptions = presets?.presets?.moss?.data ?? [];
  const precisionOptions = presets?.presets?.moss?.precision ?? [];
  const vramOptions = useMemo(() => presets?.vram_options ?? [], [presets]);

  // 表单始终可填可改;仅提交进行中禁用。引擎未就绪不拦页面,
  // 失败在「生成配置」等实际动作上由后端返回错误自然暴露。
  const disabled = submitting;

  // 面板分组(顺序跟后端 GROUP_LABELS;expert 单列),只读清单与生效覆盖统计
  const panelGroups = useMemo(() => {
    const knobs = defaults?.knobs;
    if (!defaults?.ok || !knobs || !defaults.panel) return [];
    const order = Object.keys(defaults.groups ?? {}).filter((g) => g !== "expert");
    return order
      .map((g) => ({
        key: g,
        label: defaults.groups![g],
        // 常用参数已提到基础区直接暴露,面板内去重
        items: defaults.panel!
          .filter((n) => knobs[n]?.group === g && !COMMON_NAMES.has(n))
          .map((n) => [n, knobs[n]] as [string, KnobInfo]),
      }))
      .filter((g) => g.items.length > 0);
  }, [defaults]);
  const expertKnobs = useMemo(() => {
    const knobs = defaults?.knobs;
    if (!defaults?.ok || !knobs) return [];
    return Object.entries(knobs).filter(([, k]) => k.group === "expert");
  }, [defaults]);
  // 改动项 -> 下发 overrides:JSON 非法项暂不生效(行内红字提示)
  const panelValid = useMemo(() => {
    const out: Record<string, string> = {};
    const knobs = defaults?.knobs;
    if (!knobs) return out;
    for (const [k, v] of Object.entries(panelEdits)) {
      const info = knobs[k];
      if (!info) continue;
      if (JSON_TYPES.has(info.type) && !jsonValid(v)) continue;
      out[k] = v;
    }
    return out;
  }, [defaults, panelEdits]);
  const effectiveOverrides = useMemo(() => ({
    ...Object.fromEntries(overrides.filter((r) => r.key.trim()).map((r) => [r.key.trim(), r.value])),
    ...panelValid,
  }), [overrides, panelValid]);
  const effectiveCount = Object.keys(effectiveOverrides).length;

  const setPanelEdit = (name: string, val: string) => {
    setPanelEdits((cur) => {
      const next = { ...cur };
      const info = defaults?.knobs?.[name];
      if (val.trim() === "" || (info && val === knobDefaultStr(info))) delete next[name];
      else next[name] = val;
      return next;
    });
  };

  const renderKnob = (name: string, k: KnobInfo, ui?: { label: string; tip: string }) => {
    const edited = name in panelEdits;
    const def = knobDefaultStr(k);
    const cur = edited ? panelEdits[name] : def;
    const invalid = edited && JSON_TYPES.has(k.type) && !jsonValid(panelEdits[name]);
    let control;
    if (k.choices && k.choices.length > 0) {
      const opts = k.choices.includes(cur) ? k.choices : [...k.choices, cur];
      control = (
        <select value={cur} disabled={disabled} onChange={(e) => setPanelEdit(name, e.target.value)}>
          {opts.map((c) => <option key={c} value={c}>{c}</option>)}
        </select>
      );
    } else if (k.type === "bool") {
      control = (
        <select value={cur === "true" ? "true" : "false"} disabled={disabled}
                onChange={(e) => setPanelEdit(name, e.target.value)}>
          <option value="true">true</option>
          <option value="false">false</option>
        </select>
      );
    } else if (JSON_TYPES.has(k.type)) {
      control = (
        <textarea rows={2} spellCheck={false} value={cur} disabled={disabled}
                  className={invalid ? "knob-invalid" : ""}
                  onChange={(e) => setPanelEdit(name, e.target.value)} />
      );
    } else {
      control = <input value={cur} disabled={disabled} onChange={(e) => setPanelEdit(name, e.target.value)} />;
    }
    return (
      <div className="knob-row" key={name}>
        <span className="knob-name" title={ui ? ui.tip : (k.note ?? k.layer_source ?? "")}>
          {ui ? ui.label : name}
        </span>
        <div className="knob-ctl">
          {control}
          {invalid && <span className="error-text">JSON 格式无效(该项暂不生效)</span>}
        </div>
        <span className="knob-default">默认 {k.default_text || def || "(空)"}</span>
        {edited ? (
          <button className="copy-btn" disabled={disabled} title="恢复默认"
                  onClick={() => setPanelEdits((cur2) => { const n = { ...cur2 }; delete n[name]; return n; })}>↺</button>
        ) : <span />}
      </div>
    );
  };

  const runCheck = useCallback(async (cfgName: string) => {
    setCheckResult(null);
    try {
      setCheckResult(await postCheck(cfgName));
    } catch (e) {
      setCheckResult({ ok: false, error: String(e), exit_code: null, stdout: "", stderr: "" });
    }
  }, []);

  const submit = useCallback(
    async (force: boolean) => {
      setPageError("");
      setCheckResult(null);
      const rows = prompts.filter((r) => r.text.trim());
      const plist = rows.map((r) => r.text.trim());
      if (rows.some((r) => !isPositiveInt(r.w) || !isPositiveInt(r.h))) {
        setPageError("采样分辨率的宽/高必须为正整数");
        return;
      }
      // 引擎 SAMPLE_SIZE 元素顺序为 [高, 宽]
      const psizes = rows.map((r) => [Number(r.h), Number(r.w)]);
      const form = {
        weights,
        datasets: selectedDs,
        name: name.trim(),
        algo: algo || undefined,
        source: effSource || undefined,
        steps: steps === "" ? undefined : Number(steps),
        prompt: plist.length ? plist : undefined,
        prompt_sizes: plist.length ? psizes : undefined,
        trigger: trigger.trim() || undefined,
        vram: vram || undefined,
        precision: precision || undefined,
        force,
        overrides: effectiveOverrides,
      };
      setSubmitting(true);
      try {
        const { status, body } = await postTrain(form);
        if (status === 409) {
          if (window.confirm(`已存在同名 LoRA 配置「${name.trim()}」。要用相同名字覆盖它吗?`)) {
            const retry = await postTrain({ ...form, force: true });
            setResult(retry.body);
            if (retry.body.ok) void runCheck(name.trim());
          } else {
            setResult(body);
          }
        } else {
          setResult(body);
          if (body.ok) {
            void runCheck(name.trim());
            try {
              setExisting(await fetchUserConfigs());
            } catch {
              /* 忽略 */
            }
          }
        }
      } catch (e) {
        setPageError(`生成请求失败:${String(e)}`);
      } finally {
        setSubmitting(false);
      }
    },
    [weights, selectedDs, name, algo, effSource, steps, prompts, trigger, vram, precision, effectiveOverrides, runCheck],
  );

  const doInspect = async (cfg: string) => {
    setInspecting(cfg);
    setInspectRes(null);
    const body = await runInspect(cfg);
    setInspecting(null);
    setInspectRes({ cfg, body });
  };

  const doCache = async (cfg: string, kind: "cache_text" | "cache_latents") => {
    const r = await postCache(cfg, kind);
    setCacheMsg(
      r.ok
        ? `已启动 ${kind} 任务(${r.task?.id.slice(0, 8) ?? "?"});GPU 独占,完成前无法启动训练,可在训练监控页看任务状态`
        : `${kind} 启动失败:${r.error}`,
    );
  };

  const previewExisting = async (cfg: string) => {
    setPageError("");
    try {
      const file = await fetchUserConfigFile(cfg);
      setPreview({ name: file.name, content: file.content });
      void runCheck(cfg);
    } catch (e) {
      setPageError(String(e));
    }
  };

  return (
    <div className="config-grid">
      {pageError && <div className="card error-text">{pageError}</div>}
      <div className="card">
        <h2>基础参数</h2>
        {presets && !presets.parsed && (
          <p className="warn-text">训练选项加载失败,下拉为空;引擎输出原文:
            <pre className="raw-pre">{presets.stdout}</pre>
          </p>
        )}
        {presetsError && <p className="error-text">训练选项加载失败:{presetsError}(可稍后重试)</p>}
        {presetsLoading && !presetsError && (
          <p className="loading-text">
            <span className="spinner" aria-hidden="true" />
            正在从引擎获取训练选项,首次启动约需 10 秒…
          </p>
        )}
        <label className="field">
          <span title="ComfyUI models 布局:diffusion_models/ + text_encoders/ + vae/,或 HF 目录">权重目录 ⓘ</span>
          <input value={weights} disabled={disabled} placeholder="D:\ComfyUI\models"
                 onChange={(e) => { setWeights(e.target.value); setWeightsAuto(""); }} />
          {weightsAuto && <span className="muted">{weightsAuto}</span>}
        </label>
        <div className="field">
          <span>数据集(从数据集库勾选,可多选)</span>
          {datasets.length === 0 ? (
            <p className="muted">
              库中还没有数据集。先去 <Link to="/datasets">数据集页</Link> 建库并上传图片,再回来勾选。
            </p>
          ) : (
            <div className="ds-check-list">
              {datasets.map((d) => (
                <label key={d.name} className="ds-check">
                  <input
                    type="checkbox"
                    disabled={disabled}
                    checked={selectedDs.includes(d.name)}
                    onChange={(e) =>
                      setSelectedDs((cur) =>
                        e.target.checked ? [...cur, d.name] : cur.filter((n) => n !== d.name),
                      )
                    }
                  />
                  <span>{d.name}</span>
                  {d.image_count === 0 ? (
                    <em className="ds-badge warn-badge">空库,不可训练</em>
                  ) : (
                    <>
                      <em className="ds-badge">{d.image_count} 图</em>
                      <em className={d.caption_count < d.image_count ? "ds-badge warn-badge" : "ds-badge"}>
                        caption {d.caption_count}/{d.image_count}
                      </em>
                    </>
                  )}
                </label>
              ))}
            </div>
          )}
        </div>
        <div className="field-grid">
          <label className="field">
            <span title="最终导出的模型文件即以此命名,如 my-lora-comfy.safetensors;仅限字母/数字/._-">LoRA 名称 ⓘ</span>
            <input value={name} disabled={disabled} placeholder="my-lora"
                   onChange={(e) => setName(e.target.value)} />
          </label>
          <label className="field">
            <span>训练算法</span>
            <select value={algo} disabled={disabled} onChange={(e) => setAlgo(e.target.value)}>
              <option value="">lora</option>
              {algoOptions.map((a) => <option key={a} value={a}>{a}</option>)}
            </select>
          </label>
          <label className="field">
            <span title={selectedDs.length > 0 ? "数据集库为平铺布局(图片+同名 txt),固定使用 folder 格式" : undefined}>数据格式{selectedDs.length > 0 ? "(库固定 folder)" : ""}</span>
            <select value={effSource} disabled={disabled || selectedDs.length > 0} onChange={(e) => setSource(e.target.value)}>
              <option value="">kohya</option>
              {sourceOptions.map((s) => <option key={s} value={s}>{s}</option>)}
              {effSource === "folder" && !sourceOptions.includes("folder") && (
                <option value="folder">folder</option>
              )}
            </select>
          </label>
          <label className="field">
            <span title="冒烟验证链路可先改 200 步(需大于 warmup 100),正式训练常用 2000-4000">训练步数 ⓘ</span>
            <input type="number" min={0} value={steps} disabled={disabled}
                   onChange={(e) => setSteps(e.target.value)} />
          </label>
          <label className="field">
            <span>显存档位</span>
            <select value={vram} disabled={disabled} onChange={(e) => setVram(e.target.value)}>
              <option value="">自动检测(本机 GPU)</option>
              {vramOptions.map((v) => <option key={String(v)} value={String(v)}>{v === "auto" ? "auto(内存规划器)" : `${v} GiB`}</option>)}
            </select>
          </label>
          <label className="field">
            <span title="留空 = 按 GPU 能力自动检测;bf16 为引擎默认档">计算精度 ⓘ</span>
            <select value={precision} disabled={disabled} onChange={(e) => setPrecision(e.target.value)}>
              <option value="">自动(按 GPU 能力)</option>
              {precisionOptions.map((p) => <option key={p} value={p}>{p}</option>)}
            </select>
          </label>
        </div>
        <div className="field-grid">
          <label className="field">
            <span title="角色/产品 LoRA 建议填,自动加到每条 caption 与采样 prompt 开头">触发词(可选) ⓘ</span>
            <input value={trigger} disabled={disabled} placeholder="如 skz,训练和出图都用它激活 LoRA"
                   onChange={(e) => setTrigger(e.target.value)} />
          </label>
        </div>
        <div className="field">
          <span title="训练中途出图预览,可添加多条;每条可设独立分辨率(宽 × 高),留空则取数据集首条 caption">采样 prompt ⓘ</span>
          {prompts.map((r, i) => (
            <div className="row-line" key={i}>
              <input value={r.text} disabled={disabled} placeholder="a lighthouse on a rocky coast at sunset"
                     onChange={(e) => setPrompts((rs) => rs.map((q, j) => (j === i ? { ...q, text: e.target.value } : q)))} />
              <span className="size-group">
                <input className="num-input" type="number" min={1} step={1} value={r.w} disabled={disabled} title="宽(像素)"
                       onChange={(e) => setPrompts((rs) => rs.map((q, j) => (j === i ? { ...q, w: e.target.value } : q)))} />
                <span className="size-x">×</span>
                <input className="num-input" type="number" min={1} step={1} value={r.h} disabled={disabled} title="高(像素)"
                       onChange={(e) => setPrompts((rs) => rs.map((q, j) => (j === i ? { ...q, h: e.target.value } : q)))} />
              </span>
              <button className="copy-btn" disabled={disabled}
                      onClick={() => setPrompts((rs) => rs.filter((_, j) => j !== i))}>删除</button>
            </div>
          ))}
          <button className="copy-btn" disabled={disabled}
                  onClick={() => setPrompts((rs) => [...rs, { text: "", w: "1024", h: "1024" }])}>＋ 添加采样 prompt</button>
        </div>

        <div className="common-knobs">
          <span className="common-title"
                title="灰字为运行引擎的推荐默认值,悬停标签看说明;只有改动过的项才会作为覆盖下发">
            常用参数 ⓘ
          </span>
          {defaults?.ok && defaults.knobs ? (
            COMMON_KNOBS.map(({ name, label, tip }) => {
              const k = defaults.knobs?.[name];
              return k ? renderKnob(name, k, { label, tip }) : null;
            })
          ) : (
            <p className="muted">
              {defaultsError ? `默认值加载失败:${defaultsError}(可在下方高级设置用自由键值覆盖)` : "默认值加载中……"}
            </p>
          )}
        </div>

        <details className="advanced">
          <summary>高级设置{effectiveCount > 0 ? `(生效覆盖 ${effectiveCount} 项)` : "(未改动)"}</summary>
          <p className="muted">
            此处改动会随配置名保存,启动训练时自动生效;只有与默认值不同的项才会应用。
            默认值为当前配置组合下的推荐值;「续训」无需在此设置,训练页勾选即可。
          </p>
          {defaultsError && <p className="warn-text">默认值加载失败:{defaultsError}(可改用下方自由键值)</p>}
          {!defaults && !defaultsError && <p className="muted">默认值加载中……</p>}
          {trigger.trim() && "CAPTION" in panelEdits && (
            <p className="warn-text">
              ⚠ 触发词与高级设置里的 caption 覆盖同时设置了:将以 caption 覆盖为准。
            </p>
          )}
          {panelGroups.map((g) => (
            <details className="knob-group" key={g.key}>
              <summary>{g.label}({g.items.length})</summary>
              {g.items.map(([n, k]) => renderKnob(n, k))}
            </details>
          ))}
          <details className="knob-group">
            <summary>专家(自由键值 + 只读清单)</summary>
            {overrides.map((row, i) => (
              <div className="row-line" key={i}>
                <input placeholder="旋钮名(键名须与引擎一致)" value={row.key} disabled={disabled}
                       onChange={(e) => setOverrides((rs) => rs.map((r, j) => (j === i ? { ...r, key: e.target.value } : r)))} />
                <input placeholder="值,如 16 / 2e-4 / JSON" value={row.value} disabled={disabled}
                       onChange={(e) => setOverrides((rs) => rs.map((r, j) => (j === i ? { ...r, value: e.target.value } : r)))} />
                <button className="copy-btn" disabled={disabled}
                        onClick={() => setOverrides((rs) => rs.filter((_, j) => j !== i))}>删除</button>
              </div>
            ))}
            <button className="copy-btn" disabled={disabled}
                    onClick={() => setOverrides((rs) => [...rs, { key: "", value: "" }])}>+ 添加自由键值项</button>
            {defaults?.ok && expertKnobs.length > 0 && (
              <>
                <p className="muted">
                  以下 {expertKnobs.length} 个旋钮未列入分组面板(只读;悬停看默认值,需要覆盖时在上方自由键值填写):
                </p>
                <div className="expert-list">
                  {expertKnobs.map(([n, k]) => (
                    <span key={n} className="ds-badge"
                          title={`${n}=${k.default_text || "(无默认)"}${k.note ? ` · ${k.note}` : ""}`}>
                      {n}
                    </span>
                  ))}
                </div>
              </>
            )}
          </details>
        </details>

        <button className="primary-btn" disabled={disabled || !weights || !name || selectedDs.length === 0}
                onClick={() => submit(false)}>
          {submitting ? "生成中(引擎需加载 torch,可能数分钟)……" : "生成配置"}
        </button>
      </div>

      <div className="card">
        <h2>生成结果</h2>
        {!result && <p className="muted">尚未生成。填好左侧表单后点击“生成配置”。</p>}
        {result && !result.ok && (
          <div className="error-text">
            <p>生成失败:{result.error}</p>
            {result.stdout && <pre className="raw-pre">{result.stdout}</pre>}
            {result.stderr && <pre className="raw-pre">{result.stderr}</pre>}
            {result.check && !result.check.ok && (
              <pre className="raw-pre">{result.check.stdout || result.check.stderr}</pre>
            )}
          </div>
        )}
        {result?.ok && (
          <>
            <p className="ok-text">已生成 <code>{result.path}</code></p>
            {result.datasets_appended && result.datasets_appended.length > 0 && (
              <p className="ok-text">
                多数据集:已把 {result.datasets_appended.map((n) => <code key={n}>{n}</code>)} 追加进
                DATASETS 列表,且 config check 通过。
              </p>
            )}
            {result.note && result.deferred_overrides && (
              <p className="ok-text">
                {result.note}:
                {Object.entries(result.deferred_overrides).map(([k, v]) => (
                  <code key={k}>{k}={v}</code>
                ))}
              </p>
            )}
            {result.override_warnings && result.override_warnings.length > 0 && (
              <p className="warn-text">
                {result.override_warnings.map((w) => <span key={w}>⚠ {w};</span>)}
              </p>
            )}
            <p>
              <Link className="primary-btn" to={`/train?config=${encodeURIComponent(result.name ?? name.trim())}`}>
                去启动训练 →
              </Link>
            </p>
            <h3>配置文件原文预览 <CopyButton text={result.content ?? ""} /></h3>
            <pre className="config-pre">{result.content ?? "(读回失败)"}</pre>
            {result.data_content && (
              <>
                <h3>数据配置原文预览 <CopyButton text={result.data_content} /></h3>
                <pre className="config-pre">{result.data_content}</pre>
              </>
            )}
            {checkResult && (
              <>
                <h3>check 校验</h3>
                <p className={checkResult.ok ? "ok-text" : "error-text"}>
                  {checkResult.ok ? "✓ 配置合法(exit=0)" : `✗ 配置不合法(exit=${checkResult.exit_code})`}
                  {checkResult.error ? `:${checkResult.error}` : ""}
                </p>
                <pre className="raw-pre">{checkResult.stdout || checkResult.stderr}</pre>
              </>
            )}
          </>
        )}
        {pageError && <p className="error-text">{pageError}</p>}
      </div>

      <div className="card">
        <h2>已有配置</h2>
        {existing.length === 0 && !preview && <p className="muted">暂无用户配置。</p>}
        <ul className="config-list">
          {existing.map((cfg) => (
            <li key={cfg}>
              <code>{cfg}</code>
              <button className="copy-btn" onClick={() => previewExisting(cfg)}>预览 + check</button>
              <button className="copy-btn" disabled={inspecting !== null} onClick={() => doInspect(cfg)}>
                {inspecting === cfg ? "检查中(引擎加载可能数分钟)……" : "数据检查"}
              </button>
              <button className="copy-btn" onClick={() => doCache(cfg, "cache_text")}>缓存文本</button>
              <button className="copy-btn" onClick={() => doCache(cfg, "cache_latents")}>缓存 Latents</button>
              <Link className="copy-btn" to={`/train?config=${encodeURIComponent(cfg)}`}>去启动训练</Link>
            </li>
          ))}
        </ul>
        {cacheMsg && <p className="warn-text">{cacheMsg}</p>}
        {inspectRes && (
          <>
            <h3>数据检查:{inspectRes.cfg}({inspectRes.body.config ?? "?"})</h3>
            {inspectRes.body.ok ? (
              <pre className="config-pre">{JSON.stringify(inspectRes.body.report, null, 2)}</pre>
            ) : (
              <div className="error-text">
                <p>检查失败:{inspectRes.body.error}</p>
                {(inspectRes.body.stdout || inspectRes.body.stderr) && (
                  <pre className="raw-pre">{`${inspectRes.body.stdout ?? ""}\n${inspectRes.body.stderr ?? ""}`}</pre>
                )}
              </div>
            )}
          </>
        )}
        {preview && (
          <>
            <h3>{preview.name} <CopyButton text={preview.content} /></h3>
            <pre className="config-pre">{preview.content}</pre>
            {checkResult && (
              <>
                <h3>check 校验</h3>
                <p className={checkResult.ok ? "ok-text" : "error-text"}>
                  {checkResult.ok ? "✓ 配置合法(exit=0)" : `✗ 配置不合法(exit=${checkResult.exit_code})`}
                  {checkResult.error ? `:${checkResult.error}` : ""}
                </p>
                <pre className="raw-pre">{checkResult.stdout || checkResult.stderr}</pre>
              </>
            )}
          </>
        )}
      </div>
    </div>
  );
}
