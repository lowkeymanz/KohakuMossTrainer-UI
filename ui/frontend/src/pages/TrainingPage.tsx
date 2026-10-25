// 页面 D:训练监控闭环(M2)。
// 数据流:面板只跟随所选配置——拉该配置最近任务快照(/api/train/status?config=),未选留空;
// WS 增量事件仅当运行任务与所选配置一致时进面板;
// run_reset 清曲线/日志/样图(引擎轮转或全新重跑),log_reset 只清日志(续训)。
// Loss 曲线为自绘 SVG(不引图表库);验证曲线无数据自动隐藏;样图墙 SAMPLE_EVERY=0 隐藏。
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import type { GpuCard, MetricRow, TrainStatus, TrainTaskRow } from "../api";
import {
  deleteConfig,
  fetchConfigOverrides,
  fetchTrainStatus,
  fetchUserConfigs,
  openEventSocket,
  postTrainStart,
  postTrainStop,
  runFileUrl,
} from "../api";

const LOG_CAP = 2000;

// ---------- Loss 曲线(自绘 SVG) ----------

const VAL_COLORS = ["#3ddc84", "#ffb454", "#c678dd", "#56b6c2", "#ff5c8a"];

interface Series {
  name: string;
  color: string;
  pts: [number, number][];
}

function buildSeries(rows: MetricRow[]): Series[] {
  const train: [number, number][] = [];
  const val = new Map<string, [number, number][]>();
  for (const r of rows) {
    if (typeof r.step !== "number") continue;
    if (typeof r.loss === "number") train.push([r.step, r.loss]);
    for (const [k, v] of Object.entries(r)) {
      if (k.startsWith("val_loss") && typeof v === "number") {
        if (!val.has(k)) val.set(k, []);
        val.get(k)!.push([r.step, v]);
      }
    }
  }
  const out: Series[] = [];
  if (train.length) out.push({ name: "train loss", color: "#4f8cff", pts: train });
  let i = 0;
  for (const [name, pts] of [...val.entries()].sort()) {
    out.push({ name, color: VAL_COLORS[i % VAL_COLORS.length], pts });
    i += 1;
  }
  return out;
}

function LossChart({ rows }: { rows: MetricRow[] }) {
  const series = useMemo(() => buildSeries(rows), [rows]);
  if (series.length === 0) {
    return <p className="muted">暂无 metrics 数据(等待引擎写出 metrics.jsonl)。</p>;
  }
  const W = 760;
  const H = 280;
  const ML = 56;
  const MR = 14;
  const MT = 12;
  const MB = 28;
  const xs = series.flatMap((s) => s.pts.map((p) => p[0]));
  const ys = series.flatMap((s) => s.pts.map((p) => p[1]));
  let x0 = Math.min(...xs);
  let x1 = Math.max(...xs);
  if (x0 === x1) { x0 -= 1; x1 += 1; } // 单点:居中铺开
  let y0 = Math.min(...ys);
  let y1 = Math.max(...ys);
  if (y0 === y1) { y0 -= 0.5; y1 += 0.5; }
  const pad = (y1 - y0) * 0.06;
  y0 -= pad;
  y1 += pad;
  const px = (v: number) => ML + ((v - x0) / (x1 - x0)) * (W - ML - MR);
  const py = (v: number) => MT + (1 - (v - y0) / (y1 - y0)) * (H - MT - MB);
  const fmtY = (v: number) => (Math.abs(v) >= 1000 ? v.toFixed(0) : v.toPrecision(3));
  const yTicks = [0, 1, 2, 3, 4].map((i) => y0 + ((y1 - y0) * i) / 4);
  const xTicks = [0, 1, 2, 3, 4].map((i) => Math.round(x0 + ((x1 - x0) * i) / 4));
  const dot = series.some((s) => s.pts.length <= 80);
  return (
    <div className="chart-wrap">
      <svg viewBox={`0 0 ${W} ${H}`} className="chart-svg" role="img" aria-label="loss 曲线">
        {yTicks.map((t, i) => (
          <g key={`y${i}`}>
            <line x1={ML} x2={W - MR} y1={py(t)} y2={py(t)} stroke="#2e3342" strokeWidth={1} />
            <text x={ML - 6} y={py(t) + 4} textAnchor="end" className="chart-tick">{fmtY(t)}</text>
          </g>
        ))}
        {xTicks.map((t, i) => (
          <text key={`x${i}`} x={px(t)} y={H - 8} textAnchor="middle" className="chart-tick">
            {t}
          </text>
        ))}
        {series.map((s) => (
          <g key={s.name}>
            <polyline
              fill="none"
              stroke={s.color}
              strokeWidth={1.8}
              points={s.pts.map(([x, y]) => `${px(x)},${py(y)}`).join(" ")}
            />
            {dot &&
              s.pts.map(([x, y], i) => (
                <circle key={i} cx={px(x)} cy={py(y)} r={2.6} fill={s.color} />
              ))}
          </g>
        ))}
      </svg>
      <div className="chart-legend">
        {series.map((s) => (
          <span key={s.name}>
            <i style={{ background: s.color }} /> {s.name}
          </span>
        ))}
      </div>
    </div>
  );
}

// ---------- 日志终端 ----------

function logClass(line: string): string {
  if (/error|traceback|exception/i.test(line)) return "log-err";
  if (/\bwarn\b|warning/i.test(line)) return "log-warn";
  return "";
}

function LogTerm({ lines }: { lines: string[] }) {
  const ref = useRef<HTMLDivElement>(null);
  const stick = useRef(true);
  useEffect(() => {
    const el = ref.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [lines]);
  return (
    <div
      className="term"
      ref={ref}
      onScroll={(e) => {
        const el = e.currentTarget;
        stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
      }}
    >
      {lines.length === 0 && <div className="muted">暂无日志。</div>}
      {lines.map((l, i) => (
        <div key={i} className={logClass(l)}>{l || "\u00a0"}</div>
      ))}
    </div>
  );
}

// ---------- GPU 状态条 ----------

function GpuPanel({ gpus }: { gpus: GpuCard[] | null }) {
  if (!gpus || gpus.length === 0) {
    return <p className="muted">GPU 监测不可用(无 NVIDIA GPU / 驱动 / pynvml)。</p>;
  }
  return (
    <div className="gpu-cards">
      {gpus.map((g) => {
        const memPct = g.mem_used_mb != null && g.mem_total_mb ? (g.mem_used_mb / g.mem_total_mb) * 100 : null;
        const gib = (mb: number | null) => (mb == null ? "-" : `${(mb / 1024).toFixed(1)} GiB`);
        return (
          <div className="gpu-card" key={g.index}>
            <div className="gpu-head">
              <b>{g.name ?? `GPU ${g.index}`}</b>
              <span className="muted">
                {g.temp_c != null ? `${g.temp_c}°C` : "-"} · {g.power_w != null ? `${g.power_w.toFixed(0)} W` : "-"}
              </span>
            </div>
            <div className="gpu-bar-row">
              <span>核心</span>
              <div className="gpu-bar"><i style={{ width: `${g.util_gpu ?? 0}%` }} /></div>
              <em>{g.util_gpu != null ? `${g.util_gpu}%` : "-"}</em>
            </div>
            <div className="gpu-bar-row">
              <span>显存</span>
              <div className="gpu-bar"><i className="mem" style={{ width: `${Math.min(memPct ?? 0, 100)}%` }} /></div>
              <em>{gib(g.mem_used_mb)}{g.mem_total_mb ? ` / ${gib(g.mem_total_mb)}` : ""}</em>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ---------- 样图墙 ----------

function sampleLabel(path: string): string {
  const m = path.match(/step_(\d+)/);
  const step = m ? String(Number(m[1])) : "";
  const file = path.split("/").pop() ?? path;
  return step ? `step ${step} · ${file}` : file;
}

function SampleWall({ samples, sampleEvery }: { samples: string[]; sampleEvery: number | null | undefined }) {
  if (sampleEvery === 0) {
    return <p className="muted">预览已关闭,训练中不会生成样图。可在配置页的高级设置里开启。</p>;
  }
  if (samples.length === 0) {
    return <p className="muted">暂无样图(等待引擎写出样图;首次预览生成可能需要几分钟)。</p>;
  }
  return (
    <div className="sample-wall">
      {samples.map((p) => (
        <figure key={p}>
          <a href={runFileUrl(p)} target="_blank" rel="noreferrer">
            <img src={runFileUrl(p)} loading="lazy" alt={sampleLabel(p)} />
          </a>
          <figcaption>{sampleLabel(p)}</figcaption>
        </figure>
      ))}
    </div>
  );
}

// ---------- 主页面 ----------

const PHASES: Record<string, { text: string; cls: string }> = {
  idle: { text: "空闲", cls: "ph-idle" },
  pending: { text: "排队准备中", cls: "ph-init" },
  initializing: { text: "初始化中", cls: "ph-init" },
  running: { text: "训练中", cls: "ph-run" },
  finished: { text: "已完成", cls: "ph-done" },
  failed: { text: "失败", cls: "ph-fail" },
  stopped: { text: "已停止", cls: "ph-stop" },
};

function fmtEta(s: number | null): string {
  if (s == null || !isFinite(s) || s < 0) return "-";
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = Math.round(s % 60);
  return h > 0 ? `${h}h ${m}m` : m > 0 ? `${m}m ${sec}s` : `${sec}s`;
}

// ---------- 准备阶段(initializing):从引擎日志提取可读进度 ----------

// 内核调优共 8 档,日志形如 "tuning attention_varlen (joint) 2/8 (128 tokens)"
const TUNING_RE = /tuning (\w+)[^\n]*?(\d+)\/(\d+)/;
// 粗粒度阶段(从后往前匹配,取最近出现的)
const STAGE_PATTERNS: [RegExp, string][] = [
  [/kernel prewarm/, "GPU 内核预热"],
  [/text encoder/, "加载文本编码器"],
  [/transformer loaded/, "构建模型"],
  [/loading transformer/, "加载基座模型"],
  [/building model/, "构建模型"],
  [/building the data loader/, "准备数据"],
];

function initStage(logs: string[]): string | null {
  for (let i = logs.length - 1; i >= 0; i--) {
    const m = logs[i].match(TUNING_RE);
    if (m) return `GPU 内核预热 · ${m[2]}/${m[3]} 档`;
  }
  for (let i = logs.length - 1; i >= 0; i--) {
    for (const [re, label] of STAGE_PATTERNS) {
      if (re.test(logs[i])) return label;
    }
  }
  return null;
}

function fmtElapsed(ms: number): string {
  const s = Math.max(0, Math.floor(ms / 1000));
  const m = Math.floor(s / 60);
  return m > 0 ? `${m} 分 ${s % 60} 秒` : `${s} 秒`;
}

// 所选配置的持久化键:切换页面/刷新后恢复,避免面板"看起来数据全丢"
const CONFIG_SEL_KEY = "train.configSel";

function configNameOfPath(cp: string | null | undefined): string {
  if (!cp) return "";
  return cp.split(/[\\/]/).pop()!.replace(/\.py$/, "");
}

export default function TrainingPage() {
  const [search] = useSearchParams();
  const [status, setStatus] = useState<TrainStatus | null>(null);
  const [metrics, setMetrics] = useState<Map<number, MetricRow>>(new Map());
  const [logs, setLogs] = useState<string[]>([]);
  const [samples, setSamples] = useState<string[]>([]);
  const [gpus, setGpus] = useState<GpuCard[] | null>(null);
  const [connected, setConnected] = useState(false);
  const [configs, setConfigs] = useState<string[]>([]);
  // 优先级:URL 参数(配置页跳转携带) > 上次选择(localStorage)
  const [configSel, setConfigSel] = useState(
    () => search.get("config") || localStorage.getItem(CONFIG_SEL_KEY) || "",
  );
  const [resumeChk, setResumeChk] = useState(false);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ kind: "ok" | "err"; text: string } | null>(null);
  const mounted = useRef(true);
  const configSelRef = useRef(configSel);
  configSelRef.current = configSel;
  // 面板当前展示任务所属的配置名(仅当其正在运行时接收 WS 增量事件)
  const streamNameRef = useRef("");

  useEffect(() => () => { mounted.current = false; }, []);

  // REST 快照:只拉所选配置的最近任务;未选配置 = 面板留空
  const loadSnapshot = useCallback(async () => {
    const name = configSelRef.current;
    try {
      const st = await fetchTrainStatus(name);
      if (!mounted.current) return;
      setStatus(st);
      streamNameRef.current = st.task && st.task.status === "running" ? name : "";
      if (st.task) {
        const m = new Map<number, MetricRow>();
        for (const r of st.metrics ?? []) {
          if (typeof r.step === "number") m.set(r.step, r);
        }
        setMetrics(m);
        setLogs(st.log_tail ?? []);
        setSamples(st.samples ?? []);
      } else {
        setMetrics(new Map());
        setLogs([]);
        setSamples([]);
      }
    } catch {
      /* 后端不可达时由 WS 重连兜底 */
    }
  }, []);

  // 切换所选配置:先清空面板,再拉该配置最近任务(未选 = 留空)
  useEffect(() => {
    setStatus(null);
    setMetrics(new Map());
    setLogs([]);
    setSamples([]);
    streamNameRef.current = "";
    void loadSnapshot();
  }, [configSel, loadSnapshot]);

  // 选择持久化:切页/刷新后自动恢复
  useEffect(() => {
    try {
      if (configSel) localStorage.setItem(CONFIG_SEL_KEY, configSel);
      else localStorage.removeItem(CONFIG_SEL_KEY);
    } catch {
      /* 隐私模式等存储不可用:仅失去持久化,不影响功能 */
    }
  }, [configSel]);

  // 冷启动兜底:无 URL 参数且无持久化选择时,自动跟回最近一个训练任务
  useEffect(() => {
    if (configSel) return;
    fetchTrainStatus("")
      .then((st) => {
        const nm = configNameOfPath(st.task?.config_path);
        if (nm && mounted.current) setConfigSel(nm);
      })
      .catch(() => undefined);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    fetchUserConfigs().then(setConfigs).catch(() => setConfigs([]));
    const close = openEventSocket((ev) => {
      if (!mounted.current) return;
      // WS 增量事件只归属正在运行的那个任务;与所选配置不一致时不进面板
      const streaming = streamNameRef.current !== "" &&
        streamNameRef.current === configSelRef.current;
      switch (ev.type) {
        case "hello":
          void loadSnapshot();
          break;
        case "metric": {
          if (!streaming) break;
          const d = ev.data;
          if (d && typeof d.step === "number") {
            const step = d.step;
            setMetrics((cur) => {
              const n = new Map(cur);
              n.set(step, d);
              return n;
            });
          }
          break;
        }
        case "logline":
          if (!streaming) break;
          setLogs((cur) => {
            const next = cur.length >= LOG_CAP ? cur.slice(cur.length - LOG_CAP + 1) : cur;
            return [...next, ev.line ?? ""];
          });
          break;
        case "sample":
          if (streaming && ev.path) setSamples((cur) => (cur.includes(ev.path!) ? cur : [...cur, ev.path!].sort()));
          break;
        case "gpu":
          setGpus(ev.gpus ?? null);
          break;
        case "run_reset":
          // 轮转/全新重跑:后端会整文件重推,这里只清屏,不拉快照防混叠
          if (!streaming) break;
          setMetrics(new Map());
          setLogs([]);
          setSamples([]);
          setGpus(null);
          break;
        case "log_reset":
          if (streaming) setLogs([]);
          break;
        case "task_status": {
          const t = ev.task;
          if (t && t.kind === "train") {
            const nm = t.config_path
              ? t.config_path.split(/[\\/]/).pop()!.replace(/\.py$/, "")
              : "";
            if (nm !== configSelRef.current) break; // 面板只跟随所选配置的任务
            if (t.status === "running") {
              streamNameRef.current = nm;
              setStatus((cur) => (cur ? { ...cur, task: t } : cur));
            } else {
              streamNameRef.current = "";
              void loadSnapshot(); // 终态:拉一次权威快照
            }
          }
          break;
        }
        default:
          break;
      }
    }, setConnected);
    return close;
  }, [loadSnapshot]);

  const task: TrainTaskRow | null = status?.task ?? null;
  const rows = useMemo(
    () => [...metrics.values()].sort((a, b) => (a.step ?? 0) - (b.step ?? 0)),
    [metrics],
  );
  // 相位:进程活但无 metrics = initializing(内核预热可能数分钟,不误判卡死)
  const phase = !task
    ? "idle"
    : task.status === "running"
      ? rows.length
        ? "running"
        : "initializing"
      : (({ exited_ok: "finished", exited_fail: "failed", killed: "stopped" } as Record<string, string>)[task.status] ?? task.status);
  const running = task?.status === "running";
  // 排队中:训练在数据准备(缓存)任务后等待自动启动
  const queued = task?.status === "pending";
  // 准备阶段横幅:每 5s 刷新已进行时间,阶段从引擎日志实时提取
  const [nowTick, setNowTick] = useState(0);
  useEffect(() => {
    if (phase !== "initializing") return;
    const t = setInterval(() => setNowTick((x) => x + 1), 5000);
    return () => clearInterval(t);
  }, [phase]);
  const initInfo = useMemo(() => {
    void nowTick; // 仅用于触发周期重算
    if (phase !== "initializing") return null;
    const ms = task?.started_at ? Date.now() - Date.parse(task.started_at) : NaN;
    return {
      elapsed: Number.isFinite(ms) ? fmtElapsed(ms) : null,
      stage: initStage(logs),
    };
  }, [phase, task?.started_at, logs, nowTick]);
  const total =
    status?.total_steps ?? (task?.overrides?.STEPS ? Number(task.overrides.STEPS) : null);
  const step = rows.length ? (rows[rows.length - 1].step ?? null) : null;
  const eta = useMemo(() => {
    if (phase !== "running" || !total || step == null || rows.length < 2) return null;
    const first = rows.find((r) => typeof r.time === "number" && typeof r.step === "number");
    const last = rows[rows.length - 1];
    if (!first || typeof first.time !== "number" || typeof last.time !== "number") return null;
    const dt = last.time - first.time;
    const ds = (last.step ?? 0) - (first.step ?? 0);
    if (dt <= 0 || ds <= 0 || step >= total) return null;
    return Math.round(((total - step) * dt) / ds);
  }, [phase, total, step, rows]);
  const pct = total && step != null ? Math.min(100, Math.round((step / total) * 100)) : null;
  const ph = PHASES[phase] ?? { text: phase, cls: "ph-idle" };

  const start = useCallback(async () => {
    if (!configSel) {
      setMsg({ kind: "err", text: "请先选择配置(可到配置页生成)" });
      return;
    }
    setBusy(true);
    setMsg(null);
    try {
      // 携带该配置在配置页"高级设置"持久化的覆盖项(ui/state/overrides/<name>.json);
      // 读取失败按空覆盖启动(后端缺省也会兜底读取同一文件)。
      const ov = await fetchConfigOverrides(configSel).catch(() => ({} as Record<string, string>));
      let r = await postTrainStart(configSel, ov, resumeChk);
      if (r.status === 409 && r.body.run_exists) {
        if (
          window.confirm(
            "run 目录已有产物。从最新 checkpoint 继续训练(RESUME=latest)?\n\n取消 = 放弃启动(全新重跑需先改名 RUN_NAME 或删除/移走旧 run 目录)。",
          )
        ) {
          r = await postTrainStart(configSel, ov, true);
        } else {
          setBusy(false);
          return;
        }
      }
      if (r.body.ok) {
        const n = r.body.override_count ?? Object.keys(ov).length;
        const warn = r.body.override_warnings?.length
          ? `;注意:${r.body.override_warnings.join(";")}`
          : "";
        const head = r.body.queued
          ? `已排队:${r.body.run_name ?? configSel},数据准备完成后自动开始训练${r.body.resumed ? "(续训)" : ""}`
          : `已启动训练:${r.body.run_name ?? configSel}${r.body.resumed ? "(续训)" : ""}${n ? `(生效覆盖 ${n} 项)` : ""}`;
        setMsg({
          kind: "ok",
          text: `${head}${r.body.note ? `;${r.body.note}` : ""}${warn}`,
        });
        void loadSnapshot();
      } else {
        setMsg({ kind: "err", text: r.body.error ?? `启动失败(HTTP ${r.status})` });
      }
    } catch (e) {
      setMsg({ kind: "err", text: `请求失败:${String(e)}` });
    } finally {
      setBusy(false);
    }
  }, [configSel, resumeChk, loadSnapshot]);

  // 删除选中的配置(含配套 _data.py 与持久化高级设置);二次确认后执行
  const removeConfig = useCallback(async () => {
    if (!configSel) return;
    if (!window.confirm(
      `确定删除配置「${configSel}」吗?该配置的主文件、配套数据文件与高级设置覆盖会一并删除,不可恢复。`)) {
      return;
    }
    setBusy(true);
    try {
      const r = await deleteConfig(configSel);
      if (r.ok) {
        setMsg({ kind: "ok", text: `已删除:${(r.deleted ?? []).join("、")}` });
        setConfigSel("");
        fetchUserConfigs().then(setConfigs).catch(() => setConfigs([]));
      } else {
        setMsg({ kind: "err", text: `删除失败:${r.error}` });
      }
    } catch (e) {
      setMsg({ kind: "err", text: `删除失败:${String(e)}` });
    } finally {
      setBusy(false);
    }
  }, [configSel]);

  const stop = useCallback(async () => {
    if (!task) return;
    if (!window.confirm("确定停止训练?引擎会优雅终止并保存最新 checkpoint(之后勾选「续训」重新启动即可续训)。")) return;
    setBusy(true);
    const r = await postTrainStop(task.id);
    setBusy(false);
    setMsg(r.ok ? { kind: "ok", text: "已请求停止(等待进程退出)" } : { kind: "err", text: r.error ?? "停止失败" });
  }, [task]);

  // 取消排队中的训练(连带取消尚未开始的前置准备任务)
  const cancelQueued = useCallback(async () => {
    if (!task) return;
    if (!window.confirm("确定取消排队中的训练?尚未开始的数据准备任务会一并取消。")) return;
    setBusy(true);
    const r = await postTrainStop(task.id);
    setBusy(false);
    setMsg(r.ok ? { kind: "ok", text: "已取消排队" } : { kind: "err", text: r.error ?? "取消失败" });
    void loadSnapshot();
  }, [task, loadSnapshot]);

  return (
    <div className="train-grid">
      <div className="card">
        <h2>
          训练控制
          <span className={`status-badge ${ph.cls}`}>{ph.text}</span>
          <span className={`ws-dot ${connected ? "on" : ""}`} title={connected ? "WS 已连接" : "WS 未连接(自动重连中)"} />
        </h2>
        <div className="ctrl-row">
          <select value={configSel} disabled={busy} onChange={(e) => setConfigSel(e.target.value)}>
            <option value="">选择配置…</option>
            {configs.map((c) => (
              <option key={c} value={c}>{c}</option>
            ))}
          </select>
          <button className="copy-btn danger-btn" disabled={running || queued || busy || !configSel}
                  onClick={removeConfig}>删除配置</button>
          <label className="ctrl-check">
            <input type="checkbox" checked={resumeChk} disabled={running || queued || busy}
                   onChange={(e) => setResumeChk(e.target.checked)} />
            续训(RESUME=latest)
          </label>
          <button className="primary-btn" disabled={running || queued || busy || !configSel} onClick={start}>
            {busy ? "处理中……" : queued ? "排队中……" : "启动训练"}
          </button>
          {running && (
            <button className="copy-btn danger-btn" disabled={busy} onClick={stop}>停止训练</button>
          )}
          {queued && (
            <button className="copy-btn danger-btn" disabled={busy} onClick={cancelQueued}>取消排队</button>
          )}
        </div>
        {configs.length === 0 && (
          <p className="warn-text">还没有生成过配置:先到 <Link to="/config">训练配置页</Link> 生成。</p>
        )}
        {!configSel && (
          <p className="muted">在上方选择配置,查看该配置最近一次训练的信息(曲线/日志/样图)。</p>
        )}
        {msg && <p className={msg.kind === "ok" ? "ok-text" : "error-text"}>{msg.text}</p>}
        {task && (
          <>
            <div className="progress-row">
              <div className="progress"><i style={{ width: `${pct ?? 0}%` }} /></div>
              <span className="progress-text">
                {step != null ? step : "-"} / {total ?? "?"} 步{pct != null ? `(${pct}%)` : ""}
                {phase === "running" ? ` · 预计剩余 ${fmtEta(eta)}` : ""}
              </span>
            </div>
            <table className="kv">
              <tbody>
                <tr><td>任务</td><td><code>{task.id.slice(0, 8)}</code> · {task.kind} · pid {task.pid ?? "-"}</td></tr>
                <tr><td>配置</td><td><code>{task.config_path ?? "-"}</code></td></tr>
                <tr><td>run 目录</td><td><code>{task.run_dir ?? "-"}</code></td></tr>
                {task.overrides && Object.keys(task.overrides).length > 0 && (
                  <tr><td>启动覆盖参数</td>
                    <td>{Object.entries(task.overrides).map(([k, v]) => <code key={k}>{k}={v}</code>)}</td></tr>
                )}
                {task.exit_code != null && (
                  <tr><td>退出码</td><td className={task.exit_code === 0 ? "ok-text" : "error-text"}>{task.exit_code}</td></tr>
                )}
              </tbody>
            </table>
          </>
        )}
        {queued && (
          <p className="warn-text">正在自动准备数据(文本缓存 → 图像缓存),完成后自动开始训练;进度见「训练成果」页的任务列表。</p>
        )}
        {phase === "initializing" && (
          <p className="warn-text">
            正在准备训练{initInfo?.elapsed ? `(已进行 ${initInfo.elapsed})` : ""}
            {initInfo?.stage ? ` · 当前阶段:${initInfo.stage}` : ""}。
            首次启动需要预热 GPU 内核,可能要十几分钟,详细进度见下方日志。
          </p>
        )}
      </div>

      <div className="card">
        <h2>GPU</h2>
        <GpuPanel gpus={gpus} />
      </div>

      <div className="card">
        <h2>Loss 曲线{status?.validate_every ? ` · 验证每 ${status.validate_every} 步` : ""}</h2>
        <LossChart rows={rows} />
      </div>

      <div className="card">
        <h2>日志(尾部实时)</h2>
        <LogTerm lines={logs} />
      </div>

      <div className="card">
        <h2>采样样图{status?.sample_every != null ? ` · 每 ${status.sample_every} 步` : ""}</h2>
        <SampleWall samples={samples} sampleEvery={status?.sample_every} />
      </div>
    </div>
  );
}
