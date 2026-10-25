// 页面 E:训练成果(M4 精简版)。
// 只保留两件事:checkpoint 成品下载 + 历史任务列表。
// 依据:引擎在训练过程中已自动导出每个 checkpoint 的 ComfyUI/LyCORIS 成品
// (checkpoints/step_X/export/),训练中预览在训练监控页实时展示效果;
// 手动导出、采样出图、对比网格、图库对目标受众冗余,已从页面移除
// (后端 /api/export、/api/sample、/api/grid、/api/gallery 接口保留未删)。
import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import type { CheckpointRow, HistoryResult, RunSummary, TrainTaskRow } from "../api";
import {
  exportFileUrl,
  fetchCheckpoints,
  fetchHistory,
  openEventSocket,
  postTrainStop,
} from "../api";

const KIND_LABELS: Record<string, string> = {
  train: "训练",
  cache_text: "文本缓存",
  cache_latents: "图像缓存",
  sample: "采样",
  grid: "对比网格",
  export: "导出",
};

const TASK_STATUS: Record<string, { text: string; cls: string }> = {
  pending: { text: "排队中", cls: "ph-idle" },
  running: { text: "运行中", cls: "ph-run" },
  exited_ok: { text: "已完成", cls: "ph-done" },
  exited_fail: { text: "失败", cls: "ph-fail" },
  killed: { text: "已停止", cls: "ph-stop" },
};

function fmtSize(n: number | null | undefined): string {
  if (n == null) return "-";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

function fmtTime(iso: string | number | null): string {
  if (!iso) return "-";
  const s = String(iso);
  const m = s.match(/^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})/);
  if (!m) return s;
  // 后端存 UTC,转本地时区显示
  const d = new Date(s.endsWith("Z") || s.includes("+") ? s : `${s}Z`);
  if (isNaN(d.getTime())) return s;
  const p = (v: number) => String(v).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
}

function taskRunLabel(t: TrainTaskRow): string {
  if (!t.run_dir) return "-";
  const parts = t.run_dir.replace(/\\/g, "/").split("/");
  return parts[parts.length - 1] || "-";
}

// ---------- checkpoint 表(成品直接下载) ----------

function ckptBadge(on: boolean, label: string, title: string) {
  return (
    <span className={on ? "ok-text" : "muted"} title={title} style={{ marginRight: 8 }}>
      {on ? "✓" : "·"} {label}
    </span>
  );
}

function CheckpointTable({ rows }: { rows: CheckpointRow[] }) {
  if (rows.length === 0) {
    return <p className="muted">该 run 暂无 checkpoint(训练产出后自动出现)。</p>;
  }
  return (
    <table>
      <thead>
        <tr>
          <th>步数</th>
          <th>文件完整性</th>
          <th>成品(点击下载,直接用于 ComfyUI)</th>
          <th>时间</th>
        </tr>
      </thead>
      <tbody>
        {[...rows].reverse().map((c) => (
          <tr key={c.dir}>
            <td>
              <b>{c.step != null ? `step ${c.step}` : c.dir}</b>
              {!c.usable && <span className="warn-text"> (不完整)</span>}
            </td>
            <td>
              {ckptBadge(c.has_trainable, "权重", "LoRA 训练权重")}
              {ckptBadge(!!c.files["ema.safetensors"], "平滑权重", "启用 EMA 时才有")}
              {ckptBadge(!!c.files["optim.pt"], "续训数据", "继续训练时需要")}
              {ckptBadge(!!c.files["state.json"], "完整", "落盘完成标记")}
            </td>
            <td>
              {c.exports.length === 0 ? (
                <span className="muted">-</span>
              ) : (
                c.exports.map((e) => (
                  <div key={e.path}>
                    {e.folder ? (
                      <span className="muted">{e.name}/(目录)</span>
                    ) : (
                      <a href={exportFileUrl(e.path)} target="_blank" rel="noreferrer">
                        {e.name}
                      </a>
                    )}{" "}
                    <span className="muted">{fmtSize(e.size)}</span>
                  </div>
                ))
              )}
            </td>
            <td className="muted">
              {fmtTime(c.files["state.json"]?.mtime ? new Date(c.files["state.json"].mtime * 1000).toISOString() : null)}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// ---------- 主页面 ----------

export default function ExportPage() {
  const [history, setHistory] = useState<HistoryResult | null>(null);
  const [runSel, setRunSel] = useState("");
  const [checkpoints, setCheckpoints] = useState<CheckpointRow[]>([]);
  const [connected, setConnected] = useState(false);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ kind: "ok" | "err"; text: string } | null>(null);
  const mounted = useRef(true);
  const runSeq = useRef(0);
  const runSelRef = useRef("");

  useEffect(() => () => { mounted.current = false; }, []);
  useEffect(() => { runSelRef.current = runSel; }, [runSel]);

  const reload = useCallback(async () => {
    try {
      const h = await fetchHistory();
      if (!mounted.current) return;
      setHistory(h);
      setRunSel((cur) => cur || h.runs[0]?.run || "");
    } catch {
      /* 后端不可达时由 WS 重连兜底 */
    }
  }, []);

  const loadCheckpoints = useCallback(async (run: string) => {
    const seq = ++runSeq.current;
    try {
      const r = await fetchCheckpoints(run);
      if (!mounted.current || seq !== runSeq.current) return;
      setCheckpoints(r.checkpoints ?? []);
    } catch (e) {
      if (mounted.current && seq === runSeq.current) {
        setCheckpoints([]);
        setMsg({ kind: "err", text: `checkpoint 加载失败:${String(e)}` });
      }
    }
  }, []);

  useEffect(() => {
    void reload();
    const close = openEventSocket((ev) => {
      if (!mounted.current) return;
      if (ev.type === "hello") void reload();
      if (ev.type === "task_status" && ev.task) {
        const t = ev.task;
        // 训练存盘/缓存任务变化时刷新列表(成品随训练自动出现)
        if (["train", "cache_text", "cache_latents"].includes(t.kind)
            && t.status !== "running") {
          void reload();
          const rs = runSelRef.current;
          if (rs) void loadCheckpoints(rs);
        }
      }
    }, setConnected);
    return close;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (runSel) void loadCheckpoints(runSel);
    else setCheckpoints([]);
  }, [runSel, loadCheckpoints]);

  const runs: RunSummary[] = history?.runs ?? [];
  const tasks: TrainTaskRow[] = history?.tasks ?? [];
  const auxTasks = tasks.filter(
    (t) => t.kind.startsWith("cache") || ["sample", "grid", "export"].includes(t.kind),
  );

  const stopTask = useCallback(async (id: string) => {
    setBusy(true);
    try {
      const r = await postTrainStop(id);
      setMsg(r.ok ? { kind: "ok", text: "已请求停止任务" } : { kind: "err", text: r.error || "停止失败" });
      void reload();
    } finally {
      setBusy(false);
    }
  }, [reload]);

  return (
    <div className="train-grid">
      <div className="card">
        <h2>
          训练成果
          <span className={`ws-dot${connected ? " on" : ""}`} title="实时事件连接" />
        </h2>
        <p className="muted">
          训练过程中,每个 checkpoint 都会自动生成可直接使用的 LoRA 成品文件。
          在下方列表点文件名即可下载,放进 ComfyUI 的 loras 目录加载使用;
          训练中途的效果看 <Link to="/train">训练监控</Link> 页的实时预览即可。
        </p>
        {runs.length === 0 ? (
          <p className="muted">
            还没有训练产物。先到 <Link to="/config">训练配置</Link> 生成配置、再到{" "}
            <Link to="/train">训练监控</Link> 启动训练。
          </p>
        ) : (
          <div className="ctrl-row">
            <label>
              <span className="muted">训练任务(run)</span>
              <select value={runSel} onChange={(e) => setRunSel(e.target.value)}>
                {runs.map((r) => (
                  <option key={r.run} value={r.run}>
                    {r.run}(checkpoint {r.checkpoint_count}{r.last_step != null ? ` · 最新 step ${r.last_step}` : ""})
                  </option>
                ))}
              </select>
            </label>
            <span className="muted" style={{ fontSize: 12 }}>
              算法 {runs.find((r) => r.run === runSel)?.algo ?? "-"} · 基模{" "}
              {runs.find((r) => r.run === runSel)?.weights ?? "-"}
            </span>
          </div>
        )}
        {msg && <p className={msg.kind === "ok" ? "ok-text" : "error-text"}>{msg.text}</p>}
      </div>

      <div className="card">
        <h3>Checkpoint 列表</h3>
        <p className="muted" style={{ fontSize: 12 }}>
          按步数从新到旧排列;“成品”列的 comfy 文件即 ComfyUI 可直接加载的 LoRA。
          ✓ 表示该文件存在。
        </p>
        <CheckpointTable rows={checkpoints} />
      </div>

      <div className="card">
        <h3>历史任务</h3>
        {auxTasks.length === 0 ? (
          <p className="muted">暂无数据准备等任务记录。</p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>类型</th><th>状态</th><th>关联</th><th>开始</th><th>结束</th><th>退出码</th><th></th>
              </tr>
            </thead>
            <tbody>
              {auxTasks.slice(0, 30).map((t) => {
                const st = TASK_STATUS[t.status] ?? { text: t.status, cls: "ph-idle" };
                return (
                  <tr key={t.id}>
                    <td>{KIND_LABELS[t.kind] ?? t.kind}</td>
                    <td><span className={`status-badge ${st.cls}`}>{st.text}</span></td>
                    <td className="muted">{taskRunLabel(t)}</td>
                    <td className="muted">{fmtTime(t.started_at)}</td>
                    <td className="muted">{fmtTime(t.finished_at)}</td>
                    <td className={t.exit_code ? "error-text" : "muted"}>
                      {t.exit_code ?? (t.status === "running" ? "-" : 0)}
                    </td>
                    <td>
                      {t.status === "running" && (
                        <button className="copy-btn danger-btn" disabled={busy}
                                onClick={() => void stopTask(t.id)}>
                          停止
                        </button>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
