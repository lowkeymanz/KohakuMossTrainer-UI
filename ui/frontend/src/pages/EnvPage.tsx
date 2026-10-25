// 页面 A:环境与模型(M0)——渲染 doctor 报告;引擎未就绪时显示明确提示页
import { useCallback, useEffect, useState } from "react";
import type { DoctorReport, EngineInfo } from "../api";
import { fetchEngineInfo, fetchEnvReport } from "../api";

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
    <button className="copy-btn" onClick={copy} title="复制修复命令">
      {copied ? "已复制" : "复制"}
    </button>
  );
}

function NotReady({ info, reason }: { info: EngineInfo | null; reason?: string }) {
  return (
    <div className="card warn-card">
      <h2>引擎环境未安装 / 未就绪</h2>
      <p>
        UI 不会自动重装训练环境。请按仓库 README 完成安装
        (在仓库根创建 <code>.venv</code> 并安装引擎)后,回到本页刷新。
      </p>
      {info && info.missing.length > 0 && (
        <>
          <h3>缺失的引擎命令</h3>
          <ul className="missing-list">
            {info.missing.map((m) => (
              <li key={m}>
                <code>{m}</code> 未找到
              </li>
            ))}
          </ul>
        </>
      )}
      {info?.error && <p className="error-text">探测失败原因:{info.error}</p>}
      {reason && !info?.error && <p className="error-text">原因:{reason}</p>}
    </div>
  );
}

export default function EnvPage() {
  const [info, setInfo] = useState<EngineInfo | null>(null);
  const [report, setReport] = useState<DoctorReport | null>(null);
  const [loadError, setLoadError] = useState<string>("");
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    setLoadError("");
    setReport(null);
    try {
      const engine = await fetchEngineInfo(true); // 环境页必须实探,绕过后端缓存
      setInfo(engine);
      if (engine.engine_ready) {
        try {
          setReport(await fetchEnvReport());
        } catch (e) {
          setLoadError(String(e));
        }
      }
    } catch (e) {
      setInfo(null);
      setLoadError(`无法连接 UI 后端:${String(e)}`);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  if (loading) return <div className="card">正在探测引擎环境……</div>;

  if (loadError && !info) return <div className="card error-text">{loadError}</div>;

  if (!info?.engine_ready) {
    return (
      <>
        <NotReady info={info} reason={loadError} />
        <button className="primary-btn" onClick={load}>重新探测</button>
      </>
    );
  }

  if (!report) {
    return (
      <>
        <div className="card error-text">{loadError || "doctor 报告获取失败"}</div>
        <button className="primary-btn" onClick={load}>重试</button>
      </>
    );
  }

  const missing = Array.isArray(report.missing) ? report.missing : [];
  const devices = Array.isArray(report.devices) ? report.devices : [];

  return (
    <div className="env-grid">
      <div className="card">
        <h2>系统</h2>
        <table className="kv">
          <tbody>
            <tr><td>OS</td><td>{report.os} {report.os_release}</td></tr>
            <tr><td>Python</td><td>{report.python}</td></tr>
            <tr><td>torch CUDA</td><td>{report.torch_cuda ?? "-"}</td></tr>
          </tbody>
        </table>
      </div>

      <div className="card">
        <h2>GPU / 设备</h2>
        {devices.length === 0 ? (
          <p className="muted">未检测到加速设备</p>
        ) : (
          <table>
            <thead>
              <tr><th>#</th><th>名称</th><th>架构</th><th>显存 (GiB)</th><th>BF16</th><th>FP8</th></tr>
            </thead>
            <tbody>
              {devices.map((d) => (
                <tr key={d.index}>
                  <td>{d.index}</td>
                  <td>{d.name}</td>
                  <td>{d.arch ?? "-"}</td>
                  <td>{d.memory_gib ?? "-"}</td>
                  <td>{d.bf16 === undefined ? "-" : d.bf16 ? "✓" : "✗"}</td>
                  <td>{d.fp8 === undefined ? "-" : d.fp8 ? "✓" : "✗"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {devices.length <= 1 && (
          <p className="muted">单卡 / Windows:多 GPU 并行不可用(引擎限制,仅 Linux)。</p>
        )}
      </div>

      <div className="card">
        <h2>依赖包</h2>
        <table className="kv">
          <tbody>
            {Object.entries(report.packages ?? {}).map(([name, ver]) => {
              // Windows 上官方 triton 不可用,由社区移植版 triton-windows 提供同名模块
              const by = !ver && name === "triton" ? (report.packages?.["triton-windows"] as string | undefined) : undefined;
              return (
                <tr key={name}>
                  <td>{name}</td>
                  <td className={ver || by ? "" : "error-text"}>
                    {ver ?? (by ? `由 triton-windows ${by} 提供(Windows 移植)` : "未安装")}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="card">
        <h2>缺失项与修复</h2>
        {missing.length === 0 ? (
          <p className="ok-text">环境完整,无缺失项。</p>
        ) : (
          <ul className="missing-list">
            {missing.map((m, i) => (
              <li key={i}>
                <span className="error-text">{m.what}</span>
                <div className="fix-row">
                  <code>{m.fix}</code>
                  <CopyButton text={m.fix} />
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>

      <button className="primary-btn" onClick={load}>重新探测</button>
    </div>
  );
}
