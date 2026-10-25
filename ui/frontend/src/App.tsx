// 应用外壳:侧边导航 + 路由出口(M1 配置向导、M1.5 数据集库、M2 训练监控、M3 导出与采样可用)
import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import ConfigPage from "./pages/ConfigPage";
import DatasetPage from "./pages/DatasetPage";
import EnvPage from "./pages/EnvPage";
import ExportPage from "./pages/ExportPage";
import TrainingPage from "./pages/TrainingPage";

function Placeholder({ name, milestone }: { name: string; milestone: string }) {
  return (
    <div className="card">
      <h2>{name}</h2>
      <p className="muted">该页面将在 {milestone} 里程碑实现。</p>
    </div>
  );
}

const NAV: { to: string; label: string; exact?: boolean; milestone?: string }[] = [
  { to: "/env", label: "环境与模型", exact: true },
  { to: "/datasets", label: "数据集", exact: true },
  { to: "/config", label: "训练配置", exact: true },
  { to: "/train", label: "训练监控", exact: true },
  { to: "/export", label: "训练成果", exact: true },
];

export default function App() {
  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand">KohakuMoss UI</div>
        <nav>
          {NAV.map((item) =>
            item.exact ? (
              <NavLink key={item.to} to={item.to}>
                {item.label}
              </NavLink>
            ) : (
              <span key={item.to} className="nav-disabled" title={`${item.milestone} 实现`}>
                {item.label} <em>{item.milestone}</em>
              </span>
            ),
          )}
        </nav>
        <div className="sidebar-foot">M3 导出与采样</div>
      </aside>
      <main className="content">
        <Routes>
          <Route path="/" element={<Navigate to="/env" replace />} />
          <Route path="/env" element={<EnvPage />} />
          <Route path="/datasets" element={<DatasetPage />} />
          <Route path="/config" element={<ConfigPage />} />
          <Route path="/train" element={<TrainingPage />} />
          <Route path="/export" element={<ExportPage />} />
          {NAV.filter((n) => !n.exact).map((n) => (
            <Route
              key={n.to}
              path={n.to}
              element={<Placeholder name={n.label} milestone={n.milestone!} />}
            />
          ))}
          <Route path="*" element={<Navigate to="/env" replace />} />
        </Routes>
      </main>
    </div>
  );
}
