// 应用外壳:侧边导航 + 路由出口(五个页面均已实现)
import { NavLink, Navigate, Route, Routes } from "react-router-dom";
import ConfigPage from "./pages/ConfigPage";
import DatasetPage from "./pages/DatasetPage";
import EnvPage from "./pages/EnvPage";
import ExportPage from "./pages/ExportPage";
import TrainingPage from "./pages/TrainingPage";

const NAV: { to: string; label: string }[] = [
  { to: "/env", label: "环境与模型" },
  { to: "/datasets", label: "数据集" },
  { to: "/config", label: "训练配置" },
  { to: "/train", label: "训练监控" },
  { to: "/export", label: "训练成果" },
];

export default function App() {
  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand">KohakuMoss UI</div>
        <nav>
          {NAV.map((item) => (
            <NavLink key={item.to} to={item.to}>
              {item.label}
            </NavLink>
          ))}
        </nav>
      </aside>
      <main className="content">
        <Routes>
          <Route path="/" element={<Navigate to="/env" replace />} />
          <Route path="/env" element={<EnvPage />} />
          <Route path="/datasets" element={<DatasetPage />} />
          <Route path="/config" element={<ConfigPage />} />
          <Route path="/train" element={<TrainingPage />} />
          <Route path="/export" element={<ExportPage />} />
          <Route path="*" element={<Navigate to="/env" replace />} />
        </Routes>
      </main>
    </div>
  );
}
