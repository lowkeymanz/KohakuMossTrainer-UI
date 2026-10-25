"""FastAPI 后端入口(M0 骨架 + M1 配置闭环 + M2 训练监控闭环 + M3 导出与采样)。

路由:
- GET /api/engine/info  -> adapter.probe()(引擎就绪探测,永远返回 JSON)
- GET /api/env/report   -> doctor --json 透传;引擎未就绪返回 503 + 原因
- GET /api/config/presets -> config list 解析结果(运行时拉取,不硬编码)
- GET /api/config/defaults -> 高级设置默认值/枚举/分组(运行引擎资产文本解析)
- GET /api/config/overrides -> 某配置持久化的高级设置(ui/state/overrides)
- POST /api/config/train  -> new train 生成配置;同名已存在返回 409(除非 force)
- POST /api/config/check  -> config check 校验
- GET /api/config/file    -> 读回 configs/user/<name>.py 原文(严格白名单)
- DELETE /api/config/file -> 删除 configs/user/<name>.py(含配套数据配置与高级设置)
- GET /api/config/list    -> configs/user/ 下已有配置名
- POST /api/train/start   -> 启动训练(kogine run;run 已存在需 resume=true,否则 409)
- POST /api/train/{id}/stop / resume -> 优雅终止 / RESUME=latest 重启
- GET  /api/train/status  -> ?config=<名称> 所选配置最近任务 + run 产物快照(曲线/日志/样图/进度)
- GET  /api/tasks         -> 任务表(倒序)
- GET  /api/data/inspect  -> inspect_data.py JSON 直渲
- POST /api/data/cache    -> 文本/latent 缓存长任务
- GET  /api/run-files     -> runs/moss/<run>/samples/ 图片白名单内联
- GET  /api/history       -> 历史任务表(runs/moss 扫描 + SQLite 任务合并)
- GET  /api/runs/{run}/checkpoints -> checkpoint 列表(四件套完整性判定)
- GET  /api/export/formats -> 导出格式选项(运行时探测,不硬编码)
- POST /api/export/start  -> 导出任务(export.py --set;GPU 独占互斥)
- POST /api/sample/start  -> 采样任务(new sample -> sample.py 出图)
- POST /api/grid/start    -> 对比网格任务(compare_grid.py --set)
- GET  /api/gallery       -> 采样图库 + 网格结果(ui/state 下)
- GET  /api/export-files  -> 导出产物下载(runs export 子树白名单)
- GET  /api/gallery-files -> 图库图片内联(ui/state 白名单)
- WS   /ws              -> 事件推送:metric/logline/sample/gpu/task_status/run_reset/log_reset
- 静态托管 ui/frontend/dist(存在时),单端口提供前端。
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# 兼容两种启动方式:包模式(ui.backend.app)与扁平模式(--app-dir backend)
try:
    from . import run_service
    from .adapter import EngineAdapter
    from .config_service import ConfigService
    from .dataset_service import DatasetService
    from .defaults_service import DefaultsService, load_overrides
    from .export_service import ExportService
    from .gpu_monitor import GpuMonitor
    from .sample_service import SampleService
    from .task_manager import TaskManager, RUNNING
    from .train_service import TrainService
except ImportError:  # pragma: no cover
    import run_service
    from adapter import EngineAdapter
    from config_service import ConfigService
    from dataset_service import DatasetService
    from defaults_service import DefaultsService, load_overrides
    from export_service import ExportService
    from gpu_monitor import GpuMonitor
    from sample_service import SampleService
    from task_manager import TaskManager, RUNNING
    from train_service import TrainService


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """启动 WS 监控循环;退出时取消(采集异常绝不拖垮服务)。"""
    monitor = asyncio.create_task(_monitor_loop())
    try:
        yield
    finally:
        monitor.cancel()


app = FastAPI(title="KohakuMossTrainer UI", version="0.3-m3", lifespan=_lifespan)

# 允许本地前端(dev server 任意端口)跨域
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 模块级单例:适配器、任务管理器、配置/数据集/训练/导出/采样服务与 GPU 采样
# (启动即完成建库/恢复)
adapter = EngineAdapter()
task_manager = TaskManager()
defaults_service = DefaultsService(adapter)
config_service = ConfigService(adapter)
dataset_service = DatasetService()
train_service = TrainService(adapter, task_manager, config_service)
export_service = ExportService(adapter, task_manager)
sample_service = SampleService(adapter, task_manager, config_service)
gpu_monitor = GpuMonitor()

# ---------- WS 事件中心(M2) ----------

ws_clients: set[WebSocket] = set()
_task_status_seen: dict[str, str] = {}   # task_id -> 上次广播的状态
_gpu_tick = 0                            # 每 2 个 1s tick 推一帧 GPU


async def ws_broadcast(event: dict) -> None:
    """向全部在线 WS 客户端广播;发送失败的连接直接摘除。"""
    dead = []
    for ws in list(ws_clients):
        try:
            await ws.send_json(event)
        except Exception:
            dead.append(ws)
    for ws in dead:
        ws_clients.discard(ws)


async def _broadcast(events: list[dict]) -> None:
    for ev in events:
        await ws_broadcast(ev)


def _monitor_tick() -> list[dict]:
    """一次采集(同步 IO,由监控循环放进线程执行):返回待广播事件列表。"""
    events: list[dict] = []
    tasks = task_manager.list_tasks()
    # 任务状态变化(含缓存任务)
    for t in tasks:
        if _task_status_seen.get(t["id"]) != t["status"]:
            _task_status_seen[t["id"]] = t["status"]
            events.append({"type": "task_status", "task": t})
    # 运行中的 GPU 任务:增量产物
    for t in tasks:
        if t["status"] != RUNNING or t["kind"] not in ("train", "cache_text",
                                                       "cache_latents"):
            continue
        w = train_service.watcher_for(t)
        rows, rotated = w.poll_metrics()
        if rotated:
            events.append({"type": "run_reset", "task_id": t["id"],
                           "run_dir": t.get("run_dir"), "reason": "rotation"})
        for row in rows:
            events.append({"type": "metric", "task_id": t["id"], "data": row})
        for line in w.poll_log(t.get("log_path")):
            events.append({"type": "logline", "task_id": t["id"], "line": line})
        for path in w.poll_samples():
            events.append({"type": "sample", "task_id": t["id"], "path": path})
    # GPU 采样(每 2s 一帧;不可用时静默)
    global _gpu_tick
    _gpu_tick += 1
    if _gpu_tick % 2 == 0 and gpu_monitor.available:
        snap = gpu_monitor.snapshot()
        if snap is not None:
            events.append({"type": "gpu", "gpus": snap})
    return events


async def _monitor_loop() -> None:
    while True:
        try:
            events = await asyncio.to_thread(_monitor_tick)
            if events:
                await _broadcast(events)
        except asyncio.CancelledError:
            raise
        except Exception:
            pass  # 监控循环永不因采集异常退出(坏了不影响训练)
        await asyncio.sleep(1.0)


@app.get("/api/engine/info")
def engine_info(fresh: bool = False) -> dict:
    """引擎探测结果:{engine_ready, doctor, missing, error};fresh=True 绕过缓存。"""
    return adapter.probe(fresh=fresh)


@app.get("/api/env/report")
def env_report():
    """doctor JSON 原样透传;引擎未就绪时 503 并给出原因。"""
    info = adapter.probe()
    if not info["engine_ready"]:
        raise HTTPException(status_code=503, detail={
            "reason": "引擎环境未安装/未就绪",
            "missing": info["missing"],
            "error": info["error"],
        })
    return info["doctor"]


# ---------- M1 配置闭环 ----------

def _require_engine() -> None:
    """config 端点的前置检查:引擎 CLI 缺失时统一 503。"""
    if config_service._exe() is None:
        raise HTTPException(status_code=503, detail={
            "reason": "引擎环境未安装/未就绪",
            "error": "kohakumoss-config 不存在(引擎未安装)",
        })


@app.get("/api/config/weights")
def config_weights() -> dict:
    """检测项目内已集成的权重目录(纯目录结构判断,不需要引擎就绪)。
    found=True 时前端预填 path;否则留空,placeholder 提示外部 ComfyUI 路径。"""
    return config_service.detect_weights()


@app.get("/api/config/presets")
def config_presets(all: bool = False, fresh: bool = False) -> dict:
    """config list 解析结果(recipes/presets/memory profiles/vram 档位);
    解析失败时 parsed=False 并透传原文,不阻塞前端;fresh=True 绕过缓存。"""
    _require_engine()
    return config_service.list_presets(everything=all, fresh=fresh)


@app.get("/api/config/defaults")
def config_defaults(source: str = "kohya", algo: str = "lora",
                    vram: str = "auto", precision: str = "") -> dict:
    """高级设置面板数据源:运行引擎真实资产的默认值/枚举/分组(纯只读文本
    解析,不跑引擎 CLI,故不套 _require_engine);组合参数与 new train 一致。"""
    return defaults_service.get_defaults(source=source, algo=algo,
                                         vram=vram, precision=precision)


@app.get("/api/config/overrides")
def config_overrides(name: str) -> dict:
    """读取某配置持久化的高级设置(ui/state/overrides/<name>.json)。"""
    overrides = load_overrides(name)
    return {"ok": True, "overrides": overrides, "count": len(overrides)}


@app.post("/api/config/train")
def config_train(form: dict) -> dict:
    """向导表单 -> new train 生成配置;同名已存在且未 force 返回 409。"""
    _require_engine()
    result = config_service.new_train(form or {})
    if result.get("exists"):
        raise HTTPException(status_code=409, detail=result)
    return result


@app.post("/api/config/check")
def config_check(body: dict) -> dict:
    """校验配置:{name: <configs/user 下的名字或仓库内路径>}。
    配置不合法是数据(exit_code=1, ok=False),仍返回 200。"""
    _require_engine()
    return config_service.check_config((body or {}).get("name", ""))


@app.get("/api/config/file")
def config_file(name: str) -> dict:
    """读回 configs/user/<name>.py 原文;严格白名单,非法名 400、不存在 404。"""
    result = config_service.read_user_config(name)
    if not result["ok"]:
        if result.get("not_found"):
            raise HTTPException(status_code=404, detail=result["error"])
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.get("/api/config/list")
def config_list_user() -> dict:
    """列出 configs/user/ 下已有配置名。"""
    return config_service.list_user_configs()


@app.delete("/api/config/file")
def config_delete(name: str) -> dict:
    """删除 configs/user/<name>.py(含配套 _data.py 与持久化高级设置);
    非法名 400、不存在 404。"""
    result = config_service.delete_user_config(name)
    if not result["ok"]:
        if result.get("not_found"):
            raise HTTPException(status_code=404, detail=result["error"])
        raise HTTPException(status_code=400, detail=result["error"])
    return result


# ---------- M1.5 数据集库(纯文件管理,不需要引擎就绪,不套 _require_engine) ----------

@app.get("/api/datasets")
def datasets_list() -> dict:
    """扫描 ui/state/datasets/:每项 {name, image_count, caption_count, created}。"""
    return dataset_service.list_datasets()


@app.post("/api/datasets")
def datasets_create(body: dict) -> dict:
    """建库:{name: <名称>}。名称非法/已存在是业务结果:200 + ok=False。"""
    return dataset_service.create((body or {}).get("name", ""))


@app.delete("/api/datasets/{name}")
def datasets_delete(name: str, confirm: bool = False) -> dict:
    """删库(连图片/caption);必须 confirm=true,否则 400。"""
    if not confirm:
        raise HTTPException(status_code=400, detail="删除数据集需要确认参数 confirm=true")
    result = dataset_service.delete(name, confirm=True)
    if not result["ok"]:
        if result.get("not_found"):
            raise HTTPException(status_code=404, detail=result["error"])
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.post("/api/datasets/{name}/files")
async def datasets_upload(name: str, files: list[UploadFile] = File(...)) -> dict:
    """multipart 上传:只收图片与 .txt caption,其余拒绝并在 rejected 中报告。"""
    pairs: list[tuple[str, bytes]] = []
    for f in files:
        pairs.append((f.filename or "", await f.read()))
    result = dataset_service.upload(name, pairs)
    if not result["ok"] and result.get("not_found"):
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/api/datasets/{name}/items")
def datasets_items(name: str) -> dict:
    """列出库内图片文件名 + 各自 caption 文本(缩略图/编辑弹窗数据源)。"""
    result = dataset_service.items(name)
    if not result["ok"] and result.get("not_found"):
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.put("/api/datasets/{name}/caption")
def datasets_save_caption(name: str, body: dict) -> dict:
    """写同名 .txt(UTF-8):{image: <图片文件名>, text: <caption 文本>}。"""
    body = body or {}
    result = dataset_service.save_caption(name, str(body.get("image", "")),
                                          str(body.get("text", "")))
    if not result["ok"]:
        if result.get("not_found"):
            raise HTTPException(status_code=404, detail=result["error"])
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.get("/api/files")
def files_inline(path: str):
    """内联显示数据集库内图片:白名单仅 ui/state/datasets/ 前缀,
    resolve 后复核防穿越,只放行图片扩展名。非法 400,不存在 404。"""
    result = dataset_service.resolve_file(path)
    if not result["ok"]:
        if result.get("not_found"):
            raise HTTPException(status_code=404, detail=result["error"])
        if result.get("forbidden"):
            raise HTTPException(status_code=403, detail=result["error"])
        raise HTTPException(status_code=400, detail=result["error"])
    return FileResponse(result["path"])


# ---------- M2 训练监控闭环 ----------

@app.post("/api/train/start")
async def train_start(body: dict) -> dict:
    """启动训练:{config, overrides?, resume?}。

    run 目录已有产物且未 resume -> 409(续训场景,需显式 resume=true);
    引擎缺失 -> 503;其余业务失败(GPU 占用等)200 + ok=False。
    """
    body = body or {}
    result = train_service.start(str(body.get("config", "")),
                                 body.get("overrides") or {},
                                 bool(body.get("resume")))
    if result.get("run_exists"):
        raise HTTPException(status_code=409, detail=result)
    if result.get("engine_missing"):
        raise HTTPException(status_code=503, detail=result)
    if result.get("ok"):
        t = result["task"]
        # 前端据此清屏:全新重跑清曲线+日志+样图;续训只清日志(曲线按 step 重建)
        ev = ({"type": "log_reset", "task_id": t["id"], "reason": "resume"}
              if result.get("resumed") else
              {"type": "run_reset", "task_id": t["id"], "run_dir": t.get("run_dir"),
               "reason": "fresh"})
        await ws_broadcast(ev)
    return result


@app.post("/api/train/{task_id}/stop")
def train_stop(task_id: str) -> dict:
    """停止 = 优雅终止(CTRL_BREAK_EVENT/SIGTERM,checkpoint 由引擎收尾)。"""
    return train_service.stop(task_id)


@app.post("/api/train/{task_id}/resume")
async def train_resume(task_id: str) -> dict:
    """继续 = --set RESUME=latest 重启。"""
    result = train_service.resume(task_id)
    if result.get("ok"):
        await ws_broadcast({"type": "log_reset", "task_id": result["task"]["id"],
                            "reason": "resume"})
    return result


@app.get("/api/train/status")
def train_status(config: str | None = None) -> dict:
    """所选配置最近一次训练任务 + run 产物快照;未选配置或无任务时 phase=idle。"""
    if not config:
        return {"ok": True, "task": None, "phase": "idle"}
    return train_service.status(config)


@app.get("/api/tasks")
def tasks_list() -> dict:
    """任务表(倒序),供历史/状态展示。"""
    return {"ok": True, "tasks": task_manager.list_tasks()}


@app.get("/api/data/inspect")
def data_inspect(config: str) -> dict:
    """同步跑 inspect_data.py 并解析 JSON(300s 超时,引擎要加载 torch)。"""
    return train_service.inspect(config)


@app.post("/api/data/cache")
def data_cache(body: dict) -> dict:
    """启动文本/latent 缓存长任务:{config, kind: cache_text|cache_latents}。"""
    body = body or {}
    return train_service.start_cache(str(body.get("config", "")),
                                     str(body.get("kind", "")))


@app.get("/api/run-files")
def run_files_inline(path: str):
    """内联显示 run 采样图:白名单仅 runs/moss/<run>/samples/ 子树内图片,
    resolve 后复核防穿越。非法 400,越权 403,不存在 404。"""
    result = run_service.resolve_sample_file(adapter.repo_root, path)
    if not result["ok"]:
        if result.get("not_found"):
            raise HTTPException(status_code=404, detail=result["error"])
        if result.get("forbidden"):
            raise HTTPException(status_code=403, detail=result["error"])
        raise HTTPException(status_code=400, detail=result["error"])
    return FileResponse(result["path"])


# ---------- M3 导出与采样 ----------

@app.get("/api/history")
def history() -> dict:
    """历史任务表:runs/moss 扫描摘要 + SQLite 任务记录(前端按 run_dir 关联展示)。"""
    runs = run_service.list_runs(adapter.repo_root)
    return {"ok": runs["ok"], "runs": runs["runs"],
            "tasks": task_manager.list_tasks(), "error": runs["error"]}


@app.get("/api/runs/{run}/checkpoints")
def run_checkpoints(run: str) -> dict:
    """checkpoint 列表:四件套存在性/大小 + usable/complete + 各自导出产物。"""
    result = run_service.list_checkpoints(adapter.repo_root, run)
    if not result["ok"] and result.get("not_found"):
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/api/export/formats")
def export_formats() -> dict:
    """导出格式选项(运行时探测束脚本 FORMAT 注释 + ALIASES 别名;probed=False
    表示解析失败已降级为文档清单)。"""
    return export_service.formats()


@app.post("/api/export/start")
def export_start(body: dict) -> dict:
    """导出任务:{run, checkpoint, format, multiplier?, save_dtype?, style?}。
    GPU 占用等业务失败 200 + ok=False;引擎缺失 503。"""
    result = export_service.start(body or {})
    if result.get("engine_missing"):
        raise HTTPException(status_code=503, detail=result)
    return result


@app.post("/api/sample/start")
def sample_start(body: dict) -> dict:
    """采样任务:{run, checkpoint?, prompts?, vram, precision?, name?, multiplier?,
    steps?};prompts 缺省用该 run 的 SAMPLE_PROMPTS。引擎缺失 503。"""
    result = sample_service.start(body or {})
    if result.get("engine_missing"):
        raise HTTPException(status_code=503, detail=result)
    return result


@app.post("/api/grid/start")
def grid_start(body: dict) -> dict:
    """对比网格任务:{run, checkpoints[], prompts?, name?, steps?, cfg?, size?,
    precision?, memory_profile?}。引擎缺失 503。"""
    result = sample_service.start_grid(body or {})
    if result.get("engine_missing"):
        raise HTTPException(status_code=503, detail=result)
    return result


@app.get("/api/gallery")
def gallery() -> dict:
    """采样图库 + 网格结果(ui/state/samples|grids,按修改时间倒序)。"""
    return sample_service.list_gallery()


@app.get("/api/export-files")
def export_files_download(path: str):
    """下载导出产物:白名单 runs/moss/<run>/export/ 与 checkpoints/step_*/export/
    子树;attachment  disposition。非法 400,越权 403,不存在 404。"""
    result = run_service.resolve_export_file(adapter.repo_root, path)
    if not result["ok"]:
        if result.get("not_found"):
            raise HTTPException(status_code=404, detail=result["error"])
        if result.get("forbidden"):
            raise HTTPException(status_code=403, detail=result["error"])
        raise HTTPException(status_code=400, detail=result["error"])
    return FileResponse(result["path"], filename=result["path"].name)


@app.get("/api/gallery-files")
def gallery_files_inline(path: str):
    """内联显示图库图片:白名单 ui/state/samples/ 与 ui/state/grids/ 子树,
    resolve 后复核防穿越,只放行图片扩展名。非法 400,越权 403,不存在 404。"""
    result = sample_service.resolve_gallery_file(path)
    if not result["ok"]:
        if result.get("not_found"):
            raise HTTPException(status_code=404, detail=result["error"])
        if result.get("forbidden"):
            raise HTTPException(status_code=403, detail=result["error"])
        raise HTTPException(status_code=400, detail=result["error"])
    return FileResponse(result["path"])


@app.websocket("/ws")
async def ws_events(websocket: WebSocket):
    """WS 事件通道:注册进广播中心后持续接收推送(metric/logline/sample/gpu/
    task_status/run_reset/log_reset),并保留心跳便于前端判活。"""
    await websocket.accept()
    ws_clients.add(websocket)
    try:
        await websocket.send_json({"type": "hello", "ts": time.time()})
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
                if msg == "ping":
                    await websocket.send_json({"type": "heartbeat",
                                                "ts": time.time()})
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "heartbeat", "ts": time.time()})
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        ws_clients.discard(websocket)


# 前端构建产物静态托管:dist 存在才挂载(未构建时 API 仍可用)。
# SPA 深链支持:/config 等前端路由直接访问时回退 index.html(StaticFiles 挂载
# 会吞掉未注册路径的 404,故用 catch-all 路由自行服务并做回退)。
_dist = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if _dist.is_dir():

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_fallback(full_path: str):
        candidate = (_dist / full_path).resolve()
        if full_path and candidate.is_file() and candidate.is_relative_to(_dist.resolve()):
            return FileResponse(candidate)
        return FileResponse(_dist / "index.html")
