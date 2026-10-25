// 后端 API 类型与调用封装(字段透传,不硬编码上游清单)

export interface EngineInfo {
  engine_ready: boolean;
  doctor: DoctorReport | null;
  missing: string[];
  error: string | null;
}

export interface Device {
  index: number;
  name: string;
  arch?: string;
  capability?: string;
  memory_gib?: number;
  bf16?: boolean;
  fp8?: boolean;
}

export interface MissingItem {
  what: string;
  fix: string;
}

export interface DoctorReport {
  os: string;
  os_release: string;
  python: string;
  torch_cuda: string | null;
  packages: Record<string, string | null>;
  devices: Device[];
  missing: MissingItem[];
  backend_status: Record<string, string | null>;
  backend_failures: Record<string, string>;
  [key: string]: unknown;
}

export async function fetchEngineInfo(fresh = false): Promise<EngineInfo> {
  const res = await fetch(`/api/engine/info${fresh ? "?fresh=1" : ""}`);
  if (!res.ok) throw new Error(`engine/info HTTP ${res.status}`);
  return res.json();
}

// /api/env/report 在引擎未就绪时返回 503,detail 里带原因
export async function fetchEnvReport(): Promise<DoctorReport> {
  const res = await fetch("/api/env/report");
  if (res.status === 503) {
    const body = await res.json().catch(() => null);
    const detail = body?.detail ?? {};
    throw new Error(detail.reason || "引擎环境未就绪");
  }
  if (!res.ok) throw new Error(`env/report HTTP ${res.status}`);
  return res.json();
}

// ---------- M1 配置闭环 ----------

export interface MemoryProfile {
  name: string;
  checkpoint: string;
  weights: string;
  act_offload: string;
  optimizer: string;
  grad_offload: string;
  lora_gib: number | string;
  full_ft_gib: number | string;
}

export interface PresetsResult {
  ok: boolean;
  parsed: boolean;
  error: string | null;
  recipes: string[];
  presets: Record<string, Record<string, string[]>>;
  memory_profiles: MemoryProfile[];
  catalog_more: number | null;
  vram_options?: (number | string)[];
  stdout?: string;
}

// ---------- M1.5 数据集库 ----------

export interface DatasetSummary {
  name: string;
  image_count: number;
  caption_count: number;
  created: string;
}

export interface DatasetItem {
  image: string;
  path: string; // 绝对路径,供 /api/files?path= 内联显示
  caption: string | null;
}

export interface DatasetResult {
  ok: boolean;
  error?: string | null;
  [key: string]: unknown;
}

export interface UploadResult extends DatasetResult {
  saved?: string[];
  overwritten?: string[];
  rejected?: { name: string; reason: string }[];
}

export async function fetchDatasets(): Promise<DatasetSummary[]> {
  const res = await fetch("/api/datasets");
  if (!res.ok) throw new Error(`datasets HTTP ${res.status}`);
  const body = await res.json();
  return (body.datasets ?? []) as DatasetSummary[];
}

// 后端业务失败也是 200 + ok=False,统一返回结构化结果
async function jsonResult(res: Response): Promise<DatasetResult> {
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    const detail = typeof body?.detail === "string" ? body.detail : JSON.stringify(body?.detail ?? body);
    return { ok: false, error: detail || `HTTP ${res.status}` };
  }
  return body as DatasetResult;
}

export async function createDataset(name: string): Promise<DatasetResult> {
  const res = await fetch("/api/datasets", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  return jsonResult(res);
}

export async function deleteDataset(name: string): Promise<DatasetResult> {
  const res = await fetch(`/api/datasets/${encodeURIComponent(name)}?confirm=true`, {
    method: "DELETE",
  });
  return jsonResult(res);
}

export async function uploadDatasetFiles(name: string, files: File[]): Promise<UploadResult> {
  const fd = new FormData();
  files.forEach((f) => fd.append("files", f));
  const res = await fetch(`/api/datasets/${encodeURIComponent(name)}/files`, {
    method: "POST",
    body: fd,
  });
  return (await jsonResult(res)) as UploadResult;
}

export async function fetchDatasetItems(name: string): Promise<DatasetItem[]> {
  const res = await fetch(`/api/datasets/${encodeURIComponent(name)}/items`);
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail || `items HTTP ${res.status}`);
  }
  const body = await res.json();
  return (body.items ?? []) as DatasetItem[];
}

export async function saveDatasetCaption(name: string, image: string, text: string): Promise<DatasetResult> {
  const res = await fetch(`/api/datasets/${encodeURIComponent(name)}/caption`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image, text }),
  });
  return jsonResult(res);
}

// /api/files 白名单图片 URL(后端仅放行 ui/state/datasets/ 前缀下的图片)
export function datasetFileUrl(path: string): string {
  return `/api/files?path=${encodeURIComponent(path)}`;
}

export interface TrainForm {
  weights: string;
  data?: string[];
  datasets?: string[];
  name: string;
  algo?: string;
  source?: string;
  steps?: number | "";
  prompt?: string | string[];
  // 与 prompt 平行的每条采样分辨率 [高, 宽](引擎 SAMPLE_SIZE 顺序)
  prompt_sizes?: number[][];
  trigger?: string;
  vram?: string;
  precision?: string;
  force?: boolean;
  overrides?: Record<string, string>;
}

export interface TrainResult {
  ok: boolean;
  error?: string | null;
  exists?: boolean;
  exit_code?: number | null;
  name?: string;
  path?: string;
  content?: string | null;
  data_content?: string | null;
  stdout?: string;
  stderr?: string;
  deferred_overrides?: Record<string, string>;
  note?: string;
  override_warnings?: string[];
  datasets_used?: string[];
  datasets_appended?: string[];
  trigger_applied?: string;
  check?: { ok: boolean; exit_code: number | null; stdout: string; stderr: string };
}

export interface CheckResult {
  ok: boolean;
  error?: string | null;
  exit_code?: number | null;
  stdout: string;
  stderr: string;
  path?: string;
}

export async function fetchPresets(fresh = false): Promise<PresetsResult> {
  const res = await fetch(`/api/config/presets${fresh ? "?fresh=1" : ""}`);
  if (res.status === 503) throw new Error("引擎环境未安装/未就绪");
  if (!res.ok) throw new Error(`config/presets HTTP ${res.status}`);
  return res.json();
}

// 项目内集成权重目录检测(不需要引擎就绪)
export interface WeightsDetect {
  ok: boolean;
  found: boolean;
  path: string | null;
  relative: string | null;
  layout: "comfy" | "hf" | null;
}

export async function fetchWeightsDetect(): Promise<WeightsDetect> {
  const res = await fetch("/api/config/weights");
  if (!res.ok) throw new Error(`config/weights HTTP ${res.status}`);
  return res.json();
}

export async function postTrain(form: TrainForm): Promise<{ status: number; body: TrainResult }> {
  const res = await fetch("/api/config/train", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(form),
  });
  const body = (await res.json().catch(() => ({ ok: false, error: `HTTP ${res.status}` }))) as
    TrainResult & { detail?: TrainResult };
  return { status: res.status, body: (res.status === 409 && body.detail ? body.detail : body) };
}

export async function postCheck(name: string): Promise<CheckResult> {
  const res = await fetch("/api/config/check", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  if (!res.ok) throw new Error(`config/check HTTP ${res.status}`);
  return res.json();
}

export async function fetchUserConfigFile(name: string): Promise<{ name: string; path: string; content: string }> {
  const res = await fetch(`/api/config/file?name=${encodeURIComponent(name)}`);
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail || `config/file HTTP ${res.status}`);
  }
  return res.json();
}

export async function fetchUserConfigs(): Promise<string[]> {
  const res = await fetch("/api/config/list");
  if (!res.ok) throw new Error(`config/list HTTP ${res.status}`);
  const body = await res.json();
  return body.configs ?? [];
}

export async function deleteConfig(
  name: string,
): Promise<{ ok: boolean; deleted?: string[]; error?: string }> {
  const res = await fetch(`/api/config/file?name=${encodeURIComponent(name)}`, {
    method: "DELETE",
  });
  const raw = await res.json().catch(() => null);
  if (res.ok) return raw ?? { ok: true };
  return { ok: false, error: raw?.detail ?? raw?.error ?? `HTTP ${res.status}` };
}

// ---------- 高级设置:默认值(运行引擎资产文本解析) ----------

export interface KnobInfo {
  default: unknown;
  default_text: string;
  type: string; // int/float/str/bool/dict/list/tuple/none/text/unknown
  choices: string[] | null;
  group: string;
  editable: boolean;
  layer_source: string | null;
  note: string | null;
}

export interface DefaultsResult {
  ok: boolean;
  error: string | null;
  source_of_truth?: string;
  locate_note?: string;
  asset_root?: string;
  query?: Record<string, string>;
  missing_layers?: string[];
  groups?: Record<string, string>;
  panel?: string[];
  knobs?: Record<string, KnobInfo>;
  renamed?: Record<string, string>;
  encoder_fields?: Record<string, { knob: string; fields: string[] }>;
  valid_key_count?: number;
}

export async function fetchDefaults(params: {
  source?: string; algo?: string; vram?: string; precision?: string;
} = {}): Promise<DefaultsResult> {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== undefined) q.set(k, v);
  const res = await fetch(`/api/config/defaults?${q.toString()}`);
  if (!res.ok) throw new Error(`config/defaults HTTP ${res.status}`);
  return res.json();
}

export async function fetchConfigOverrides(name: string): Promise<Record<string, string>> {
  const res = await fetch(`/api/config/overrides?name=${encodeURIComponent(name)}`);
  if (!res.ok) throw new Error(`config/overrides HTTP ${res.status}`);
  const body = await res.json();
  return (body.overrides ?? {}) as Record<string, string>;
}

// ---------- M2 训练监控闭环 ----------

export interface TrainTaskRow {
  id: string;
  kind: string;
  status: string; // running / exited_ok / exited_fail / killed
  argv: string[];
  config_path: string | null;
  run_dir: string | null;
  overrides: Record<string, string>;
  pid: number | null;
  log_path: string | null;
  started_at: string | null; // 后端落库的 ISO 时间(UTC)
  finished_at: string | null;
  exit_code: number | null;
}

export interface MetricRow {
  step?: number;
  time?: number;
  loss?: number;
  [key: string]: unknown; // grad_norm/lr/val_loss/val_loss@<sigma>...
}

export interface GpuCard {
  index: number;
  name: string | null;
  util_gpu: number | null;
  util_mem: number | null;
  mem_used_mb: number | null;
  mem_total_mb: number | null;
  temp_c: number | null;
  power_w: number | null;
}

export interface TrainStatus {
  ok: boolean;
  task: TrainTaskRow | null;
  phase: string;
  run_dir?: string | null;
  total_steps?: number | null;
  step?: number | null;
  eta_seconds?: number | null;
  sample_every?: number | null;
  validate_every?: number | null;
  config?: Record<string, unknown> | null;
  metrics?: MetricRow[];
  log_tail?: string[];
  samples?: string[];
}

export interface TrainStartResult {
  ok: boolean;
  task?: TrainTaskRow;
  run_name?: string;
  resumed?: boolean;
  run_exists?: boolean;
  engine_missing?: boolean;
  error?: string | null;
  override_count?: number;
  override_warnings?: string[];
  // 缓存未就绪时自动排队:queued=true,prep 为前置准备任务类型
  queued?: boolean;
  prep?: string[];
  note?: string | null;
  preview_workaround?: boolean;
}

export interface TrainOpResult {
  ok: boolean;
  task?: TrainTaskRow;
  error?: string | null;
}

export interface InspectResult {
  ok: boolean;
  config?: string;
  report?: Record<string, unknown>;
  error?: string | null;
  exit_code?: number | null;
  stdout?: string;
  stderr?: string;
}

// WS 事件:metric/logline/sample/gpu/task_status/run_reset/log_reset/hello/heartbeat
export interface WsEvent {
  type: string;
  task_id?: string;
  data?: MetricRow;
  line?: string;
  path?: string;
  gpus?: GpuCard[];
  task?: TrainTaskRow;
  reason?: string;
  run_dir?: string | null;
}

export async function fetchTrainStatus(config: string): Promise<TrainStatus> {
  const u = config
    ? `/api/train/status?config=${encodeURIComponent(config)}`
    : "/api/train/status";
  const res = await fetch(u);
  if (!res.ok) throw new Error(`train/status HTTP ${res.status}`);
  return res.json();
}

// start 的 409(run 已存在)/503(引擎缺失)统一解包 detail 为业务体
export async function postTrainStart(
  config: string,
  overrides: Record<string, string>,
  resume: boolean,
): Promise<{ status: number; body: TrainStartResult }> {
  const res = await fetch("/api/train/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ config, overrides, resume }),
  });
  const raw = await res.json().catch(() => ({ ok: false, error: `HTTP ${res.status}` }));
  const body = (res.status === 409 || res.status === 503 ? raw.detail ?? raw : raw) as TrainStartResult;
  return { status: res.status, body };
}

async function postTrainOp(url: string): Promise<TrainOpResult> {
  const res = await fetch(url, { method: "POST" });
  const body = await res.json().catch(() => ({ ok: false, error: `HTTP ${res.status}` }));
  return body as TrainOpResult;
}

export const postTrainStop = (taskId: string) =>
  postTrainOp(`/api/train/${encodeURIComponent(taskId)}/stop`);
export const postTrainResume = (taskId: string) =>
  postTrainOp(`/api/train/${encodeURIComponent(taskId)}/resume`);

export async function fetchTasks(): Promise<TrainTaskRow[]> {
  const res = await fetch("/api/tasks");
  if (!res.ok) throw new Error(`tasks HTTP ${res.status}`);
  const body = await res.json();
  return body.tasks ?? [];
}

// inspect 同步跑引擎(可能数分钟),不设前端超时
export async function runInspect(config: string): Promise<InspectResult> {
  try {
    const res = await fetch(`/api/data/inspect?config=${encodeURIComponent(config)}`);
    return await res.json();
  } catch (e) {
    return { ok: false, error: `请求失败:${String(e)}` };
  }
}

export async function postCache(config: string, kind: string): Promise<TrainOpResult> {
  const res = await fetch("/api/data/cache", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ config, kind }),
  });
  return res.json().catch(() => ({ ok: false, error: `HTTP ${res.status}` }));
}

// run 采样图内联 URL(后端白名单仅 runs/moss/<run>/samples/ 子树)
export function runFileUrl(path: string): string {
  return `/api/run-files?path=${encodeURIComponent(path)}`;
}

// ---------- M3 导出与采样 ----------

export interface RunSummary {
  run: string;
  run_dir: string;
  algo: string | null;
  steps: number | string | null;
  weights: string | null;
  sample_every: number | string | null;
  checkpoint_count: number;
  last_step: number | null;
  export_count: number;
  sample_count: number;
  mtime: number;
}

export interface HistoryResult {
  ok: boolean;
  runs: RunSummary[];
  tasks: TrainTaskRow[];
  error: string | null;
}

export interface CheckpointFile {
  name: string;
  size: number;
  mtime: number;
}

export interface ExportArtifact {
  name: string;
  size: number | null;
  mtime: number;
  path: string;
  folder?: boolean;
}

export interface CheckpointRow {
  dir: string;
  step: number | null;
  files: Record<string, CheckpointFile>;
  has_trainable: boolean;
  has_ema: boolean;
  usable: boolean;
  complete: boolean;
  exports: ExportArtifact[];
  path: string;
}

export interface CheckpointsResult {
  ok: boolean;
  run: string;
  run_dir: string;
  checkpoints: CheckpointRow[];
  error: string | null;
}

export interface ExportFormat {
  value: string;
  label: string;
}

export interface ExportFormatsResult {
  ok: boolean;
  probed: boolean;
  formats: ExportFormat[];
  error: string | null;
}

export interface GalleryImage {
  name: string;
  path: string; // 绝对 posix 路径,供 /api/gallery-files?path= 内联显示
  size: number;
  mtime: number;
}

export interface GalleryGroup {
  kind: "sample" | "grid";
  name: string;
  out: string;
  mtime: number;
  images: GalleryImage[];
  metadata: Record<string, unknown>;
}

export interface GalleryResult {
  ok: boolean;
  groups: GalleryGroup[];
  error: string | null;
}

// start 类接口的统一返回:业务失败(GPU 占用等)200 + ok=False
export interface TaskStartResult {
  ok: boolean;
  task?: TrainTaskRow;
  error?: string | null;
  engine_missing?: boolean;
  [key: string]: unknown;
}

export async function fetchHistory(): Promise<HistoryResult> {
  const res = await fetch("/api/history");
  if (!res.ok) throw new Error(`history HTTP ${res.status}`);
  return res.json();
}

export async function fetchCheckpoints(run: string): Promise<CheckpointsResult> {
  const res = await fetch(`/api/runs/${encodeURIComponent(run)}/checkpoints`);
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail || `checkpoints HTTP ${res.status}`);
  }
  return res.json();
}

export async function fetchExportFormats(): Promise<ExportFormatsResult> {
  const res = await fetch("/api/export/formats");
  if (!res.ok) throw new Error(`export/formats HTTP ${res.status}`);
  return res.json();
}

async function postJson(url: string, body: unknown): Promise<TaskStartResult> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const raw = await res.json().catch(() => ({ ok: false, error: `HTTP ${res.status}` }));
  return ((res.status === 503 && raw.detail ? raw.detail : raw)) as TaskStartResult;
}

export const postExportStart = (body: Record<string, unknown>) =>
  postJson("/api/export/start", body);
export const postSampleStart = (body: Record<string, unknown>) =>
  postJson("/api/sample/start", body);
export const postGridStart = (body: Record<string, unknown>) =>
  postJson("/api/grid/start", body);

export async function fetchGallery(): Promise<GalleryResult> {
  const res = await fetch("/api/gallery");
  if (!res.ok) throw new Error(`gallery HTTP ${res.status}`);
  return res.json();
}

// 导出产物下载 URL(后端白名单 runs/moss/<run>/export|checkpoints/*/export 子树)
export function exportFileUrl(path: string): string {
  return `/api/export-files?path=${encodeURIComponent(path)}`;
}

// 图库图片内联 URL(后端白名单 ui/state/samples|grids 子树)
export function galleryFileUrl(path: string): string {
  return `/api/gallery-files?path=${encodeURIComponent(path)}`;
}

// WS 事件通道:断线 1.5s 自动重连;返回关闭函数
export function openEventSocket(
  onEvent: (ev: WsEvent) => void,
  onConn: (up: boolean) => void,
): () => void {
  let ws: WebSocket | null = null;
  let closed = false;
  let timer: number | undefined;
  const connect = () => {
    if (closed) return;
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.onopen = () => onConn(true);
    ws.onmessage = (m) => {
      try {
        onEvent(JSON.parse(m.data) as WsEvent);
      } catch {
        /* 忽略坏帧 */
      }
    };
    ws.onclose = () => {
      onConn(false);
      if (!closed) timer = window.setTimeout(connect, 1500);
    };
    ws.onerror = () => ws?.close();
  };
  connect();
  return () => {
    closed = true;
    window.clearTimeout(timer);
    ws?.close();
  };
}
