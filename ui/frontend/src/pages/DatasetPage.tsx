// 页面 B:数据集库(M1.5,aitoolkit 式)——建库/上传/预览/编辑提示词
// 库固定在后端 ui/state/datasets/<名称>/;图片经 /api/files 白名单内联显示。
import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import type { DatasetItem, DatasetSummary, UploadResult } from "../api";
import {
  createDataset,
  datasetFileUrl,
  deleteDataset,
  fetchDatasetItems,
  fetchDatasets,
  saveDatasetCaption,
  uploadDatasetFiles,
} from "../api";

// 内联打标卡片:图片下方直接是 caption 文本框 + 保存按钮,无二级弹窗
function CaptionCard({
  item,
  dataset,
  onSaved,
}: {
  item: DatasetItem;
  dataset: string;
  onSaved: () => void;
}) {
  const [text, setText] = useState(item.caption ?? "");
  const [saving, setSaving] = useState(false);
  const [status, setStatus] = useState("");
  const [hasCaption, setHasCaption] = useState(item.caption !== null);

  const save = async () => {
    setSaving(true);
    setStatus("");
    const res = await saveDatasetCaption(dataset, item.image, text);
    setSaving(false);
    if (!res.ok) {
      setStatus(res.error || "保存失败");
      return;
    }
    setHasCaption(true);
    setStatus("已保存");
    onSaved();
    window.setTimeout(() => setStatus((s) => (s === "已保存" ? "" : s)), 1500);
  };

  return (
    <div className="cap-card">
      <img src={datasetFileUrl(item.path)} alt={item.image} loading="lazy" />
      <div className="cap-name">
        <span className="cap-file">{item.image}</span>
        {!hasCaption && <span className="cap-missing">无 caption</span>}
      </div>
      <div className="cap-row">
        <textarea
          value={text}
          placeholder="输入提示词(保存为同名 .txt)"
          onChange={(e) => setText(e.target.value)}
        />
        <button className="primary-btn cap-save" disabled={saving} onClick={save}>
          {saving ? "保存中" : "保存"}
        </button>
      </div>
      {/* 状态行常驻(空时占位),保证卡片高度固定不跳动 */}
      <div className={status === "已保存" ? "cap-status ok-text" : status ? "cap-status error-text" : "cap-status"}>
        {status}
      </div>
    </div>
  );
}

export default function DatasetPage() {
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [selected, setSelected] = useState("");
  const [items, setItems] = useState<DatasetItem[]>([]);
  const [newName, setNewName] = useState("");
  const [pageError, setPageError] = useState("");
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [uploadResult, setUploadResult] = useState<UploadResult | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  // 请求序号:快速切换库时,只有最新一次 items 请求能写入 state,防止旧响应覆盖新库
  const itemsSeq = useRef(0);

  const loadItems = useCallback(async (name: string) => {
    const seq = ++itemsSeq.current;
    if (!name) {
      setItems([]);
      return;
    }
    try {
      const list = await fetchDatasetItems(name);
      if (seq === itemsSeq.current) setItems(list);
    } catch (e) {
      if (seq === itemsSeq.current) {
        setItems([]);
        setPageError(String(e));
      }
    }
  }, []);

  const load = useCallback(async () => {
    setPageError("");
    try {
      const list = await fetchDatasets();
      setDatasets(list);
      if (selected && !list.some((d) => d.name === selected)) setSelected("");
      void loadItems(selected && list.some((d) => d.name === selected) ? selected : "");
    } catch (e) {
      setPageError(`无法连接 UI 后端:${String(e)}`);
    }
  }, [selected, loadItems]);

  useEffect(() => {
    void load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const select = (name: string) => {
    setSelected(name);
    setUploadResult(null);
    void loadItems(name);
  };

  const create = async () => {
    setPageError("");
    const trimmed = newName.trim();
    const res = await createDataset(trimmed);
    if (!res.ok) {
      setPageError(res.error || "建库失败");
      return;
    }
    // 建库成功后直接切换到新库:不经 load() 的旧 selected 闭包,避免预览区残留旧库内容
    const name = (res.name as string) || trimmed;
    setNewName("");
    setSelected(name);
    setUploadResult(null);
    void loadItems(name);
    try {
      setDatasets(await fetchDatasets());
    } catch (e) {
      setPageError(`无法刷新数据集列表:${String(e)}`);
    }
  };

  const remove = async (name: string) => {
    if (!window.confirm(`确定删除数据集「${name}」?库内全部图片与 caption 将被移除,不可恢复。`)) return;
    setPageError("");
    const res = await deleteDataset(name);
    if (!res.ok) {
      setPageError(res.error || "删除失败");
      return;
    }
    if (selected === name) setSelected("");
    await load();
  };

  const doUpload = async (files: File[]) => {
    if (!selected || files.length === 0) return;
    setUploading(true);
    setUploadResult(null);
    setPageError("");
    try {
      const res = await uploadDatasetFiles(selected, files);
      setUploadResult(res);
      if (!res.ok) setPageError(res.error || "上传失败");
      await Promise.all([load(), loadItems(selected)]);
    } catch (e) {
      setPageError(`上传失败:${String(e)}`);
    } finally {
      setUploading(false);
    }
  };

  const coverage = (d: DatasetSummary) =>
    d.image_count === 0 ? "—" : `${Math.round((d.caption_count / d.image_count) * 100)}%`;

  return (
    <div className="dataset-grid">
      <div className="card">
        <h2>数据集库</h2>
        {datasets.length === 0 && (
          <p className="muted">
            库是空的。三步开始:<b>1)</b> 在下方输入名称「新建数据集」;<b>2)</b> 选中它并上传
            图片(png/jpg/jpeg/webp/bmp,可带同名 .txt 提示词);<b>3)</b> 回到
            {" "}<Link to="/config">训练配置</Link> 页勾选该数据集生成配置。
          </p>
        )}
        <div className="row-line">
          <input
            value={newName}
            placeholder="新数据集名称(字母/数字/下划线/连字符/中文)"
            onChange={(e) => setNewName(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && create()}
          />
          <button className="primary-btn" disabled={!newName.trim()} onClick={create}>
            新建数据集
          </button>
        </div>
        {pageError && <p className="error-text">{pageError}</p>}
        {datasets.length === 0 ? (
          <p className="muted">暂无数据集。</p>
        ) : (
          <div className="ds-cards">
            {datasets.map((d) => (
              <div
                key={d.name}
                className={`ds-card${selected === d.name ? " active" : ""}`}
                onClick={() => select(d.name)}
              >
                <b>{d.name}</b>
                <span className="muted">
                  {d.image_count} 图 · caption {d.caption_count}/{d.image_count}({coverage(d)})
                </span>
                <span className="muted">创建于 {d.created}</span>
                <button
                  className="copy-btn danger-btn"
                  onClick={(e) => {
                    e.stopPropagation();
                    void remove(d.name);
                  }}
                >
                  删除
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {selected && (
        <div className="card">
          <h2>{selected}:上传与预览</h2>
          <div
            className={`dropzone${dragOver ? " drag-over" : ""}`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              void doUpload(Array.from(e.dataTransfer.files));
            }}
            onClick={() => fileInput.current?.click()}
          >
            {uploading ? "上传中……" : "点击选择或拖拽文件到这里(图片 + 同名 .txt,可多选)"}
            <input
              ref={fileInput}
              type="file"
              multiple
              hidden
              accept=".png,.jpg,.jpeg,.webp,.bmp,.txt"
              onChange={(e) => {
                void doUpload(Array.from(e.target.files ?? []));
                e.target.value = "";
              }}
            />
          </div>
          {uploadResult?.ok && (
            <p className="ok-text">
              已保存 {(uploadResult.saved ?? []).length} 个文件
              {uploadResult.overwritten?.length ? `(覆盖 ${uploadResult.overwritten.length} 个)` : ""}
              。
            </p>
          )}
          {uploadResult?.rejected && uploadResult.rejected.length > 0 && (
            <ul className="reject-list">
              {uploadResult.rejected.map((r, i) => (
                <li key={i} className="error-text">
                  <code>{r.name}</code>:{r.reason}
                </li>
              ))}
            </ul>
          )}
          {items.length === 0 && !uploading && (
            <p className="muted">该库还没有图片。上传后即可在每张图下方直接编辑提示词。</p>
          )}
          {/* 图片量大时可考虑分页/虚拟滚动,当前直接全量内联渲染 */}
          <div className="cap-grid">
            {items.map((it) => (
              <CaptionCard key={it.image} item={it} dataset={selected} onSaved={() => void load()} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
