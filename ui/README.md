# KohakuMossTrainer 图形界面(UI)

给 MOSS 图像 LoRA 训练器(KohakuMossTrainer)套一层浏览器图形界面:建数据集、
填表生成训练配置、看曲线和样图、导出 LoRA、一键采样出图——全程不用敲命令行。

> 本目录(`ui/`)是独立产品层,**与上游训练引擎零耦合**(见下文承诺)。

---

## 一、怎么启动

前置(只需准备一次):

1. 仓库根 `.venv` 里已安装训练引擎(`kohakumoss-doctor` 能通过)、模型权重已就位;
2. 便携 Node 已解压到 `ui/tools/node/`(仅前端首次构建需要);
3. UI 自己的后端环境不需要手动建——启动脚本会自动创建 `ui/.venv` 并安装
   `ui/requirements.txt`。

启动:

- **Windows**:双击或在终端执行 `ui\launch.ps1`
  (默认端口 8712;被占用时脚本会清晰报错,可用 `ui\launch.ps1 -Port 8713` 换端口)
- **Linux/macOS**:`./ui/launch.sh`(同样支持 `./ui/launch.sh 8713` 换端口;
  **本脚本在 Linux 上未实测**,逻辑与 launch.ps1 一致,遇到问题请反馈)

看到绿色提示后,浏览器打开 **http://127.0.0.1:8712** 即可。前端构建产物由
后端直接托管,单端口,无需另开 dev server。

## 二、五个页面怎么用

### 1. 环境页(`/env`)
打开先看这里。显示引擎体检报告(doctor):GPU 型号显存、缺什么依赖(红色高亮,
可复制修复命令)。绿灯全亮才代表引擎就绪,后面页面才能干活。

### 2. 数据集页(`/datasets`)
建库 → 上传图片(支持拖拽,图片+同名 .txt 描述自动配对)→ 在线编辑 caption。
数据集存在 `ui/state/datasets/` 里,不会污染上游仓库目录。

### 3. 配置页(`/config`)
填表向导:选权重、选数据集(可多选)、选算法(lora/loha/lokr/oft/dora/full)、
步数(预填引擎推荐值 4000)、采样提示词、触发词、显存档位(8GB 卡选 8)、精度。
"常用参数"区直接暴露主流训练器高频调节项(学习率、rank、alpha、优化器、
调度、批大小、梯度累积、分辨率面积、dropout、保存/采样间隔等),默认值实时解析
自运行引擎的真实配置资产,悬停标签看说明,只改动想改的项。"高级设置"面板列出
其余旋钮(按 LoRA 结构/优化与调度/采样与验证/数据与缓存/显存与并行/导出与日志
六组折叠,可一键 ↺ 恢复默认);其余旋钮收进"专家"组(只读清单 + 自由键值)。
改动项随配置名持久化到
`ui/state/overrides/<名字>.json`,训练启动时自动经 `kogine run --set` 应用。
点"生成"后引擎写出 `configs/user/<名字>.py`,页面同时展示配置原文和
`config check` 校验结果——和手敲 CLI 生成的文件完全一致。

### 4. 训练页(`/train`)
先做数据检查(inspect)和文本/latent 缓存,再点"开始训练"。训练中实时看:
进度条、Loss 曲线、验证曲线、日志终端(错误行高亮)、样图墙。支持优雅停止
(引擎收尾保存 checkpoint)和断点继续(RESUME=latest)。后端重启也能重新附着
到正在跑的训练。

### 5. 导出页(`/export`)
历史任务与 run 列表、checkpoint 浏览(四件套完整性角标)、一键导出
(comfy/lycoris/hf/merged/diffusers 五种格式)、采样出图图库、多 checkpoint
对比网格(compare_grid)。

## 三、架构(文字版)

```
浏览器(React+TS,ui/frontend,构建产物 dist/ 由后端托管)
        │  HTTP + WebSocket(127.0.0.1:8712)
        ▼
FastAPI 后端(ui/backend,独立 venv:ui/.venv)
        │  只有三类契约,绝不 import 引擎代码:
        │  ① CLI 子进程:kogine / kohakumoss-config / kohakumoss-doctor
        │  ② 产物文件:runs/moss/<run>/{metrics.jsonl,log.txt,config.json,
        │              checkpoints/step_*,samples,export}
        │  ③ stdout/stderr 解析
        ▼
上游训练引擎(仓库根 .venv + src/ + scripts/,UI 从不修改)
```

UI 自身的状态(SQLite 任务表 `ui/state/tasks.db`、任务日志、数据集库、采样图库)
全部放在 `ui/state/`,已在 `.gitignore` 内。

## 四、"与上游零耦合"承诺

含义:**上游引擎仓库的任何更新都能直接 merge 进来,UI 不会造成冲突,也不会被
上游改动悄悄破坏。** 具体保证:

- `ui/` 之外的文件,UI 一个都不改(引擎 `src/`、`configs/`、`scripts/` 全只读);
- UI 后端从不 `import` 任何引擎 Python 模块,只通过 CLI/文件/stdout 交互;
- 引擎清单(算法、显存档位、导出格式)运行时探测,不硬编码复制;
- 上述约束有静态审计测试防回归(`ui/tests/test_platform_audit.py`)。

升级引擎后只需重跑 `ui\.venv\Scripts\python.exe -m pytest ui\tests -q`
验证契约仍然吻合。

## 五、故障排查

**Q:Windows 上采样直接崩溃(access violation / 进程硬退出)?**
A:这是引擎侧在 Windows + torch 2.14.1 + safetensors 0.8.0 上的已知问题:
DiT 权重流式加载时文件句柄跨文件 mmap 交叉读会硬崩。**UI 已自动绕开**
(Windows 上采样强制 DiT resident + 文本编码器放 CPU),正常情况不会再遇到。
若仍崩溃,走**离线缓存路线**:先在训练页把文本缓存和 latent 缓存都跑完
(缓存命中后采样不再在线编码文本),再采样。Linux 无此问题,保持引擎原生行为。

**Q:8GB 显存(RTX 4060 Laptop 等)怎么设?**
A:配置页显存选 `8` 档即可(引擎按算法自动选 8g_full 层)。训练冒烟建议
高级参数 `STEPS` 先给 30 左右;**步数太小会被引擎 warmup 校验拒绝**(生成期
报错,属正常保护,加大步数即可)。训练/缓存/采样/导出/网格共用一把"GPU 独占锁",
同一时刻只能跑一个,页面会提示谁在占用。

**Q:训练结束已经自动导出了,UI 导出页再导出一次会怎样?**
A:引擎在训练正常结束时会把最后一个 checkpoint 自动导出 comfy+lycoris 到
`runs/moss/<run>/export/`。UI 导出走同一条引擎导出链路,产物命名与引擎一致
(`<run>-<格式>.safetensors`),但落在对应 checkpoint 的
`checkpoints/step_*/export/` 子目录,可以选择任意中间 checkpoint 与格式;
同名文件会被覆写(这是 UI 导出,以你点的格式为准)。中断的 run 没有自动导出,
必须用 UI 导出。

**Q:端口 8712 被占用?**
A:多半是已有 UI 后端在跑。关掉旧窗口,或 `-Port 8713` 换端口启动。

**Q:采样图库/数据集在哪?**
A:`ui/state/samples/`、`ui/state/grids/`、`ui/state/datasets/`;训练产物在
上游惯例位置 `runs/moss/<RUN_NAME>/`。checkpoint 目录四件套:
`trainable.safetensors`(导出/采样必需)、`optim.pt` + `state.json`(续训必需)、
`ema.safetensors`(仅启用 EMA 时存在)。

## 六、开发自检(可选)

```powershell
# 后端测试(167 项,含 M4 契约/审计/高级设置默认值测试;不碰 GPU、不真跑引擎训练)
cd <仓库根>; ui\.venv\Scripts\python.exe -m pytest ui\tests -q
# 前端构建
$env:PATH = "<仓库根>\ui\tools\node;" + $env:PATH
cd ui\frontend; npm run build
```
