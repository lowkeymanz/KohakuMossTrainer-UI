# 安装说明

本文档描述如何在 Windows 与 Linux 上安装本仓库。请在仓库根目录（即包含 `ui/`、`scripts/`、`pyproject.toml` 的目录）下完成全部步骤，无需从上游重新 clone。

## 前置要求

- **uv**（安装脚本优先使用它创建环境；未安装时脚本会回退到 py launcher）
- **NVIDIA 驱动**：`nvidia-smi` 显示的 CUDA Version ≥ 12.6
- 磁盘空间：安装环境依赖（CUDA 版 torch 等）约需 10GB 可用空间；数据集缓存与训练输出按需预留
- 模型权重：需将以下三个文件放入仓库根目录的 `weights/models/`（ComfyUI 布局），配置页会自动检测填入：

  ```text
  weights/models/
  ├── diffusion_models/moss_image_base_int8_convrot.safetensors
  ├── text_encoders/qwen3.5_9b_int8_convrot.safetensors
  └── vae/flux2-vae_small.safetensors
  ```

## Windows 安装

在仓库根目录打开 PowerShell：

```powershell
# 1. 官方安装脚本：自动创建 .venv、按驱动安装对应 CUDA 版 torch、
#    以 editable 方式从仓库源码安装引擎（见下方"依赖版本注意事项"），
#    结束时自动运行一次环境自检 kohakumoss-doctor（应全部通过；含内核编译探测，冷缓存时耗时数分钟属正常）
powershell -ExecutionPolicy Bypass -File scripts\install\install.ps1

# 2. 钉住 kohakuvault 版本（必做，见下方"依赖版本注意事项"）
uv pip install "kohakuvault==0.9.0"

# 3. 允许激活脚本运行（每用户执行一次）
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# 4. 激活环境
.venv\Scripts\Activate.ps1
```

## Linux 安装

在仓库根目录打开终端：

```bash
# 1. 官方安装脚本（结束时自动运行一次环境自检 kohakumoss-doctor，应全部通过）
bash scripts/install/install.sh

# 2. 钉住 kohakuvault 版本（必做，见下方"依赖版本注意事项"）
uv pip install "kohakuvault==0.9.0"

# 3. 激活环境
source .venv/bin/activate
```

## 依赖版本注意事项

- **从仓库源码安装，不要从 PyPI 安装**：PyPI 上的 `kohakumoss` 包与本仓库 `scripts/`、`configs/` 的版本不一致时，缓存脚本与训练配置会报"未知参数"类错误。安装脚本检测到仓库根目录存在 `pyproject.toml` 时会自动改用 editable 安装（指向仓库 `src/`），保证环境与代码一致。请勿手动执行上游 README "By hand" 一节中的 `uv pip install kohakumoss[cuda]`。
- **`kohakuvault` 必须钉住 0.9.0**：仓库 `pyproject.toml` 中该依赖没有版本上限，而 1.0.0 及以上版本修改了底层接口，不钉版本会导致训练启动时报 `AttributeError: '_ColumnVault' object has no attribute 'read_range'`。Windows 与 Linux 均需执行此步骤；待依赖声明修复后可移除。

## 可选：调大虚拟内存（Windows，物理内存 32GB 以下的机器）

训练中输出预览图对内存需求更高，物理内存 32GB 以下的机器建议调大虚拟内存：

1. `Win+R` → `sysdm.cpl` → 高级 → 性能"设置" → 高级 → 虚拟内存"更改"
2. 取消"自动管理所有驱动器的分页文件大小"，选中系统盘 → 自定义大小（内存越小，需要设置得越大）
3. 输入数值后点击"设置"按钮再点确定，重启电脑生效

不配置不影响训练本身，仅训练中途不输出预览图（可将预览间隔设为 0，在训练完成后验证效果）。

## 启动 UI 服务

安装完成后直接运行启动脚本即可。UI 使用独立的 `ui/.venv` 后端环境，启动脚本会自动创建并安装依赖，无需手动激活上面安装的引擎环境：

- **Windows**：`ui\launch.ps1`（默认端口 8712，被占用时换端口：`ui\launch.ps1 -Port 8713`）
- **Linux**：`./ui/launch.sh`（换端口：`./ui/launch.sh 8713`）

首次启动需要构建前端产物：脚本优先使用解压在 `ui/tools/node/` 的便携 Node；若没有，也可在系统装有 Node.js 的情况下手动在 `ui/frontend` 执行一次 `npm run build`，之后无需重复。

启动成功后，浏览器打开 **http://127.0.0.1:8712** 即可使用。更多使用说明见 [ui/README.md](ui/README.md)。

### 远程访问（云平台等场景）

启动脚本默认只监听 `127.0.0.1`——UI 没有鉴权，这是安全默认值。需要从其他机器访问时，二选一：

**方式一：SSH 端口转发（推荐）**。在本地电脑执行（保持终端不关），然后照常访问本地地址：

```bash
ssh -L 8712:127.0.0.1:8712 -p <SSH端口> <用户>@<远程主机>
```

**方式二：直接对外监听**。绕过启动脚本，手动以 `0.0.0.0` 启动：

```bash
# Linux
ui/.venv/bin/python -m uvicorn app:app --app-dir ui/backend --host 0.0.0.0 --port 8712
```

```powershell
# Windows
ui\.venv\Scripts\python.exe -m uvicorn app:app --app-dir ui\backend --host 0.0.0.0 --port 8712
```

> ⚠️ 方式二会把无鉴权的训练控制台暴露到网络中，仅在防火墙/安全组已限制来源 IP 时使用，不用时尽快关闭。
