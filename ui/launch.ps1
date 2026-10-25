# KohakuMossTrainer UI 一键启动(M4 最终形态,Windows)。
#
# 用法:
#   ui\launch.ps1                 # 默认端口 8712
#   ui\launch.ps1 -Port 8713      # 指定端口(自测/避让占用)
#   $env:KOHAKU_UI_PORT=8713; ui\launch.ps1
#
# 流程:检查/创建 ui\.venv(缺依赖时自动 pip 安装 requirements)->
#       前端 dist 缺失时用便携 Node(ui\tools\node)构建 ->
#       uvicorn 绑定 127.0.0.1:<端口>(端口被占用时清晰报错,不崩)。
param([int]$Port = 0)
$ErrorActionPreference = "Stop"

if ($Port -le 0) {
    if ($env:KOHAKU_UI_PORT) { $Port = [int]$env:KOHAKU_UI_PORT } else { $Port = 8712 }
}

$uiRoot = $PSScriptRoot
$py = Join-Path $uiRoot ".venv\Scripts\python.exe"
$req = Join-Path $uiRoot "requirements.txt"
$frontend = Join-Path $uiRoot "frontend"
$dist = Join-Path $frontend "dist"
$nodeDir = Join-Path $uiRoot "tools\node"

# ---------- 1. UI 后端虚拟环境 ----------
if (-not (Test-Path $py)) {
    Write-Host "[launch] 未找到 UI 后端环境,尝试创建 $uiRoot\.venv ..." -ForegroundColor Yellow
    $uv = Get-Command uv -ErrorAction SilentlyContinue
    if ($uv) {
        & uv venv (Join-Path $uiRoot ".venv")
        if ($LASTEXITCODE -ne 0) { Write-Error "uv venv 创建失败"; exit 1 }
    } else {
        $syspy = Get-Command python -ErrorAction SilentlyContinue
        if (-not $syspy) { $syspy = Get-Command py -ErrorAction SilentlyContinue }
        if (-not $syspy) {
            Write-Error "未找到 UI 环境且系统无 python/uv。请安装 uv 或 Python 3.10+ 后重试。"
            exit 1
        }
        & $syspy.Source -m venv (Join-Path $uiRoot ".venv")
        if ($LASTEXITCODE -ne 0) { Write-Error "python -m venv 创建失败"; exit 1 }
    }
}

# 依赖完整性检查(探测关键 import,缺失才安装,不每次跑 pip)。
# 注意:pynvml 会向 stderr 打 FutureWarning,PS5 在 EAP=Stop + 2>$null 下
# 会把 stderr 输出当异常抛出导致误判"依赖缺失",故探测段临时降 EAP。
$depsOk = $false
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = "SilentlyContinue"
& $py -W ignore -c "import fastapi, uvicorn, multipart, websockets, pynvml, watchdog, httpx" 2>$null
$depsOk = ($LASTEXITCODE -eq 0)
$ErrorActionPreference = $prevEAP
if (-not $depsOk) {
    Write-Host "[launch] 安装 UI 后端依赖($req)..." -ForegroundColor Yellow
    $uv = Get-Command uv -ErrorAction SilentlyContinue
    if ($uv) {
        & uv pip install -r $req --python $py
    } else {
        $ErrorActionPreference = "SilentlyContinue"
        & $py -m pip --version 2>$null
        $hasPip = ($LASTEXITCODE -eq 0)
        $ErrorActionPreference = $prevEAP
        if (-not $hasPip) { & $py -m ensurepip --upgrade | Out-Null }
        & $py -m pip install -r $req
    }
    if ($LASTEXITCODE -ne 0) { Write-Error "依赖安装失败,请检查网络或手动执行 pip install -r ui\requirements.txt"; exit 1 }
}

# ---------- 2. 前端构建产物 ----------
if (-not (Test-Path (Join-Path $dist "index.html"))) {
    Write-Host "[launch] 前端 dist 缺失,用便携 Node 构建 ..." -ForegroundColor Yellow
    $npm = Join-Path $nodeDir "npm.cmd"
    if (-not (Test-Path $npm)) {
        Write-Error "未找到便携 Node($npm)。请先按计划阶段 0 下载便携 Node 解压到 ui\tools\node\,或手动在 ui\frontend 执行 npm run build。"
        exit 1
    }
    $oldPath = $env:PATH
    $env:PATH = "$nodeDir;$oldPath"
    try {
        Push-Location $frontend
        & $npm run build
        $buildExit = $LASTEXITCODE
    } finally {
        Pop-Location
        $env:PATH = $oldPath
    }
    if ($buildExit -ne 0 -or -not (Test-Path (Join-Path $dist "index.html"))) {
        Write-Error "前端构建失败(npm run build,退出码 $buildExit)"
        exit 1
    }
}

# ---------- 3. 端口占用检查(清晰报错,不崩) ----------
$listener = $null
try {
    $listener = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Port)
    $listener.Start()
} catch {
    Write-Error "端口 $Port 已被占用(可能已有 UI 后端在运行)。请关闭旧实例,或用 ui\launch.ps1 -Port <其他端口> 启动。"
    exit 1
} finally {
    if ($listener) { $listener.Stop() }
}

# ---------- 4. 启动 ----------
Write-Host "KohakuMossTrainer UI 启动中…… 请在浏览器打开 http://127.0.0.1:$Port" -ForegroundColor Green
& $py -m uvicorn app:app --app-dir (Join-Path $uiRoot "backend") --host 127.0.0.1 --port $Port
