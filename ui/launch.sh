#!/usr/bin/env bash
# KohakuMossTrainer UI 一键启动(M4 最终形态,Linux/macOS)。
#
# 用法:
#   ./ui/launch.sh              # 默认端口 8712
#   ./ui/launch.sh 8713         # 指定端口(自测/避让占用)
#   KOHAKU_UI_PORT=8713 ./ui/launch.sh
#
# 流程与 launch.ps1 一致:检查/创建 ui/.venv(缺依赖时安装 requirements)->
# 前端 dist 缺失时用便携 Node(ui/tools/node)构建 -> uvicorn 绑 127.0.0.1:<端口>。
# 注意:本脚本按 launch.ps1 同一契约编写,Linux 路径分支未经实机验证。
set -euo pipefail

UI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${1:-${KOHAKU_UI_PORT:-8712}}"
PY="$UI_ROOT/.venv/bin/python"
REQ="$UI_ROOT/requirements.txt"
FRONTEND="$UI_ROOT/frontend"
DIST="$FRONTEND/dist"
NODE_DIR="$UI_ROOT/tools/node"

# ---------- 1. UI 后端虚拟环境 ----------
if [ ! -x "$PY" ]; then
    echo "[launch] 未找到 UI 后端环境,尝试创建 $UI_ROOT/.venv ..." >&2
    if command -v uv >/dev/null 2>&1; then
        uv venv "$UI_ROOT/.venv"
    elif command -v python3 >/dev/null 2>&1; then
        python3 -m venv "$UI_ROOT/.venv"
    else
        echo "错误:未找到 UI 环境且系统无 python3/uv。请安装 uv 或 Python 3.10+ 后重试。" >&2
        exit 1
    fi
fi

# 依赖完整性检查(探测关键 import,缺失才安装)
if ! "$PY" -c "import fastapi, uvicorn, multipart, websockets, pynvml, watchdog, httpx" 2>/dev/null; then
    echo "[launch] 安装 UI 后端依赖($REQ)..." >&2
    if command -v uv >/dev/null 2>&1; then
        uv pip install -r "$REQ" --python "$PY"
    else
        "$PY" -m pip --version >/dev/null 2>&1 || "$PY" -m ensurepip --upgrade >/dev/null
        "$PY" -m pip install -r "$REQ"
    fi
fi

# ---------- 2. 前端构建产物 ----------
if [ ! -f "$DIST/index.html" ]; then
    echo "[launch] 前端 dist 缺失,用便携 Node 构建 ..." >&2
    if [ ! -x "$NODE_DIR/npm" ]; then
        echo "错误:未找到便携 Node($NODE_DIR/npm)。请先下载便携 Node 解压到 ui/tools/node/,或手动在 ui/frontend 执行 npm run build。" >&2
        exit 1
    fi
    (cd "$FRONTEND" && PATH="$NODE_DIR:$PATH" "$NODE_DIR/npm" run build)
    if [ ! -f "$DIST/index.html" ]; then
        echo "错误:前端构建失败(npm run build)" >&2
        exit 1
    fi
fi

# ---------- 3. 端口占用检查(清晰报错,不崩) ----------
if "$PY" - "$PORT" <<'EOF' 2>/dev/null
import socket, sys
s = socket.socket()
try:
    s.bind(("127.0.0.1", int(sys.argv[1])))
except OSError:
    sys.exit(1)
finally:
    s.close()
EOF
then :; else
    echo "错误:端口 $PORT 已被占用(可能已有 UI 后端在运行)。请关闭旧实例,或用 ./ui/launch.sh <其他端口> 启动。" >&2
    exit 1
fi

# ---------- 4. 启动 ----------
echo "KohakuMossTrainer UI 启动中…… 请在浏览器打开 http://127.0.0.1:$PORT"
exec "$PY" -m uvicorn app:app --app-dir "$UI_ROOT/backend" --host 127.0.0.1 --port "$PORT"
