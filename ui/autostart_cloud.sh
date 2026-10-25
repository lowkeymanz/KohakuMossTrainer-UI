#!/usr/bin/env bash
# KohakuMossTrainer UI 镜像自启动脚本(环境已在镜像内装好,这里只负责启动)。
# 平台约定:前台常驻进程,绑 0.0.0.0,经平台端口映射/自定义服务访问。
#
# 按镜像实际情况调整这两处即可:
REPO="${KOHAKU_REPO:-/home/KohakuMossTrainer-UI}"   # 仓库根目录(镜像实际路径)
PORT="${KOHAKU_PORT:-8712}"                       # UI 端口

[ -x "$REPO/ui/.venv/bin/python" ] || { echo "[autostart] 未找到 UI 环境: $REPO/ui/.venv" >&2; exit 1; }

exec "$REPO/ui/.venv/bin/python" -m uvicorn app:app --app-dir "$REPO/ui/backend" --host 0.0.0.0 --port "$PORT"
