"""任务生命周期管理(SQLite 持久化 + Popen 进程句柄)。

- 任务状态机:pending | running | exited_ok | exited_fail | killed
- 长任务一律 Popen 参数数组异步执行,HTTP 请求不持有进程;
- 依赖排队:start(..., after=<task_id>) 落库为 pending,前置任务 exited_ok
  且 GPU 空闲时自动升格启动;前置失败/被杀则级联标 exited_fail(取消链);
- 每个任务的 stdout/stderr 合并落盘 ui/state/logs/<task_id>.log;
- 后端重启后按 pid 存活检测恢复跟踪(句柄丢失则降级为"仅观察"),
  并推进未完成的依赖链;
- terminate 平台分支:Windows 发 CTRL_BREAK_EVENT(需 CREATE_NEW_PROCESS_GROUP),
  POSIX 发 SIGTERM;超时未退出则 kill;pending 任务直接标记 killed 并取消链。
"""

from __future__ import annotations

import ctypes
import json
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from adapter import engine_env

# 状态常量
PENDING, RUNNING = "pending", "running"
EXITED_OK, EXITED_FAIL, KILLED = "exited_ok", "exited_fail", "killed"

# GPU 独占任务全集(M3):同一时刻只允许其中一个在跑。
# 导出脚本默认 DEVICE=cpu,但可能加载基模占大内存,与训练并发会互相挤爆,
# 故与训练/缓存同按独占语义互斥。
GPU_EXCLUSIVE_KINDS = ("train", "cache_text", "cache_latents",
                       "sample", "grid", "export")


def gpu_busy(tasks: list[dict]) -> dict | None:
    """从任务列表(任意顺序)找出正在运行的 GPU 独占任务;无则 None。"""
    for t in tasks:
        if t["status"] == RUNNING and t["kind"] in GPU_EXCLUSIVE_KINDS:
            return t
    return None

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,            -- train | cache_text | cache_latents | sample | export
    status      TEXT NOT NULL,
    config_path TEXT,                     -- 关联的引擎配置文件(M0 可为空)
    run_dir     TEXT,                     -- runs/moss/<RUN_NAME>(训练开始后才创建)
    pid         INTEGER,
    started_at  TEXT,
    finished_at TEXT,
    exit_code   INTEGER,
    overrides   TEXT,                     -- JSON:--set 覆盖项
    argv        TEXT,                     -- JSON:完整参数数组(resume 复用)
    log_path    TEXT,                     -- ui/state/logs/<task_id>.log
    after_task  TEXT                      -- 依赖排队:前置任务 id(pending 期间等待其成功)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pid_alive(pid: int) -> bool:
    """跨平台 pid 存活检测。

    Windows 不能用 os.kill(pid, 0)(会把 signal 当退出码终止进程!),
    改用 kernel32.OpenProcess + GetExitCodeProcess。
    """
    if pid is None or pid <= 0:
        return False
    if os.name == "nt":
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return code.value == STILL_ACTIVE
            return True
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


class TaskManager:
    """任务队列与进程句柄管理;同一进程内单例使用。"""

    def __init__(self, state_dir: str | os.PathLike | None = None) -> None:
        # 默认 ui/state;测试可注入 tmp 目录
        self.state_dir = Path(state_dir) if state_dir else Path(__file__).resolve().parents[1] / "state"
        self.log_dir = self.state_dir / "logs"
        self.db_path = self.state_dir / "tasks.db"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._advance_lock = threading.Lock()          # 串行化依赖推进(防并发双启动)
        self._procs: dict[str, subprocess.Popen] = {}   # task_id -> 进程句柄(重启后为空)
        self._killing: set[str] = set()                  # 用户主动 terminate 的任务
        self._init_db()
        self._recover()

    # ---------- SQLite ----------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            # 旧库无 after_task 列时补列(幂等;新库 CREATE 已含该列,ALTER 报重复列即忽略)
            cols = {r[1] for r in conn.execute("PRAGMA table_info(tasks)")}
            if "after_task" not in cols:
                conn.execute("ALTER TABLE tasks ADD COLUMN after_task TEXT")

    def _update(self, task_id: str, **fields) -> None:
        fields["status"] = fields.get("status")  # 保持可读性
        keys = ", ".join(f"{k} = ?" for k in fields)
        with self._lock, self._connect() as conn:
            conn.execute(f"UPDATE tasks SET {keys} WHERE id = ?", (*fields.values(), task_id))

    def _row_to_dict(self, row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        d = dict(row)
        d["overrides"] = json.loads(d["overrides"]) if d.get("overrides") else {}
        d["argv"] = json.loads(d["argv"]) if d.get("argv") else []
        return d

    # ---------- 重启恢复 ----------

    def _recover(self) -> None:
        """后端重启后:running 行按 pid 存活检测。
        进程已死 -> 标记 exited_fail(退出码未知,进程早期崩溃靠日志兜底);
        进程仍活但句柄丢失 -> 保持 running,降级为"仅观察产物"。
        最后推进一次依赖链(pending 行可能因重启错过升格时机)。
        """
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM tasks WHERE status = ?", (RUNNING,)).fetchall()
        for row in rows:
            if not _pid_alive(row["pid"] or -1):
                self._update(row["id"], status=EXITED_FAIL, finished_at=_now(), exit_code=None)
        self._advance()

    # ---------- 操作 ----------

    def start(self, kind: str, argv: list[str], config_path: str | None = None,
              run_dir: str | None = None, overrides: dict | None = None,
              after: str | None = None) -> dict:
        """以参数数组启动任务进程,立即返回任务记录(不等待退出)。

        after=<task_id>:依赖排队,先落库为 pending,由 _advance 在前置任务
        exited_ok 且 GPU 空闲时升格启动(缓存文本->缓存图像->训练 的自动链)。
        """
        task_id = uuid.uuid4().hex
        log_path = self.log_dir / f"{task_id}.log"
        argv = [str(a) for a in argv]
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO tasks (id, kind, status, config_path, run_dir, pid, started_at,"
                " exit_code, overrides, argv, log_path, after_task) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (task_id, kind, PENDING if after else RUNNING, config_path, run_dir,
                 None, _now(), None,
                 json.dumps(overrides or {}), json.dumps(argv), str(log_path), after),
            )
        if after:
            # 竞态兜底:入队瞬间前置可能已经成功(快任务),立即推进一次
            self._advance()
            return self.get(task_id)
        return self._spawn(task_id)

    def _spawn(self, task_id: str) -> dict:
        """启动已落库行的进程(直接启动与依赖升格共用);失败标 exited_fail 并抛出。"""
        row = self.get(task_id)
        argv = row["argv"]
        logf = open(row["log_path"], "w", encoding="utf-8", errors="replace")
        popen_kwargs: dict = dict(
            stdout=logf,
            stderr=subprocess.STDOUT,
            cwd=str(Path(__file__).resolve().parents[2]),  # 仓库根
            env=engine_env(Path(__file__).resolve().parents[2]),
            shell=False,
        )
        if os.name == "nt":
            # 新建进程组,才能用 CTRL_BREAK_EVENT 优雅终止
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        try:
            proc = subprocess.Popen(argv, **popen_kwargs)
        except OSError:
            logf.close()
            self._update(task_id, status=EXITED_FAIL, finished_at=_now())
            raise
        self._update(task_id, status=RUNNING, pid=proc.pid, started_at=_now(),
                     finished_at=None, exit_code=None)
        self._procs[task_id] = proc
        threading.Thread(target=self._monitor, args=(task_id, proc, logf),
                         name=f"task-monitor-{task_id[:8]}", daemon=True).start()
        return self.get(task_id)

    def _advance(self) -> None:
        """依赖推进:前置成功且 GPU 空闲 -> 升格启动 pending;
        前置失败/被取消/丢失 -> 级联标 exited_fail。每个任务终态后与重启时调用。
        """
        with self._advance_lock:
            for _ in range(100):  # 有界轮次:链式升格一轮推进一步,防依赖成环时死循环
                with self._connect() as conn:
                    rows = conn.execute("SELECT * FROM tasks ORDER BY rowid").fetchall()
                by_id = {r["id"]: r for r in rows}
                pending = [r for r in rows if r["status"] == PENDING]
                if not pending:
                    return
                gpu_busy_now = any(r["status"] == RUNNING and r["kind"] in GPU_EXCLUSIVE_KINDS
                                   for r in rows)
                changed = False
                for r in pending:
                    dep = by_id.get(r["after_task"]) if r["after_task"] else None
                    if dep is None:
                        # 前置行丢失(数据异常):不悬挂等待,直接判失败
                        self._update(r["id"], status=EXITED_FAIL, finished_at=_now())
                        changed = True
                        continue
                    if dep["status"] in (EXITED_FAIL, KILLED):
                        self._update(r["id"], status=EXITED_FAIL, finished_at=_now())
                        changed = True
                        continue
                    if dep["status"] == EXITED_OK:
                        if gpu_busy_now:
                            continue  # GPU 被其它任务占用:等它退出时再推进
                        try:
                            self._spawn(r["id"])
                        except OSError:
                            pass  # _spawn 已标 exited_fail,下轮级联其后继
                        gpu_busy_now = True
                        changed = True
                if not changed:
                    return

    def _monitor(self, task_id: str, proc: subprocess.Popen, logf) -> None:
        """后台线程:等待进程退出并落库最终状态,随后推进依赖链。"""
        code = proc.wait()
        logf.close()
        self._procs.pop(task_id, None)
        with self._lock:
            killed = task_id in self._killing
            self._killing.discard(task_id)
        # terminate() 已写 killed 时不覆盖
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()
            if row and row["status"] == KILLED:
                self._advance()
                return
        status = KILLED if killed else (EXITED_OK if code == 0 else EXITED_FAIL)
        self._update(task_id, status=status, exit_code=code, finished_at=_now())
        self._advance()

    def terminate(self, task_id: str, timeout: float = 15.0) -> dict:
        """停止任务:pending 直接取消(级联取消其后继链);
        Windows 发 CTRL_BREAK_EVENT / POSIX 发 SIGTERM,超时 kill。"""
        proc = self._procs.get(task_id)
        if proc is None:
            # 句柄丢失(后端重启后):POSIX 可按 pid 发 SIGTERM;Windows 无法向
            # 非本控制台进程组发 CTRL_BREAK,只能降级为仅观察
            row = self.get(task_id)
            if row is None:
                raise KeyError(f"任务不存在:{task_id}")
            if row["status"] == PENDING:
                # 排队中(进程从未启动):直接取消,并推进依赖链级联取消后继
                self._update(task_id, status=KILLED, finished_at=_now())
                self._advance()
                return self.get(task_id)
            if row["status"] != RUNNING:
                return row
            if os.name != "nt" and row["pid"] and _pid_alive(row["pid"]):
                os.kill(row["pid"], signal.SIGTERM)
                self._update(task_id, status=KILLED, finished_at=_now())
                self._advance()
                return self.get(task_id)
            raise RuntimeError("进程句柄已丢失(后端重启),该任务降级为仅观察,无法终止")
        with self._lock:
            self._killing.add(task_id)
        try:
            if os.name == "nt":
                os.kill(proc.pid, signal.CTRL_BREAK_EVENT)  # 命中引擎 SIGTERM 等价 handler
            else:
                proc.terminate()  # SIGTERM
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()  # 硬杀兜底
                proc.wait(timeout=5)
        finally:
            with self._connect() as conn:
                row = conn.execute("SELECT status FROM tasks WHERE id = ?", (task_id,)).fetchone()
            if row and row["status"] == RUNNING:
                self._update(task_id, status=KILLED, finished_at=_now())
        return self.get(task_id)

    def resume(self, task_id: str) -> dict:
        """"继续" = 以 --set RESUME=latest 重新启动(上游无暂停信号)。"""
        row = self.get(task_id)
        if row is None:
            raise KeyError(f"任务不存在:{task_id}")
        if row["status"] == RUNNING:
            raise RuntimeError("任务仍在运行,不能 resume")
        if row["status"] == PENDING:
            raise RuntimeError("任务仍在排队等待前置完成,不能 resume")
        argv = list(row["argv"]) + ["--set", "RESUME=latest"]
        overrides = dict(row["overrides"])
        overrides["RESUME"] = "latest"
        return self.start(row["kind"], argv, config_path=row["config_path"],
                          run_dir=row["run_dir"], overrides=overrides)

    def get(self, task_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return self._row_to_dict(row)

    def list_tasks(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM tasks ORDER BY started_at DESC").fetchall()
        return [self._row_to_dict(r) for r in rows]
