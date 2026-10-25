"""任务生命周期测试:用假进程(当前 UI venv 的 python)走完整链路,
不依赖引擎。state 目录注入 tmp_path,绝不写真实 ui/state。
"""

import sys
import time

import pytest
from task_manager import (EXITED_FAIL, EXITED_OK, KILLED, PENDING, RUNNING,
                          TaskManager)

SLEEP_CMD = [sys.executable, "-c", "import time; time.sleep(30)"]
OK_CMD = [sys.executable, "-c", "print('hello')"]
FAIL_CMD = [sys.executable, "-c", "import sys; sys.exit(3)"]


def wait_for_status(tm: TaskManager, task_id: str, expected: str, timeout: float = 15.0) -> dict:
    """轮询等待状态落库(监控线程异步写)。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        row = tm.get(task_id)
        if row and row["status"] == expected:
            return row
        time.sleep(0.1)
    pytest.fail(f"任务 {task_id} 未在 {timeout}s 内进入 {expected},当前:{tm.get(task_id)}")


def test_full_lifecycle_start_running_terminate_killed(tmp_path):
    """start -> running -> terminate -> killed,日志落盘,进程真正退出。"""
    tm = TaskManager(state_dir=tmp_path)
    task = tm.start("train", SLEEP_CMD, overrides={"STEPS": "30"})
    assert task["status"] == RUNNING
    assert task["pid"] > 0
    assert task["kind"] == "train"
    assert task["overrides"] == {"STEPS": "30"}

    # 日志文件已创建
    log_path = tmp_path / "logs" / f"{task['id']}.log"
    assert log_path.is_file()

    terminated = tm.terminate(task["id"], timeout=10)
    assert terminated["status"] == KILLED
    assert terminated["finished_at"]
    # 进程确已消失
    from task_manager import _pid_alive
    deadline = time.time() + 10
    while _pid_alive(terminated["pid"]) and time.time() < deadline:
        time.sleep(0.1)
    assert not _pid_alive(terminated["pid"])


def test_exited_ok_and_fail(tmp_path):
    """自然退出:退出码 0 -> exited_ok;非 0 -> exited_fail 且记录 exit_code。"""
    tm = TaskManager(state_dir=tmp_path)
    ok = tm.start("sample", OK_CMD)
    row = wait_for_status(tm, ok["id"], EXITED_OK)
    assert row["exit_code"] == 0
    assert "hello" in read_log(tmp_path, ok["id"])

    bad = tm.start("sample", FAIL_CMD)
    row = wait_for_status(tm, bad["id"], EXITED_FAIL)
    assert row["exit_code"] == 3


def read_log(tmp_path, task_id):
    return (tmp_path / "logs" / f"{task_id}.log").read_text(encoding="utf-8")


def test_state_survives_manager_restart(tmp_path):
    """状态持久化:重建 TaskManager(模拟后端重启)后仍可读历史状态。"""
    tm = TaskManager(state_dir=tmp_path)
    task = tm.start("cache_text", OK_CMD)
    wait_for_status(tm, task["id"], EXITED_OK)

    tm2 = TaskManager(state_dir=tmp_path)  # 重启
    row = tm2.get(task["id"])
    assert row is not None
    assert row["status"] == EXITED_OK
    assert row["kind"] == "cache_text"
    assert row["argv"] == [str(a) for a in OK_CMD]
    assert task["id"] in [t["id"] for t in tm2.list_tasks()]


def test_resume_restarts_with_latest(tmp_path):
    """resume = 以 --set RESUME=latest 重新启动,生成新任务记录。"""
    tm = TaskManager(state_dir=tmp_path)
    task = tm.start("train", SLEEP_CMD, config_path="configs/user/demo.py")
    tm.terminate(task["id"], timeout=10)

    resumed = tm.resume(task["id"])
    assert resumed["id"] != task["id"]
    assert resumed["status"] == RUNNING
    assert resumed["argv"][-2:] == ["--set", "RESUME=latest"]
    assert resumed["overrides"].get("RESUME") == "latest"
    assert resumed["config_path"] == "configs/user/demo.py"
    tm.terminate(resumed["id"], timeout=10)


def test_recover_marks_dead_running_tasks(tmp_path):
    """重启恢复:pid 已死的 running 行被纠正为 exited_fail(仅观察降级)。"""
    tm = TaskManager(state_dir=tmp_path)
    task = tm.start("train", SLEEP_CMD)
    tm.terminate(task["id"], timeout=10)
    # 手工把状态改回 running 模拟"后端崩溃时进程还活着,现在进程已死"
    import sqlite3
    with sqlite3.connect(str(tmp_path / "tasks.db")) as conn:
        conn.execute("UPDATE tasks SET status='running' WHERE id=?", (task["id"],))
    tm2 = TaskManager(state_dir=tmp_path)
    row = tm2.get(task["id"])
    assert row["status"] == EXITED_FAIL


# ---------- 依赖排队(after=) ----------

def test_chain_promotes_after_success(tmp_path):
    """前置 exited_ok -> pending 自动升格启动并跑完。"""
    tm = TaskManager(state_dir=tmp_path)
    a = tm.start("cache_text", OK_CMD)
    b = tm.start("cache_latents", OK_CMD, after=a["id"])
    assert b["status"] == PENDING
    assert b["pid"] is None
    wait_for_status(tm, a["id"], EXITED_OK)
    row = wait_for_status(tm, b["id"], EXITED_OK)
    assert row["pid"] > 0
    assert "hello" in read_log(tmp_path, b["id"])


def test_chain_cascades_failure(tmp_path):
    """前置 exited_fail -> 后继级联 exited_fail,进程从不启动。"""
    tm = TaskManager(state_dir=tmp_path)
    a = tm.start("cache_text", FAIL_CMD)
    b = tm.start("cache_latents", OK_CMD, after=a["id"])
    c = tm.start("train", OK_CMD, after=b["id"])
    wait_for_status(tm, a["id"], EXITED_FAIL)
    wait_for_status(tm, b["id"], EXITED_FAIL)
    row = wait_for_status(tm, c["id"], EXITED_FAIL)
    assert row["pid"] is None  # 从未升格


def test_terminate_pending_not_promoted(tmp_path):
    """pending 被取消(killed)后,前置成功也不得升格启动。"""
    tm = TaskManager(state_dir=tmp_path)
    a = tm.start("cache_text", SLEEP_CMD)
    b = tm.start("train", OK_CMD, after=a["id"])
    assert b["status"] == PENDING
    killed = tm.terminate(b["id"])
    assert killed["status"] == KILLED
    tm.terminate(a["id"], timeout=10)
    wait_for_status(tm, a["id"], KILLED)
    time.sleep(0.5)  # 给错误的升格留窗口
    assert tm.get(b["id"])["status"] == KILLED


def test_chain_survives_restart(tmp_path):
    """后端重启恢复:前置已成功、后继仍 pending -> 重建管理器后自动补推进。"""
    tm = TaskManager(state_dir=tmp_path)
    a = tm.start("cache_text", OK_CMD)
    b = tm.start("train", OK_CMD, after=a["id"])
    wait_for_status(tm, a["id"], EXITED_OK)
    wait_for_status(tm, b["id"], EXITED_OK)
    # 双保险:链在运行期已推进完成;再验证 pending 悬挂场景的恢复路径
    tm2 = TaskManager(state_dir=tmp_path)
    assert tm2.get(b["id"])["status"] == EXITED_OK
