"""Integration tests for Agent Teams coordination behavior."""

from __future__ import annotations

from src.agents.file_lock import FileLockManager
from src.agents.mailbox import Mailbox
from src.agents.team_manager import TeamManager
from src.types.core import AgentDefinition, HookEvent, HookResult, TeamTask


def _agent(name: str) -> AgentDefinition:
    return AgentDefinition(
        name=name,
        description=f"teammate {name}",
        model="sonnet",
        system_prompt="team worker",
    )


def _task(
    task_id: str,
    title: str,
    assignee: str,
    dependencies: list[str] | None = None,
    files: list[str] | None = None,
) -> TeamTask:
    return TeamTask(
        id=task_id,
        title=title,
        assignee=assignee,
        status="pending",
        dependencies=dependencies or [],
        files=files or [],
    )


def test_two_teammates_shared_tasks_mailbox_and_subagent_stop(tmp_path):
    mailbox = Mailbox(base_dir=str(tmp_path / "mailboxes"))
    hook_calls: list[tuple[str, str | None]] = []

    class RecordingHookEngine:
        def fire(self, event, context):
            hook_calls.append((event.value, context.source))
            return HookResult()

    def teammate_callback(agent_def, task_title, budget):
        if agent_def.name == "worker-1":
            mailbox.send_message(
                sender="worker-1",
                recipient="worker-2",
                msg_type="finding",
                payload={"from_task": task_title},
            )
        return {
            "summary": f"{agent_def.name} completed {task_title}",
            "tokens_consumed": {"input": 120, "output": 60, "cache_read": 0},
            "tools_used": ["read_file"],
            "files_modified": ["src/example.py"],
            "exit_reason": "complete",
            "turns_used": 2,
        }

    manager = TeamManager(
        hook_engine=RecordingHookEngine(),
        teammate_callback=teammate_callback,
        session_id="sess-team",
        cwd=str(tmp_path),
    )

    lead = _agent("lead")
    teammates = [_agent("worker-1"), _agent("worker-2")]
    tasks = [
        _task("t1", "Analyze module", "worker-1", files=["src/module.py"]),
        _task("t2", "Implement fix", "worker-2", dependencies=["t1"], files=["src/module.py"]),
    ]

    result = manager.execute_team(lead, teammates, tasks)

    assert len(result.task_list) == 2
    assert all(task.status == "complete" for task in result.task_list)
    assert len(result.teammate_results) == 2
    assert {r.agent_name for r in result.teammate_results} == {"worker-1", "worker-2"}

    inbound = mailbox.read_messages("worker-2")
    assert len(inbound) == 1
    assert inbound[0].sender == "worker-1"
    assert inbound[0].type == "finding"

    subagent_stops = [h for h in hook_calls if h[0] == HookEvent.SUBAGENT_STOP.value]
    assert len(subagent_stops) == 2
    assert {source for _, source in subagent_stops} == {"worker-1", "worker-2"}


def test_file_lock_mutual_exclusion_between_teammates(tmp_path):
    lock_mgr = FileLockManager(lock_dir=str(tmp_path / "locks"))
    target_file = "src/shared_file.py"

    assert lock_mgr.acquire_lock(target_file, "worker-1") is True
    assert lock_mgr.acquire_lock(target_file, "worker-2") is False
    assert lock_mgr.release_lock(target_file, "worker-1") is True
    assert lock_mgr.acquire_lock(target_file, "worker-2") is True
