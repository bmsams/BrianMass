"""Unit tests for the Team Manager (Agent Teams topology).

Covers:
- SharedTaskList: add/claim/complete/block/unblock, dependency tracking,
  file ownership mapping
- TeamManager: team execution, budget creation, teammate selection,
  hook firing, error handling
"""

from __future__ import annotations

import pytest

from src.agents.team_manager import (
    SharedTaskList,
    TeamManager,
    TeamResult,
    _default_teammate_callback,
)
from src.types.core import (
    AgentBudget,
    AgentDefinition,
    HookEvent,
    HookResult,
    TeamTask,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(**overrides) -> TeamTask:
    defaults = {
        "id": "t1",
        "title": "Do something",
        "assignee": None,
        "status": "pending",
        "dependencies": [],
        "files": [],
    }
    defaults.update(overrides)
    return TeamTask(**defaults)


def _make_agent_def(**overrides) -> AgentDefinition:
    defaults = {
        "name": "test-agent",
        "description": "A test agent",
        "model": "sonnet",
        "system_prompt": "You are a test agent.",
    }
    defaults.update(overrides)
    return AgentDefinition(**defaults)


def _make_budget(**overrides) -> AgentBudget:
    defaults = {
        "input_budget_tokens": 200_000,
        "output_budget_tokens": 50_000,
        "session_budget_usd": 5.0,
        "team_budget_usd": 5.0,
    }
    defaults.update(overrides)
    return AgentBudget(**defaults)


# ===================================================================
# SharedTaskList Tests
# ===================================================================

class TestSharedTaskListAdd:
    def test_add_task(self):
        tl = SharedTaskList()
        task = _make_task(id="t1")
        tl.add_task(task)
        assert tl.get_task("t1") is task

    def test_add_duplicate_raises(self):
        tl = SharedTaskList()
        tl.add_task(_make_task(id="t1"))
        with pytest.raises(ValueError, match="already exists"):
            tl.add_task(_make_task(id="t1"))

    def test_add_invalid_status_raises(self):
        tl = SharedTaskList()
        with pytest.raises(ValueError, match="Invalid status"):
            tl.add_task(_make_task(id="t1", status="invalid"))

    def test_get_all_tasks(self):
        tl = SharedTaskList()
        tl.add_task(_make_task(id="t1"))
        tl.add_task(_make_task(id="t2"))
        assert len(tl.get_all_tasks()) == 2

    def test_get_nonexistent_returns_none(self):
        tl = SharedTaskList()
        assert tl.get_task("nope") is None

    def test_remove_task(self):
        tl = SharedTaskList()
        tl.add_task(_make_task(id="t1"))
        tl.remove_task("t1")
        assert tl.get_task("t1") is None

    def test_remove_nonexistent_raises(self):
        tl = SharedTaskList()
        with pytest.raises(KeyError, match="not found"):
            tl.remove_task("nope")


class TestSharedTaskListStatusTransitions:
    def test_claim_pending_task(self):
        tl = SharedTaskList()
        tl.add_task(_make_task(id="t1", status="pending"))
        tl.claim_task("t1", "agent-a")
        task = tl.get_task("t1")
        assert task.status == "claimed"
        assert task.assignee == "agent-a"

    def test_claim_non_pending_raises(self):
        tl = SharedTaskList()
        tl.add_task(_make_task(id="t1", status="pending"))
        tl.claim_task("t1", "agent-a")
        with pytest.raises(ValueError, match="Cannot claim"):
            tl.claim_task("t1", "agent-b")

    def test_complete_claimed_task(self):
        tl = SharedTaskList()
        tl.add_task(_make_task(id="t1", status="pending"))
        tl.claim_task("t1", "agent-a")
        tl.complete_task("t1")
        assert tl.get_task("t1").status == "complete"

    def test_complete_non_claimed_raises(self):
        tl = SharedTaskList()
        tl.add_task(_make_task(id="t1", status="pending"))
        with pytest.raises(ValueError, match="Cannot complete"):
            tl.complete_task("t1")

    def test_block_pending_task(self):
        tl = SharedTaskList()
        tl.add_task(_make_task(id="t1", status="pending"))
        tl.block_task("t1")
        assert tl.get_task("t1").status == "blocked"

    def test_block_complete_raises(self):
        tl = SharedTaskList()
        tl.add_task(_make_task(id="t1", status="pending"))
        tl.claim_task("t1", "a")
        tl.complete_task("t1")
        with pytest.raises(ValueError, match="Cannot block"):
            tl.block_task("t1")

    def test_unblock_blocked_task(self):
        tl = SharedTaskList()
        tl.add_task(_make_task(id="t1", status="pending"))
        tl.block_task("t1")
        tl.unblock_task("t1")
        assert tl.get_task("t1").status == "pending"

    def test_unblock_non_blocked_raises(self):
        tl = SharedTaskList()
        tl.add_task(_make_task(id="t1", status="pending"))
        with pytest.raises(ValueError, match="Cannot unblock"):
            tl.unblock_task("t1")

    def test_nonexistent_task_raises_key_error(self):
        tl = SharedTaskList()
        with pytest.raises(KeyError, match="not found"):
            tl.claim_task("nope", "a")


class TestSharedTaskListDependencies:
    def test_claim_with_unmet_dependency_raises(self):
        tl = SharedTaskList()
        tl.add_task(_make_task(id="t1", status="pending"))
        tl.add_task(_make_task(id="t2", status="pending", dependencies=["t1"]))
        with pytest.raises(ValueError, match="unmet dependencies"):
            tl.claim_task("t2", "agent-a")

    def test_claim_after_dependency_complete(self):
        tl = SharedTaskList()
        tl.add_task(_make_task(id="t1", status="pending"))
        tl.add_task(_make_task(id="t2", status="pending", dependencies=["t1"]))
        tl.claim_task("t1", "agent-a")
        tl.complete_task("t1")
        # Now t2 should be claimable
        tl.claim_task("t2", "agent-b")
        assert tl.get_task("t2").status == "claimed"

    def test_get_ready_tasks_respects_dependencies(self):
        tl = SharedTaskList()
        tl.add_task(_make_task(id="t1", status="pending"))
        tl.add_task(_make_task(id="t2", status="pending", dependencies=["t1"]))
        tl.add_task(_make_task(id="t3", status="pending"))
        ready = tl.get_ready_tasks()
        ready_ids = [t.id for t in ready]
        assert "t1" in ready_ids
        assert "t3" in ready_ids
        assert "t2" not in ready_ids

    def test_completing_dependency_unblocks_blocked_task(self):
        tl = SharedTaskList()
        tl.add_task(_make_task(id="t1", status="pending"))
        tl.add_task(_make_task(id="t2", status="pending", dependencies=["t1"]))
        tl.block_task("t2")
        assert tl.get_task("t2").status == "blocked"
        # Complete t1 — t2 should auto-unblock
        tl.claim_task("t1", "a")
        tl.complete_task("t1")
        assert tl.get_task("t2").status == "pending"

    def test_unblock_with_unmet_deps_raises(self):
        tl = SharedTaskList()
        tl.add_task(_make_task(id="t1", status="pending"))
        tl.add_task(_make_task(id="t2", status="pending", dependencies=["t1"]))
        tl.block_task("t2")
        with pytest.raises(ValueError, match="unmet dependencies"):
            tl.unblock_task("t2")

    def test_are_all_complete(self):
        tl = SharedTaskList()
        tl.add_task(_make_task(id="t1", status="pending"))
        tl.add_task(_make_task(id="t2", status="pending"))
        assert not tl.are_all_complete()
        tl.claim_task("t1", "a")
        tl.complete_task("t1")
        assert not tl.are_all_complete()
        tl.claim_task("t2", "b")
        tl.complete_task("t2")
        assert tl.are_all_complete()

    def test_get_tasks_by_status(self):
        tl = SharedTaskList()
        tl.add_task(_make_task(id="t1", status="pending"))
        tl.add_task(_make_task(id="t2", status="pending"))
        tl.claim_task("t1", "a")
        assert len(tl.get_tasks_by_status("claimed")) == 1
        assert len(tl.get_tasks_by_status("pending")) == 1


class TestSharedTaskListFileOwnership:
    def test_get_file_owner(self):
        tl = SharedTaskList()
        tl.add_task(_make_task(id="t1", status="pending", files=["src/auth.py"]))
        tl.claim_task("t1", "agent-a")
        assert tl.get_file_owner("src/auth.py") == "agent-a"

    def test_get_file_owner_unclaimed_returns_none(self):
        tl = SharedTaskList()
        tl.add_task(_make_task(id="t1", status="pending", files=["src/auth.py"]))
        assert tl.get_file_owner("src/auth.py") is None

    def test_get_file_owner_unknown_file_returns_none(self):
        tl = SharedTaskList()
        assert tl.get_file_owner("nope.py") is None

    def test_get_files_for_assignee(self):
        tl = SharedTaskList()
        tl.add_task(_make_task(id="t1", status="pending", files=["a.py", "b.py"]))
        tl.add_task(_make_task(id="t2", status="pending", files=["c.py"]))
        tl.claim_task("t1", "agent-a")
        tl.claim_task("t2", "agent-a")
        files = tl.get_files_for_assignee("agent-a")
        assert set(files) == {"a.py", "b.py", "c.py"}

    def test_get_file_ownership_map(self):
        tl = SharedTaskList()
        tl.add_task(_make_task(id="t1", status="pending", files=["a.py"]))
        tl.add_task(_make_task(id="t2", status="pending", files=["b.py"]))
        tl.claim_task("t1", "agent-a")
        tl.claim_task("t2", "agent-b")
        ownership = tl.get_file_ownership_map()
        assert ownership == {"a.py": "agent-a", "b.py": "agent-b"}

    def test_first_claimed_wins_file_ownership(self):
        """When multiple tasks list the same file, the first claimed one owns it."""
        tl = SharedTaskList()
        tl.add_task(_make_task(id="t1", status="pending", files=["shared.py"]))
        tl.add_task(_make_task(id="t2", status="pending", files=["shared.py"]))
        tl.claim_task("t1", "agent-a")
        tl.claim_task("t2", "agent-b")
        assert tl.get_file_owner("shared.py") == "agent-a"


# ===================================================================
# Default Callback Tests
# ===================================================================

class TestDefaultTeammateCallback:
    def test_returns_complete_result(self):
        agent_def = _make_agent_def(name="worker")
        budget = _make_budget()
        result = _default_teammate_callback(agent_def, "do stuff", budget)
        assert result["exit_reason"] == "complete"
        assert "worker" in result["summary"]


# ===================================================================
# TeamManager Tests
# ===================================================================

class TestTeamManagerExecute:
    def test_execute_team_basic(self):
        lead = _make_agent_def(name="lead")
        teammates = [_make_agent_def(name="worker-1")]
        tasks = [_make_task(id="t1", title="Task 1", assignee="worker-1")]

        mgr = TeamManager()
        result = mgr.execute_team(lead, teammates, tasks)

        assert isinstance(result, TeamResult)
        assert result.lead_result.agent_name == "lead"
        assert len(result.teammate_results) == 1
        assert result.teammate_results[0].agent_name == "worker-1"
        # Task should be complete
        assert all(t.status == "complete" for t in result.task_list)

    def test_execute_team_multiple_tasks(self):
        lead = _make_agent_def(name="lead")
        teammates = [
            _make_agent_def(name="worker-1"),
            _make_agent_def(name="worker-2"),
        ]
        tasks = [
            _make_task(id="t1", title="Task 1", assignee="worker-1"),
            _make_task(id="t2", title="Task 2", assignee="worker-2"),
        ]

        mgr = TeamManager()
        result = mgr.execute_team(lead, teammates, tasks)

        assert len(result.teammate_results) == 2
        assert all(t.status == "complete" for t in result.task_list)

    def test_execute_team_with_dependencies(self):
        lead = _make_agent_def(name="lead")
        teammates = [_make_agent_def(name="worker-1")]
        tasks = [
            _make_task(id="t1", title="First"),
            _make_task(id="t2", title="Second", dependencies=["t1"]),
        ]

        mgr = TeamManager()
        result = mgr.execute_team(lead, teammates, tasks)

        assert len(result.teammate_results) == 2
        assert all(t.status == "complete" for t in result.task_list)
        # t1 should have been executed before t2
        assert result.teammate_results[0].summary  # t1 ran
        assert result.teammate_results[1].summary  # t2 ran after

    def test_execute_team_custom_callback(self):
        call_log = []

        def custom_callback(agent_def, task, budget):
            call_log.append((agent_def.name, task))
            return {
                "summary": f"Custom: {task}",
                "tokens_consumed": {"input": 100, "output": 50, "cache_read": 0},
                "tools_used": ["read_file"],
                "files_modified": ["out.py"],
                "exit_reason": "complete",
                "turns_used": 3,
            }

        lead = _make_agent_def(name="lead")
        teammates = [_make_agent_def(name="w1")]
        tasks = [_make_task(id="t1", title="Build it", assignee="w1")]

        mgr = TeamManager(teammate_callback=custom_callback)
        result = mgr.execute_team(lead, teammates, tasks)

        assert len(call_log) == 1
        assert call_log[0] == ("w1", "Build it")
        assert result.teammate_results[0].turns_used == 3
        assert result.total_tokens["input"] == 100
        assert result.total_cost_usd > 0.0

    def test_execute_team_callback_error_blocks_task(self):
        def failing_callback(agent_def, task, budget):
            raise RuntimeError("boom")

        lead = _make_agent_def(name="lead")
        teammates = [_make_agent_def(name="w1")]
        tasks = [_make_task(id="t1", title="Fail", assignee="w1")]

        mgr = TeamManager(teammate_callback=failing_callback)
        result = mgr.execute_team(lead, teammates, tasks)

        assert len(result.teammate_results) == 1
        assert result.teammate_results[0].exit_reason == "error"
        # Task should be blocked, not complete
        blocked = [t for t in result.task_list if t.status == "blocked"]
        assert len(blocked) == 1

    def test_execute_team_no_teammates_available(self):
        lead = _make_agent_def(name="lead")
        teammates = []  # no teammates
        tasks = [_make_task(id="t1", title="Orphan")]

        mgr = TeamManager()
        result = mgr.execute_team(lead, teammates, tasks)

        # Task can't be assigned — should remain incomplete
        assert not all(t.status == "complete" for t in result.task_list)


class TestTeamManagerBudgets:
    def test_budget_split_proportional(self):
        lead = _make_agent_def(name="lead")
        teammates = [
            _make_agent_def(name="w1"),
            _make_agent_def(name="w2"),
        ]
        team_budget = _make_budget(session_budget_usd=10.0)

        mgr = TeamManager()
        budgets = mgr._create_team_budgets(lead, teammates, team_budget)

        assert f"lead:{lead.name}" in budgets
        assert "teammate:w1" in budgets
        assert "teammate:w2" in budgets

        lead_budget = budgets[f"lead:{lead.name}"]
        assert lead_budget.session_budget_usd == pytest.approx(1.0)  # 10%

        w1_budget = budgets["teammate:w1"]
        assert w1_budget.session_budget_usd == pytest.approx(4.5)  # 45% each

    def test_budget_default_when_none(self):
        lead = _make_agent_def(name="lead")
        teammates = [_make_agent_def(name="w1")]

        mgr = TeamManager()
        budgets = mgr._create_team_budgets(lead, teammates, None)

        assert f"lead:{lead.name}" in budgets
        assert "teammate:w1" in budgets


class TestTeamManagerHooks:
    def test_fires_subagent_stop_for_each_teammate(self):
        fired_events = []

        class MockHookEngine:
            def fire(self, event, context):
                fired_events.append((event, context.source))
                return HookResult()

        lead = _make_agent_def(name="lead")
        teammates = [_make_agent_def(name="w1"), _make_agent_def(name="w2")]
        tasks = [
            _make_task(id="t1", title="T1", assignee="w1"),
            _make_task(id="t2", title="T2", assignee="w2"),
        ]

        mgr = TeamManager(hook_engine=MockHookEngine())
        mgr.execute_team(lead, teammates, tasks)

        assert len(fired_events) == 2
        assert all(e[0] == HookEvent.SUBAGENT_STOP for e in fired_events)
        sources = {e[1] for e in fired_events}
        assert sources == {"w1", "w2"}

    def test_no_hook_engine_does_not_crash(self):
        lead = _make_agent_def(name="lead")
        teammates = [_make_agent_def(name="w1")]
        tasks = [_make_task(id="t1", title="T1", assignee="w1")]

        mgr = TeamManager(hook_engine=None)
        result = mgr.execute_team(lead, teammates, tasks)
        assert all(t.status == "complete" for t in result.task_list)


class TestTeamManagerCostGovernor:
    def test_registers_agents_with_cost_governor(self):
        registered = {}

        class MockCostGovernor:
            def register_agent(self, agent_id, budget):
                registered[agent_id] = budget
            def record_usage(self, **kwargs):
                pass

        lead = _make_agent_def(name="lead")
        teammates = [_make_agent_def(name="w1")]
        tasks = [_make_task(id="t1", title="T1", assignee="w1")]

        mgr = TeamManager(cost_governor=MockCostGovernor())
        mgr.execute_team(lead, teammates, tasks)

        assert "lead:lead" in registered
        assert "teammate:w1" in registered

    def test_records_usage_with_cost_governor(self):
        usage_records = []

        class MockCostGovernor:
            def register_agent(self, agent_id, budget):
                pass
            def record_usage(self, **kwargs):
                usage_records.append(kwargs)
            def calculate_cost(self, **kwargs):
                return 0.25

        def callback_with_tokens(agent_def, task, budget):
            return {
                "summary": "done",
                "tokens_consumed": {"input": 500, "output": 200, "cache_read": 0},
                "tools_used": [],
                "files_modified": [],
                "exit_reason": "complete",
                "turns_used": 1,
            }

        lead = _make_agent_def(name="lead")
        teammates = [_make_agent_def(name="w1")]
        tasks = [_make_task(id="t1", title="T1", assignee="w1")]

        mgr = TeamManager(
            cost_governor=MockCostGovernor(),
            teammate_callback=callback_with_tokens,
        )
        mgr.execute_team(lead, teammates, tasks)

        assert len(usage_records) == 1
        assert usage_records[0]["agent_id"] == "teammate:w1"
        assert usage_records[0]["input_tokens"] == 500
        assert usage_records[0]["output_tokens"] == 200


class TestTeamManagerTeammateSelection:
    def test_assignee_hint_used(self):
        """If a task has an assignee hint matching a teammate, use it."""
        lead = _make_agent_def(name="lead")
        teammates = [
            _make_agent_def(name="w1"),
            _make_agent_def(name="w2"),
        ]
        tasks = [_make_task(id="t1", title="T1", assignee="w2")]

        call_log = []

        def tracking_callback(agent_def, task, budget):
            call_log.append(agent_def.name)
            return _default_teammate_callback(agent_def, task, budget)

        mgr = TeamManager(teammate_callback=tracking_callback)
        mgr.execute_team(lead, teammates, tasks)

        assert call_log == ["w2"]

    def test_fallback_to_first_teammate(self):
        """If no assignee hint, pick the first available teammate."""
        lead = _make_agent_def(name="lead")
        teammates = [_make_agent_def(name="w1"), _make_agent_def(name="w2")]
        tasks = [_make_task(id="t1", title="T1")]  # no assignee

        call_log = []

        def tracking_callback(agent_def, task, budget):
            call_log.append(agent_def.name)
            return _default_teammate_callback(agent_def, task, budget)

        mgr = TeamManager(teammate_callback=tracking_callback)
        mgr.execute_team(lead, teammates, tasks)

        assert call_log == ["w1"]
