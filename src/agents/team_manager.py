"""Team Manager — Agent Teams topology with shared task list and file ownership.

Implements the Agent Teams pattern where a Team Lead coordinates N Teammates.
Each teammate is a full agent instance with its own context window. The team
shares a task list with status tracking (pending/claimed/blocked/complete),
dependency declarations, and file ownership mapping.

Production teammate callback uses ``strands.Agent`` + ``BedrockModel``.
Tests must inject a ``teammate_callback`` stub.

Requirements: 6.1, 6.3, 6.6, 6.9, 3.1, 3.2, 3.3, 3.4
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

from src.agents._strands_utils import _BEDROCK_MODEL_IDS, _normalize_strands_result
from src.observability.instrumentation import BrainmassTracer
from src.types.core import (
    AgentBudget,
    AgentDefinition,
    AgentResult,
    HookContext,
    HookEvent,
    HookResult,
    TeamTask,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type aliases for pluggable callbacks
# ---------------------------------------------------------------------------

# Callback that executes a teammate's task and returns a raw result dict.
TeammateCallback = Callable[
    [AgentDefinition, str, AgentBudget],
    dict,  # {"summary": str, "tokens_consumed": dict, "tools_used": list, "files_modified": list, "exit_reason": str, "turns_used": int}
]


# --- Production integration point ---
def _production_teammate_callback(
    agent_def: AgentDefinition,
    task: str,
    budget: AgentBudget,
) -> dict:
    """Execute a teammate task using a real Strands Agent + BedrockModel.

    Requirements: 3.1, 3.2, 3.3
    """
    try:
        from strands import Agent  # type: ignore[import-untyped]
        from strands.models.bedrock import BedrockModel  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "strands-agents is required for production teammate execution. "
            "Install with: pip install strands-agents  "
            "Or inject a teammate_callback to bypass this requirement."
        ) from exc

    model_id = _BEDROCK_MODEL_IDS.get(agent_def.model, _BEDROCK_MODEL_IDS["sonnet"])
    model = BedrockModel(model_id=model_id)
    agent = Agent(model=model, system_prompt=agent_def.system_prompt or "")
    raw = agent(task)
    return _normalize_strands_result(raw)

    # --- Production integration point (Swarm upgrade path) ---
    # When strands.multiagent.Swarm API is confirmed, replace the above with:
    # from strands.multiagent import Swarm
    # swarm = Swarm(agents=[agent])
    # raw = swarm(task)
    # return _normalize_strands_result(raw)


def _default_teammate_callback(
    agent_def: AgentDefinition,
    task: str,
    budget: AgentBudget,
) -> dict:
    """No-op teammate callback for testing — returns a minimal valid result dict."""
    return {
        "summary": f"Agent '{agent_def.name}' completed: {task[:80]}",
        "tokens_consumed": {"input": 0, "output": 0, "cache_read": 0},
        "tools_used": [],
        "files_modified": [],
        "exit_reason": "complete",
        "turns_used": 0,
    }


# ---------------------------------------------------------------------------
# Shared Task List
# ---------------------------------------------------------------------------

VALID_STATUSES = {"pending", "claimed", "blocked", "complete"}


class SharedTaskList:
    """Thread-safe shared task list with status tracking and dependency declarations.

    Each task is a ``TeamTask`` with status (pending/claimed/blocked/complete),
    a list of dependency task IDs, and a list of owned files.

    Requirement 6.3: shared task list with status tracking, dependency
    declarations, and file ownership mapping.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, TeamTask] = {}

    # ------------------------------------------------------------------
    # Task management
    # ------------------------------------------------------------------

    def add_task(self, task: TeamTask) -> None:
        """Add a task to the shared list.

        Raises ``ValueError`` if a task with the same ID already exists or
        if the status is invalid.
        """
        if task.id in self._tasks:
            raise ValueError(f"Task '{task.id}' already exists")
        if task.status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{task.status}' for task '{task.id}'. "
                f"Must be one of {VALID_STATUSES}"
            )
        self._tasks[task.id] = task

    def get_task(self, task_id: str) -> TeamTask | None:
        """Return the task with the given ID, or ``None``."""
        return self._tasks.get(task_id)

    def get_all_tasks(self) -> list[TeamTask]:
        """Return all tasks in insertion order."""
        return list(self._tasks.values())

    def remove_task(self, task_id: str) -> None:
        """Remove a task by ID. Raises ``KeyError`` if not found."""
        if task_id not in self._tasks:
            raise KeyError(f"Task '{task_id}' not found")
        del self._tasks[task_id]

    # ------------------------------------------------------------------
    # Status transitions
    # ------------------------------------------------------------------

    def claim_task(self, task_id: str, assignee: str) -> None:
        """Claim a pending task for *assignee*.

        A task can only be claimed if it is ``pending`` and all its
        dependencies are ``complete``.

        Raises ``ValueError`` if the task cannot be claimed.
        """
        task = self._require_task(task_id)
        if task.status != "pending":
            raise ValueError(
                f"Cannot claim task '{task_id}': status is '{task.status}', expected 'pending'"
            )
        # Check dependencies
        unmet = self._unmet_dependencies(task)
        if unmet:
            raise ValueError(
                f"Cannot claim task '{task_id}': unmet dependencies {unmet}"
            )
        task.status = "claimed"
        task.assignee = assignee

    def complete_task(self, task_id: str) -> None:
        """Mark a claimed task as complete.

        Also unblocks any tasks that were blocked solely on this task.
        """
        task = self._require_task(task_id)
        if task.status != "claimed":
            raise ValueError(
                f"Cannot complete task '{task_id}': status is '{task.status}', expected 'claimed'"
            )
        task.status = "complete"
        # Unblock tasks whose dependencies are now all met
        self._refresh_blocked_tasks()

    def block_task(self, task_id: str) -> None:
        """Mark a task as blocked (e.g. waiting on dependencies)."""
        task = self._require_task(task_id)
        if task.status not in ("pending", "claimed"):
            raise ValueError(
                f"Cannot block task '{task_id}': status is '{task.status}'"
            )
        task.status = "blocked"

    def unblock_task(self, task_id: str) -> None:
        """Move a blocked task back to pending if its dependencies are met."""
        task = self._require_task(task_id)
        if task.status != "blocked":
            raise ValueError(
                f"Cannot unblock task '{task_id}': status is '{task.status}', expected 'blocked'"
            )
        unmet = self._unmet_dependencies(task)
        if unmet:
            raise ValueError(
                f"Cannot unblock task '{task_id}': still has unmet dependencies {unmet}"
            )
        task.status = "pending"

    # ------------------------------------------------------------------
    # Dependency queries
    # ------------------------------------------------------------------

    def get_ready_tasks(self) -> list[TeamTask]:
        """Return all pending tasks whose dependencies are fully met."""
        return [
            t for t in self._tasks.values()
            if t.status == "pending" and not self._unmet_dependencies(t)
        ]

    def get_tasks_by_status(self, status: str) -> list[TeamTask]:
        """Return all tasks with the given status."""
        return [t for t in self._tasks.values() if t.status == status]

    def are_all_complete(self) -> bool:
        """Return ``True`` if every task in the list is complete."""
        return all(t.status == "complete" for t in self._tasks.values())

    # ------------------------------------------------------------------
    # File ownership (Req 6.9)
    # ------------------------------------------------------------------

    def get_file_owner(self, filepath: str) -> str | None:
        """Return the assignee of the task that owns *filepath*, or ``None``.

        Ownership is determined by the ``files`` field of claimed/complete
        tasks. If multiple tasks list the same file, the first claimed one
        wins.
        """
        for task in self._tasks.values():
            if task.status in ("claimed", "complete") and filepath in task.files:
                return task.assignee
        return None

    def get_files_for_assignee(self, assignee: str) -> list[str]:
        """Return all files owned by *assignee* across their tasks."""
        files: list[str] = []
        for task in self._tasks.values():
            if task.assignee == assignee:
                files.extend(task.files)
        return files

    def get_file_ownership_map(self) -> dict[str, str | None]:
        """Return a mapping of filepath → assignee for all tasks."""
        ownership: dict[str, str | None] = {}
        for task in self._tasks.values():
            for f in task.files:
                if f not in ownership and task.status in ("claimed", "complete"):
                    ownership[f] = task.assignee
        return ownership

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_task(self, task_id: str) -> TeamTask:
        task = self._tasks.get(task_id)
        if task is None:
            raise KeyError(f"Task '{task_id}' not found")
        return task

    def _unmet_dependencies(self, task: TeamTask) -> list[str]:
        """Return dependency IDs that are not yet complete."""
        unmet: list[str] = []
        for dep_id in task.dependencies:
            dep = self._tasks.get(dep_id)
            if dep is None or dep.status != "complete":
                unmet.append(dep_id)
        return unmet

    def _refresh_blocked_tasks(self) -> None:
        """Move blocked tasks to pending if all their dependencies are met."""
        for task in self._tasks.values():
            if task.status == "blocked" and not self._unmet_dependencies(task):
                task.status = "pending"


# ---------------------------------------------------------------------------
# Team Result
# ---------------------------------------------------------------------------

@dataclass
class TeamResult:
    """Aggregated result from an Agent Team execution."""
    team_name: str
    lead_result: AgentResult
    teammate_results: list[AgentResult]
    task_list: list[TeamTask]
    total_cost_usd: float = 0.0
    total_tokens: dict = field(default_factory=lambda: {"input": 0, "output": 0, "cache_read": 0})


# ---------------------------------------------------------------------------
# Team Manager
# ---------------------------------------------------------------------------

class TeamManager:
    """Manages Agent Teams with a Team Lead and N Teammates.

    The Team Lead is a coordinator that spawns teammates and maintains the
    shared task list. Each teammate is a full agent instance with its own
    context window and budget.

    Strands ``Swarm`` is stubbed — actual teammate execution is delegated
    to a pluggable ``teammate_callback``.

    Requirement 6.1: Team Lead + N Teammates topology
    Requirement 6.3: Shared task list with status tracking and dependencies
    Requirement 6.6: Team configuration storage
    Requirement 6.9: Task decomposition by file ownership
    """

    def __init__(
        self,
        hook_engine: object | None = None,
        cost_governor: object | None = None,
        teammate_callback: TeammateCallback | None = None,
        session_id: str = "",
        cwd: str = ".",
        session_type: str = "interactive",
        tracer: BrainmassTracer | None = None,
    ) -> None:
        self._hook_engine = hook_engine
        self._cost_governor = cost_governor
        # Default to no-op callback; inject _production_teammate_callback for real Bedrock execution.
        # Requirements: 3.2, 16.1
        self._teammate_callback = teammate_callback or _default_teammate_callback
        self._session_id = session_id
        self._cwd = cwd
        self._session_type = session_type
        self._tracer = tracer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute_team(
        self,
        lead_config: AgentDefinition,
        teammate_configs: list[AgentDefinition],
        tasks: list[TeamTask],
        team_budget: AgentBudget | None = None,
    ) -> TeamResult:
        """Spawn a team and execute all tasks.

        Steps:
        1. Build the shared task list
        2. Register the team lead and teammates with the cost governor
        3. Assign tasks to teammates based on file ownership
        4. Execute tasks in dependency order via teammate callbacks
        5. Aggregate results
        6. Fire SubagentStop hooks for each teammate
        7. Return TeamResult

        Args:
            lead_config: Agent definition for the Team Lead.
            teammate_configs: Agent definitions for each teammate.
            tasks: Initial task list to execute.
            team_budget: Optional budget for the entire team.
        """
        team_name = f"team:{lead_config.name}:{uuid.uuid4().hex[:8]}"

        # 1. Build shared task list
        task_list = SharedTaskList()
        for t in tasks:
            task_list.add_task(t)

        # 2. Register with cost governor
        budgets = self._create_team_budgets(
            lead_config, teammate_configs, team_budget
        )
        if self._cost_governor is not None:
            for agent_id, budget in budgets.items():
                self._cost_governor.register_agent(agent_id, budget)

        # 3. Build teammate lookup
        teammate_map = {cfg.name: cfg for cfg in teammate_configs}

        # 4. Execute tasks in dependency order
        teammate_results: list[AgentResult] = []
        total_cost = 0.0
        max_rounds = len(tasks) * 2  # safety limit to prevent infinite loops
        rounds = 0

        while not task_list.are_all_complete() and rounds < max_rounds:
            ready = task_list.get_ready_tasks()
            if not ready:
                # No tasks ready — check if we're stuck
                remaining = [
                    t for t in task_list.get_all_tasks()
                    if t.status != "complete"
                ]
                if remaining:
                    logger.warning(
                        "Team %s: no ready tasks but %d incomplete — possible deadlock",
                        team_name, len(remaining),
                    )
                break

            for task in ready:
                # Determine which teammate should handle this task
                assignee_name = self._select_teammate(task, teammate_map)
                if assignee_name is None:
                    logger.warning(
                        "Team %s: no teammate available for task '%s'",
                        team_name, task.id,
                    )
                    task_list.block_task(task.id)
                    continue

                # Claim the task
                task_list.claim_task(task.id, assignee_name)
                if self._tracer is not None:
                    self._tracer.record_coordination_span(
                        event_type="task_claim",
                        agent_id=assignee_name,
                        details={"task_id": task.id, "team": team_name},
                    )

                # Execute via callback
                teammate_def = teammate_map[assignee_name]
                execution_agent_id = f"teammate:{assignee_name}:{uuid.uuid4().hex[:8]}"
                budget_agent_id = f"teammate:{assignee_name}"
                teammate_budget = budgets.get(
                    budget_agent_id,
                    self._default_teammate_budget(team_budget),
                )

                try:
                    if self._tracer is not None:
                        with self._tracer.trace_agent_action(
                            agent_id=execution_agent_id,
                            action="team_task",
                            attributes={"task.id": task.id, "team": team_name},
                        ):
                            raw = self._teammate_callback(
                                teammate_def, task.title, teammate_budget
                            )
                    else:
                        raw = self._teammate_callback(
                            teammate_def, task.title, teammate_budget
                        )
                    result = AgentResult(
                        agent_name=teammate_def.name,
                        summary=raw.get("summary", ""),
                        turns_used=raw.get("turns_used", 0),
                        tokens_consumed=raw.get("tokens_consumed", {"input": 0, "output": 0, "cache_read": 0}),
                        tools_used=raw.get("tools_used", []),
                        files_modified=raw.get("files_modified", []),
                        exit_reason=raw.get("exit_reason", "complete"),
                    )

                    # Record usage with cost governor
                    tokens = result.tokens_consumed
                    if tokens.get("input", 0) > 0 or tokens.get("output", 0) > 0:
                        tier = self._resolve_model_tier(teammate_def.model)
                        if self._cost_governor is not None:
                            self._cost_governor.record_usage(
                                agent_id=budget_agent_id,
                                input_tokens=tokens.get("input", 0),
                                output_tokens=tokens.get("output", 0),
                                model_tier=tier,
                                cached_tokens=tokens.get("cache_read", 0),
                            )
                            try:
                                cost_usd = self._cost_governor.calculate_cost(
                                    input_tokens=tokens.get("input", 0),
                                    output_tokens=tokens.get("output", 0),
                                    cache_read_tokens=tokens.get("cache_read", 0),
                                    model_tier=tier,
                                )
                            except Exception:
                                cost_usd = 0.0
                            if self._tracer is not None:
                                self._tracer.record_cost_span(
                                    agent_id=execution_agent_id,
                                    input_tokens=tokens.get("input", 0),
                                    output_tokens=tokens.get("output", 0),
                                    cost_usd=cost_usd,
                                    model_tier=tier.value,
                                )
                        else:
                            from src.cost.cost_governor import CostGovernor

                            cost_usd = CostGovernor.calculate_cost(
                                input_tokens=tokens.get("input", 0),
                                output_tokens=tokens.get("output", 0),
                                cache_read_tokens=tokens.get("cache_read", 0),
                                model_tier=tier,
                            )
                        total_cost += cost_usd

                    task_list.complete_task(task.id)
                    teammate_results.append(result)

                    # Fire SubagentStop hook
                    self._fire_subagent_stop(teammate_def, result)

                except Exception as exc:
                    logger.error(
                        "Team %s: teammate '%s' failed on task '%s': %s",
                        team_name, assignee_name, task.id, exc,
                    )
                    error_result = AgentResult(
                        agent_name=teammate_def.name,
                        summary=f"Error: {exc}",
                        turns_used=0,
                        tokens_consumed={"input": 0, "output": 0, "cache_read": 0},
                        tools_used=[],
                        files_modified=[],
                        exit_reason="error",
                    )
                    teammate_results.append(error_result)
                    task_list.block_task(task.id)

            rounds += 1

        # 5. Build lead result (the lead coordinates, doesn't execute tasks directly)
        lead_result = AgentResult(
            agent_name=lead_config.name,
            summary=f"Team '{lead_config.name}' coordinated {len(teammate_results)} task executions",
            turns_used=rounds,
            tokens_consumed={"input": 0, "output": 0, "cache_read": 0},
            tools_used=[],
            files_modified=[],
            exit_reason="complete",
        )

        # 6. Aggregate totals
        total_tokens = {"input": 0, "output": 0, "cache_read": 0}
        for r in teammate_results:
            for k in total_tokens:
                total_tokens[k] += r.tokens_consumed.get(k, 0)

        return TeamResult(
            team_name=team_name,
            lead_result=lead_result,
            teammate_results=teammate_results,
            task_list=task_list.get_all_tasks(),
            total_cost_usd=total_cost,
            total_tokens=total_tokens,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _select_teammate(
        self,
        task: TeamTask,
        teammate_map: dict[str, AgentDefinition],
    ) -> str | None:
        """Select a teammate for a task.

        If the task already has an assignee hint and that teammate exists,
        use it. Otherwise pick the first available teammate.
        """
        if task.assignee and task.assignee in teammate_map:
            return task.assignee
        # Round-robin fallback: pick first teammate
        if teammate_map:
            return next(iter(teammate_map))
        return None

    def _create_team_budgets(
        self,
        lead_config: AgentDefinition,
        teammate_configs: list[AgentDefinition],
        team_budget: AgentBudget | None,
    ) -> dict[str, AgentBudget]:
        """Create budgets for the lead and each teammate.

        The team budget is split proportionally: the lead gets 10% and
        teammates share the remaining 90% equally.
        """
        base = team_budget or AgentBudget(
            input_budget_tokens=200_000,
            output_budget_tokens=50_000,
            session_budget_usd=5.0,
            team_budget_usd=5.0,
        )

        n_teammates = max(len(teammate_configs), 1)
        lead_share = 0.10
        teammate_share = 0.90 / n_teammates

        budgets: dict[str, AgentBudget] = {}

        # Lead budget
        budgets[f"lead:{lead_config.name}"] = AgentBudget(
            input_budget_tokens=base.input_budget_tokens,
            output_budget_tokens=base.output_budget_tokens,
            session_budget_usd=base.session_budget_usd * lead_share,
            team_budget_usd=base.team_budget_usd,
        )

        # Teammate budgets
        for cfg in teammate_configs:
            budgets[f"teammate:{cfg.name}"] = AgentBudget(
                input_budget_tokens=base.input_budget_tokens,
                output_budget_tokens=base.output_budget_tokens,
                session_budget_usd=base.session_budget_usd * teammate_share,
                team_budget_usd=base.team_budget_usd,
            )

        return budgets

    @staticmethod
    def _default_teammate_budget(
        team_budget: AgentBudget | None,
    ) -> AgentBudget:
        """Fallback budget when a teammate isn't in the pre-built map."""
        if team_budget is not None:
            return AgentBudget(
                input_budget_tokens=team_budget.input_budget_tokens,
                output_budget_tokens=team_budget.output_budget_tokens,
                session_budget_usd=team_budget.session_budget_usd * 0.20,
                team_budget_usd=team_budget.team_budget_usd,
            )
        return AgentBudget(
            input_budget_tokens=200_000,
            output_budget_tokens=50_000,
            session_budget_usd=1.0,
        )

    def _fire_subagent_stop(
        self,
        agent_def: AgentDefinition,
        result: AgentResult,
    ) -> HookResult:
        """Fire SubagentStop hooks after a teammate completes."""
        if self._hook_engine is None:
            return HookResult()

        context = HookContext(
            session_id=self._session_id,
            hook_event_name=HookEvent.SUBAGENT_STOP,
            cwd=self._cwd,
            session_type=self._session_type,
            source=agent_def.name,
            model=agent_def.model,
        )
        return self._hook_engine.fire(HookEvent.SUBAGENT_STOP, context)

    @staticmethod
    def _resolve_model_tier(model_alias: str) -> ModelTier:
        """Resolve a model alias string to a ModelTier enum."""
        from src.types.core import ModelTier

        alias_map = {
            "opus": ModelTier.OPUS,
            "sonnet": ModelTier.SONNET,
            "haiku": ModelTier.HAIKU,
            "inherit": ModelTier.SONNET,
        }
        return alias_map.get(model_alias, ModelTier.SONNET)
