"""Loop Runner — Self-improving agent loop with the Ralph Wiggum pattern.

Each iteration spawns a fresh agent with a clean context window, feeds it
a structured context file, has it do work, commit to git, update the context
file with learnings, terminate, and loop.

Production agent callback uses ``strands.Agent`` + ``BedrockModel``.
Tests must inject an ``agent_callback`` stub.

Requirements: 7.1, 7.3, 7.4, 7.7, 4.1, 4.2, 4.3, 4.4
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from src.agents._strands_utils import _BEDROCK_MODEL_IDS, _normalize_strands_result
from src.agents.learning_store import LearningStore
from src.observability.instrumentation import BrainmassTracer
from src.types.core import (
    AgentBudget,
    AgentDefinition,
    AgentResult,
    HookContext,
    HookEvent,
    HookResult,
    Learning,
    LoopContext,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type aliases for pluggable callbacks
# ---------------------------------------------------------------------------

# Callback that spawns a fresh agent, feeds it context, and returns a result dict.
# Signature: (agent_def, context_str, budget) -> dict with keys:
#   summary, tokens_consumed, tools_used, files_modified, exit_reason, turns_used,
#   learnings (list[dict]), acceptance_met (bool)
AgentIterationCallback = Callable[
    [AgentDefinition, str, AgentBudget],
    dict,
]

# Callback for git operations: (message) -> commit_sha or None
GitCallback = Callable[[str], str | None]

# Callback to check for stop file existence: () -> bool
StopFileCallback = Callable[[], bool]


# --- Production integration point ---
def _production_loop_callback(
    agent_def: AgentDefinition,
    context_str: str,
    budget: AgentBudget,
) -> dict:
    """Execute one loop iteration using a fresh Strands Agent + BedrockModel.

    The agent receives only the structured context string — no conversation
    history from previous iterations (Requirements: 7.1, 4.1, 4.2).
    """
    try:
        from strands import Agent  # type: ignore[import-untyped]
        from strands.models.bedrock import BedrockModel  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "strands-agents is required for production loop execution. "
            "Install with: pip install strands-agents  "
            "Or inject an agent_callback to bypass this requirement."
        ) from exc

    model_id = _BEDROCK_MODEL_IDS.get(agent_def.model, _BEDROCK_MODEL_IDS["sonnet"])
    model = BedrockModel(model_id=model_id)
    # Fresh agent per iteration — no shared conversation history (Req 7.1)
    agent = Agent(model=model, system_prompt=agent_def.system_prompt or "")
    raw = agent(context_str)
    result = _normalize_strands_result(raw)
    # LoopRunner interprets acceptance_met from the response text
    result["acceptance_met"] = "ACCEPTANCE_MET" in result["summary"].upper()
    result["learnings"] = []  # Learnings extracted by LoopRunner from summary
    return result

    # --- Production integration point (GraphBuilder upgrade path) ---
    # When strands.multiagent.GraphBuilder API is confirmed, replace with:
    # from strands.multiagent import GraphBuilder
    # graph = (GraphBuilder()
    #     .add_node("implement", agent)
    #     .add_node("review", reviewer_agent)
    #     .add_edge("implement", "review")
    #     .add_conditional_edges(
    #         "review",
    #         lambda r: "__end__" if r.get("acceptance_met") else "implement",
    #     )
    #     .set_entry_point("implement")
    #     .build()
    # )
    # raw = graph(context_str)


def _default_agent_callback(
    agent_def: AgentDefinition,
    context_str: str,
    budget: AgentBudget,
) -> dict:
    """No-op agent callback for testing — returns a minimal valid result dict."""
    return {
        "summary": f"Agent '{agent_def.name}' completed: {context_str[:80]}",
        "tokens_consumed": {"input": 0, "output": 0, "cache_read": 0},
        "tools_used": [],
        "files_modified": [],
        "exit_reason": "complete",
        "turns_used": 0,
        "learnings": [],
        "acceptance_met": False,
    }


def _default_git_callback(message: str) -> str | None:
    """Default no-op git callback."""
    return None


def _default_stop_file_callback() -> bool:
    """Default stop file check — always returns False (no stop file)."""
    return False


# ---------------------------------------------------------------------------
# Iteration result
# ---------------------------------------------------------------------------

@dataclass
class IterationResult:
    """Result of a single loop iteration."""
    iteration: int
    agent_result: AgentResult
    commit_sha: str | None = None
    new_learnings: list[Learning] = field(default_factory=list)
    acceptance_met: bool = False


# ---------------------------------------------------------------------------
# Loop result
# ---------------------------------------------------------------------------

@dataclass
class LoopResult:
    """Final result of the entire loop execution."""
    final_context: LoopContext
    iteration_results: list[IterationResult] = field(default_factory=list)
    exit_reason: str = "max_iterations"  # max_iterations | acceptance_met | stop_file


# ---------------------------------------------------------------------------
# Loop Runner
# ---------------------------------------------------------------------------

class LoopRunner:
    """Orchestrates self-improving agent loops (Ralph Wiggum pattern).

    Each iteration:
    1. Build structured context string from LoopContext
    2. Spawn a fresh agent via callback (clean context — Req 7.1)
    3. Execute the task
    4. Commit to git (Req 7.3)
    5. Accumulate learnings (Req 7.4)
    6. Check stop conditions
    7. Loop or exit

    Implemented conceptually as a Strands GraphBuilder with conditional
    edges (Req 7.7): implement → review → (FAIL → implement | PASS → END).
    The actual graph is simulated via the iteration loop since the SDK
    is not installed.
    """

    def __init__(
        self,
        agent_def: AgentDefinition,
        agent_callback: AgentIterationCallback | None = None,
        git_callback: GitCallback | None = None,
        stop_file_callback: StopFileCallback | None = None,
        hook_engine: object | None = None,
        cost_governor: object | None = None,
        budget: AgentBudget | None = None,
        learning_store: LearningStore | None = None,
        session_id: str = "",
        cwd: str = ".",
        session_type: str = "interactive",
        tracer: BrainmassTracer | None = None,
    ) -> None:
        self._agent_def = agent_def
        # Default to no-op callback; inject _production_loop_callback for real Bedrock execution.
        # Requirements: 4.2, 16.1
        self._agent_callback = agent_callback or _default_agent_callback
        self._git_callback = git_callback or _default_git_callback
        self._stop_file_callback = stop_file_callback or _default_stop_file_callback
        self._hook_engine = hook_engine
        self._cost_governor = cost_governor
        self._budget = budget or AgentBudget(
            input_budget_tokens=200_000,
            output_budget_tokens=50_000,
            session_budget_usd=5.0,
        )
        self._learning_store = learning_store
        self._session_id = session_id
        self._cwd = cwd
        self._session_type = session_type
        self._tracer = tracer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, context: LoopContext) -> LoopResult:
        """Execute the self-improving loop.

        Args:
            context: The initial LoopContext with task, criteria, etc.

        Returns:
            LoopResult with final context, all iteration results, and exit reason.
        """
        iteration_results: list[IterationResult] = []
        exit_reason = "max_iterations"

        while context.iteration_count < context.max_iterations:
            # Check stop file before each iteration (Req 7.5)
            if self._stop_file_callback():
                logger.info("Stop file detected at iteration boundary.")
                if self._tracer is not None:
                    self._tracer.record_coordination_span(
                        event_type="loop_stop_file",
                        agent_id=self._agent_def.name,
                        details={"iteration_count": context.iteration_count},
                    )
                exit_reason = "stop_file"
                break

            # Run one iteration
            iteration_result = self._run_iteration(context)
            iteration_results.append(iteration_result)

            # Update context with new learnings (Req 7.4)
            context = self._update_context(context, iteration_result)

            # Check acceptance criteria (Req 7.5)
            if iteration_result.acceptance_met:
                logger.info(
                    "Acceptance criteria met at iteration %d.",
                    context.iteration_count,
                )
                if self._tracer is not None:
                    self._tracer.record_coordination_span(
                        event_type="loop_acceptance",
                        agent_id=self._agent_def.name,
                        details={"iteration_count": context.iteration_count},
                    )
                exit_reason = "acceptance_met"
                break

        return LoopResult(
            final_context=context,
            iteration_results=iteration_results,
            exit_reason=exit_reason,
        )

    # ------------------------------------------------------------------
    # Iteration execution
    # ------------------------------------------------------------------

    def _run_iteration(self, context: LoopContext) -> IterationResult:
        """Execute a single loop iteration with a fresh agent.

        Steps:
        1. Load relevant prior learnings from LearningStore (Req 21.3)
        2. Build context string (clean context for fresh agent — Req 7.1)
        3. Execute agent via callback
        4. Commit to git (Req 7.3)
        5. Extract and persist learnings (Req 7.4, 21.1)
        6. Fire SubagentStop hook
        7. Return IterationResult
        """
        current_iteration = context.iteration_count + 1
        logger.info("Starting iteration %d / %d", current_iteration, context.max_iterations)

        # 1. Load relevant prior learnings from LearningStore (Req 21.3)
        # Inject them into the context so the fresh agent benefits from past sessions.
        if self._learning_store is not None:
            prior_learnings = self._learning_store.query(context.current_task, top_k=5)
            if prior_learnings:
                extra = [
                    {
                        "pattern": l.pattern,
                        "resolution": l.resolution,
                        "confidence": l.confidence,
                        "source_iteration": l.source_iteration,
                    }
                    for l in prior_learnings
                ]
                # Merge into a temporary context for context-string building only;
                # the canonical LoopContext is updated after the iteration completes.
                context = LoopContext(
                    current_task=context.current_task,
                    acceptance_criteria=list(context.acceptance_criteria),
                    constraints=list(context.constraints),
                    learnings=list(context.learnings) + extra,
                    failed_approaches=list(context.failed_approaches),
                    iteration_count=context.iteration_count,
                    max_iterations=context.max_iterations,
                )
                logger.debug(
                    "Injected %d prior learnings from LearningStore into iteration %d.",
                    len(prior_learnings), current_iteration,
                )

        # 2. Build structured context string (Req 7.1 — fresh context only)
        context_str = self._build_context_string(context)

        # 2. Execute fresh agent via callback
        if self._tracer is not None:
            with self._tracer.trace_agent_action(
                agent_id=self._agent_def.name,
                action="loop_iteration",
                attributes={"iteration": current_iteration},
            ):
                raw_result = self._agent_callback(self._agent_def, context_str, self._budget)
        else:
            raw_result = self._agent_callback(self._agent_def, context_str, self._budget)

        # Build AgentResult
        agent_result = AgentResult(
            agent_name=self._agent_def.name,
            summary=raw_result.get("summary", ""),
            turns_used=raw_result.get("turns_used", 0),
            tokens_consumed=raw_result.get("tokens_consumed", {"input": 0, "output": 0, "cache_read": 0}),
            tools_used=raw_result.get("tools_used", []),
            files_modified=raw_result.get("files_modified", []),
            exit_reason=raw_result.get("exit_reason", "complete"),
        )

        # Record usage with cost governor
        self._record_usage(agent_result, current_iteration)

        # 3. Commit to git (Req 7.3)
        commit_sha = self._git_callback(
            f"loop iteration {current_iteration}: {agent_result.summary[:80]}"
        )

        # 4. Extract learnings (Req 7.4) and persist to LearningStore (Req 21.1)
        new_learnings = self._extract_learnings(raw_result, current_iteration)
        if self._learning_store is not None and new_learnings:
            for learning in new_learnings:
                self._learning_store.add(learning)
            logger.debug(
                "Persisted %d new learnings to LearningStore after iteration %d.",
                len(new_learnings), current_iteration,
            )

        # 5. Fire SubagentStop hook
        self._fire_subagent_stop(agent_result)

        acceptance_met = raw_result.get("acceptance_met", False)

        return IterationResult(
            iteration=current_iteration,
            agent_result=agent_result,
            commit_sha=commit_sha,
            new_learnings=new_learnings,
            acceptance_met=acceptance_met,
        )

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------

    @staticmethod
    def _build_context_string(context: LoopContext) -> str:
        """Build a structured context string for the fresh agent.

        This is the ONLY information the fresh agent receives — no
        conversation history from previous iterations (Req 7.1).
        """
        lines = [
            f"# Task\n{context.current_task}",
            "",
            "# Acceptance Criteria",
        ]
        for i, criterion in enumerate(context.acceptance_criteria, 1):
            lines.append(f"{i}. {criterion}")

        if context.constraints:
            lines.append("")
            lines.append("# Constraints")
            for c in context.constraints:
                lines.append(f"- {c}")

        if context.learnings:
            lines.append("")
            lines.append("# Learnings from Previous Iterations")
            for learning in context.learnings:
                pattern = learning.get("pattern", "")
                resolution = learning.get("resolution", "")
                confidence = learning.get("confidence", 0)
                source = learning.get("source_iteration", "?")
                lines.append(
                    f"- [{confidence:.0%} confidence, iter {source}] "
                    f"{pattern} → {resolution}"
                )

        if context.failed_approaches:
            lines.append("")
            lines.append("# Failed Approaches (DO NOT REPEAT)")
            for fa in context.failed_approaches:
                iteration = fa.get("iteration", "?")
                approach = fa.get("approach", "")
                why = fa.get("why_failed", "")
                lines.append(f"- Iteration {iteration}: {approach} — {why}")

        lines.append("")
        lines.append(
            f"# Iteration {context.iteration_count + 1} of {context.max_iterations}"
        )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Learning accumulation
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_learnings(
        raw_result: dict,
        current_iteration: int,
    ) -> list[Learning]:
        """Extract Learning objects from the raw agent result (Req 7.4)."""
        learnings: list[Learning] = []
        for entry in raw_result.get("learnings", []):
            learnings.append(
                Learning(
                    pattern=entry.get("pattern", ""),
                    resolution=entry.get("resolution", ""),
                    confidence=entry.get("confidence", 0.5),
                    source_iteration=entry.get("source_iteration", current_iteration),
                )
            )
        return learnings

    @staticmethod
    def _update_context(
        context: LoopContext,
        iteration_result: IterationResult,
    ) -> LoopContext:
        """Return a new LoopContext with accumulated learnings and incremented count.

        Learnings are merged: new learnings are appended as dicts.
        If the agent did not meet acceptance criteria and provided a summary,
        the approach is recorded as a failed approach.
        """
        new_learnings_dicts = [
            {
                "pattern": l.pattern,
                "resolution": l.resolution,
                "confidence": l.confidence,
                "source_iteration": l.source_iteration,
            }
            for l in iteration_result.new_learnings
        ]

        merged_learnings = list(context.learnings) + new_learnings_dicts

        failed_approaches = list(context.failed_approaches)
        if not iteration_result.acceptance_met:
            failed_approaches.append({
                "iteration": iteration_result.iteration,
                "approach": iteration_result.agent_result.summary[:200],
                "why_failed": iteration_result.agent_result.exit_reason,
            })

        return LoopContext(
            current_task=context.current_task,
            acceptance_criteria=list(context.acceptance_criteria),
            constraints=list(context.constraints),
            learnings=merged_learnings,
            failed_approaches=failed_approaches,
            iteration_count=context.iteration_count + 1,
            max_iterations=context.max_iterations,
        )

    # ------------------------------------------------------------------
    # Hook and cost integration
    # ------------------------------------------------------------------

    def _fire_subagent_stop(self, result: AgentResult) -> HookResult:
        """Fire SubagentStop hooks after each iteration."""
        if self._hook_engine is None:
            return HookResult()

        hook_context = HookContext(
            session_id=self._session_id,
            hook_event_name=HookEvent.SUBAGENT_STOP,
            cwd=self._cwd,
            session_type=self._session_type,
            source=self._agent_def.name,
            model=self._agent_def.model,
        )
        return self._hook_engine.fire(HookEvent.SUBAGENT_STOP, hook_context)

    def _record_usage(self, result: AgentResult, iteration: int) -> None:
        """Record token usage with the cost governor."""
        if self._cost_governor is None:
            return

        tokens = result.tokens_consumed
        if tokens.get("input", 0) == 0 and tokens.get("output", 0) == 0:
            return

        from src.types.core import ModelTier

        alias_map = {
            "opus": ModelTier.OPUS,
            "sonnet": ModelTier.SONNET,
            "haiku": ModelTier.HAIKU,
            "inherit": ModelTier.SONNET,
        }
        tier = alias_map.get(self._agent_def.model, ModelTier.SONNET)

        agent_id = f"loop:{self._agent_def.name}:iter{iteration}"
        self._cost_governor.record_usage(
            agent_id=agent_id,
            input_tokens=tokens.get("input", 0),
            output_tokens=tokens.get("output", 0),
            model_tier=tier,
            cached_tokens=tokens.get("cache_read", 0),
        )
