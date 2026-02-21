"""Compound Loop Orchestration — chains Analysis → Planning → Execution loops.

Each stage is a separate agent/loop with its own context window. The output
of one stage feeds as input to the next stage. Pipeline stages are defined
via YAML configuration. SubagentStop hooks fire after each stage and write
status to .brainmass/pipeline-state.json.

Production stage callback uses ``strands.Agent`` + ``BedrockModel``.
Tests must inject a ``stage_callback`` stub.

Requirements: 22.1, 22.2, 22.3, 22.4, 5.1, 5.2, 5.3, 5.4
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import yaml

from src.agents._strands_utils import _BEDROCK_MODEL_IDS, _normalize_strands_result
from src.types.core import (
    AgentBudget,
    AgentDefinition,
    AgentResult,
    HookContext,
    HookEvent,
    HookResult,
    LoopContext,
    PipelineConfig,
    PipelineStage,
    PipelineStageStatus,
)

logger = logging.getLogger(__name__)

# Default path for pipeline state file (Req 22.4)
DEFAULT_PIPELINE_STATE_PATH = ".brainmass/pipeline-state.json"


# ---------------------------------------------------------------------------
# Type aliases for pluggable callbacks
# ---------------------------------------------------------------------------

# Callback that executes a single loop stage and returns a result dict.
# Signature: (agent_def, loop_context, budget) -> dict with keys:
#   summary, output (dict of output values), tokens_consumed, tools_used,
#   files_modified, exit_reason, turns_used, acceptance_met (bool)
StageAgentCallback = Callable[
    [AgentDefinition, LoopContext, AgentBudget],
    dict,
]


def _default_stage_callback(
    agent_def: AgentDefinition,
    loop_context: LoopContext,
    budget: AgentBudget,
) -> dict:
    """No-op stage callback used when no callback is injected and SDK is absent.

    Returns a minimal valid result dict for testing purposes.
    """
    return {
        "summary": f"Stage '{agent_def.name}' completed: {loop_context.current_task[:80]}",
        "output": {},
        "tokens_consumed": {"input": 0, "output": 0, "cache_read": 0},
        "tools_used": [],
        "files_modified": [],
        "exit_reason": "complete",
        "turns_used": 0,
        "acceptance_met": True,
    }


# --- Production integration point ---
def _production_stage_callback(
    agent_def: AgentDefinition,
    loop_context: LoopContext,
    budget: AgentBudget,
) -> dict:
    """Execute a pipeline stage using a real Strands Agent + BedrockModel.

    Requirements: 5.1, 5.2, 5.3
    """
    try:
        from strands import Agent  # type: ignore[import-untyped]
        from strands.models.bedrock import BedrockModel  # type: ignore[import-untyped]
    except ImportError as exc:
        raise RuntimeError(
            "strands-agents is required for production stage execution. "
            "Install with: pip install strands-agents  "
            "Or inject a stage_callback to bypass this requirement."
        ) from exc

    model_id = _BEDROCK_MODEL_IDS.get(agent_def.model, _BEDROCK_MODEL_IDS["sonnet"])
    model = BedrockModel(model_id=model_id)
    agent = Agent(model=model, system_prompt=agent_def.system_prompt or "")
    raw = agent(loop_context.current_task)
    result = _normalize_strands_result(raw)
    result["output"] = {"result": result["summary"]}
    result["acceptance_met"] = result["exit_reason"] == "complete"
    return result

    # --- Production integration point (Workflow upgrade path) ---
    # When strands.multiagent.Workflow API is confirmed, replace with:
    # from strands.multiagent import Workflow
    # workflow = Workflow(agents=[agent])
    # raw = workflow(loop_context.current_task)
    # result = _normalize_strands_result(raw)
    # result["output"] = {"result": result["summary"]}
    # result["acceptance_met"] = result["exit_reason"] == "complete"
    # return result


# ---------------------------------------------------------------------------
# Stage result
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    """Result of a single pipeline stage execution."""

    stage_name: str
    status: PipelineStageStatus
    agent_result: AgentResult
    output: dict  # key-value pairs passed to the next stage
    started_at: datetime
    completed_at: datetime
    error: str | None = None


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Final result of the entire compound loop pipeline."""

    pipeline_name: str
    stage_results: list[StageResult] = field(default_factory=list)
    final_output: dict = field(default_factory=dict)
    exit_reason: str = "complete"  # complete | stage_failed | stopped
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @property
    def succeeded(self) -> bool:
        """True if all stages completed successfully."""
        return self.exit_reason == "complete" and all(
            r.status == PipelineStageStatus.COMPLETE for r in self.stage_results
        )


# ---------------------------------------------------------------------------
# Pipeline state (written to .brainmass/pipeline-state.json — Req 22.4)
# ---------------------------------------------------------------------------

@dataclass
class PipelineState:
    """Serializable state written after each stage via SubagentStop hook."""

    pipeline_name: str
    current_stage: str
    stage_statuses: dict[str, str]  # stage_name → PipelineStageStatus.value
    stage_outputs: dict[str, dict]  # stage_name → output dict
    started_at: str  # ISO-8601
    updated_at: str  # ISO-8601
    exit_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "pipeline_name": self.pipeline_name,
            "current_stage": self.current_stage,
            "stage_statuses": self.stage_statuses,
            "stage_outputs": self.stage_outputs,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "exit_reason": self.exit_reason,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PipelineState:
        return cls(
            pipeline_name=data["pipeline_name"],
            current_stage=data["current_stage"],
            stage_statuses=data.get("stage_statuses", {}),
            stage_outputs=data.get("stage_outputs", {}),
            started_at=data["started_at"],
            updated_at=data["updated_at"],
            exit_reason=data.get("exit_reason"),
        )


# ---------------------------------------------------------------------------
# YAML pipeline config loader
# ---------------------------------------------------------------------------

def load_pipeline_config(yaml_path: str) -> PipelineConfig:
    """Load a PipelineConfig from a YAML file.

    Expected YAML structure::

        name: my-pipeline
        description: "Analysis → Planning → Execution"
        stages:
          - name: analysis
            agent_def_path: .brainmass/agents/analyzer.md
            max_iterations: 3
            acceptance_criteria:
              - "Prioritized task list produced"
            constraints:
              - "Read-only access"
            input_mapping: {}
            output_mapping:
              task_list: task_list
          - name: planning
            agent_def_path: .brainmass/agents/planner.md
            max_iterations: 5
            acceptance_criteria:
              - "PRD and subtasks produced"
            input_mapping:
              task_list: task_list
            output_mapping:
              prd: prd
              subtasks: subtasks
          - name: execution
            agent_def_path: .brainmass/agents/executor.md
            max_iterations: 10
            acceptance_criteria:
              - "All subtasks implemented and tests pass"
            input_mapping:
              prd: prd
              subtasks: subtasks
            output_mapping:
              status: status

    Requirements: 22.3
    """
    with open(yaml_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    stages: list[PipelineStage] = []
    for s in raw.get("stages", []):
        stages.append(
            PipelineStage(
                name=s["name"],
                agent_def_path=s["agent_def_path"],
                input_mapping=s.get("input_mapping", {}),
                output_mapping=s.get("output_mapping", {}),
                max_iterations=s.get("max_iterations", 5),
                acceptance_criteria=s.get("acceptance_criteria", []),
                constraints=s.get("constraints", []),
            )
        )

    return PipelineConfig(
        name=raw["name"],
        description=raw.get("description", ""),
        stages=stages,
    )


def load_pipeline_config_from_dict(data: dict) -> PipelineConfig:
    """Load a PipelineConfig from a plain dict (useful for tests).

    Requirements: 22.3
    """
    stages: list[PipelineStage] = []
    for s in data.get("stages", []):
        stages.append(
            PipelineStage(
                name=s["name"],
                agent_def_path=s.get("agent_def_path", ""),
                input_mapping=s.get("input_mapping", {}),
                output_mapping=s.get("output_mapping", {}),
                max_iterations=s.get("max_iterations", 5),
                acceptance_criteria=s.get("acceptance_criteria", []),
                constraints=s.get("constraints", []),
            )
        )

    return PipelineConfig(
        name=data["name"],
        description=data.get("description", ""),
        stages=stages,
    )


# ---------------------------------------------------------------------------
# CompoundLoopOrchestrator
# ---------------------------------------------------------------------------

class CompoundLoopOrchestrator:
    """Orchestrates compound loop pipelines chaining multiple agent stages.

    Each stage:
    1. Receives input mapped from the previous stage's output (Req 22.2)
    2. Runs as a LoopContext-driven agent loop
    3. Produces output mapped to the next stage's input keys
    4. Fires SubagentStop hooks on completion (Req 22.4)
    5. Writes pipeline state to .brainmass/pipeline-state.json (Req 22.4)

    The pipeline is configurable via YAML-defined stages (Req 22.3).

    Conceptually implemented as a Strands ``Workflow`` or ``Graph`` for
    sequential stages — the actual graph is simulated via the stage loop
    since the SDK is not installed.

    Usage::

        config = load_pipeline_config("pipeline.yaml")
        orchestrator = CompoundLoopOrchestrator(config)
        result = orchestrator.run(initial_input={"task": "Build auth module"})
    """

    def __init__(
        self,
        config: PipelineConfig,
        stage_callback: StageAgentCallback | None = None,
        hook_engine: Any | None = None,
        cost_governor: Any | None = None,
        budget: AgentBudget | None = None,
        pipeline_state_path: str = DEFAULT_PIPELINE_STATE_PATH,
        session_id: str = "",
        cwd: str = ".",
        session_type: str = "interactive",
    ) -> None:
        self._config = config
        # Default to no-op callback; inject _production_stage_callback for real Bedrock execution.
        # Requirements: 5.2, 16.1
        self._stage_callback = stage_callback or _default_stage_callback
        self._hook_engine = hook_engine
        self._cost_governor = cost_governor
        self._budget = budget or AgentBudget(
            input_budget_tokens=200_000,
            output_budget_tokens=50_000,
            session_budget_usd=10.0,
        )
        self._pipeline_state_path = pipeline_state_path
        self._session_id = session_id
        self._cwd = cwd
        self._session_type = session_type

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, initial_input: dict | None = None) -> PipelineResult:
        """Execute the full pipeline, chaining stages sequentially.

        Args:
            initial_input: Key-value pairs fed into the first stage.

        Returns:
            PipelineResult with all stage results and final output.

        Requirements: 22.1, 22.2
        """
        started_at = datetime.now(UTC)
        stage_results: list[StageResult] = []
        current_input: dict = initial_input or {}
        exit_reason = "complete"

        # Initialise pipeline state file
        state = PipelineState(
            pipeline_name=self._config.name,
            current_stage="",
            stage_statuses={s.name: PipelineStageStatus.PENDING.value for s in self._config.stages},
            stage_outputs={},
            started_at=started_at.isoformat(),
            updated_at=started_at.isoformat(),
        )
        self._write_pipeline_state(state)

        for stage in self._config.stages:
            logger.info(
                "Pipeline '%s': starting stage '%s'",
                self._config.name,
                stage.name,
            )

            # Map previous output to this stage's input (Req 22.2)
            stage_input = self._map_input(current_input, stage.input_mapping)

            # Update state: mark stage as running
            state.current_stage = stage.name
            state.stage_statuses[stage.name] = PipelineStageStatus.RUNNING.value
            state.updated_at = datetime.now(UTC).isoformat()
            self._write_pipeline_state(state)

            # Execute the stage
            stage_result = self._run_stage(stage, stage_input)
            stage_results.append(stage_result)

            # Update state with stage outcome
            state.stage_statuses[stage.name] = stage_result.status.value
            state.stage_outputs[stage.name] = stage_result.output
            state.updated_at = datetime.now(UTC).isoformat()

            if stage_result.status == PipelineStageStatus.FAILED:
                exit_reason = "stage_failed"
                state.exit_reason = exit_reason
                self._write_pipeline_state(state)
                logger.error(
                    "Pipeline '%s': stage '%s' failed — aborting pipeline.",
                    self._config.name,
                    stage.name,
                )
                break

            # Map this stage's output to the next stage's input (Req 22.2)
            current_input = self._map_output(stage_result.output, stage.output_mapping)
            self._write_pipeline_state(state)

        completed_at = datetime.now(UTC)
        state.exit_reason = exit_reason
        state.updated_at = completed_at.isoformat()
        self._write_pipeline_state(state)

        return PipelineResult(
            pipeline_name=self._config.name,
            stage_results=stage_results,
            final_output=current_input,
            exit_reason=exit_reason,
            started_at=started_at,
            completed_at=completed_at,
        )

    # ------------------------------------------------------------------
    # Stage execution
    # ------------------------------------------------------------------

    def _run_stage(
        self,
        stage: PipelineStage,
        stage_input: dict,
    ) -> StageResult:
        """Execute a single pipeline stage as a LoopContext-driven agent.

        Steps:
        1. Build a LoopContext from the stage config and input
        2. Execute via the pluggable stage_callback
        3. Fire SubagentStop hook (Req 22.4)
        4. Return StageResult

        Requirements: 22.1, 22.2, 22.4
        """
        started_at = datetime.now(UTC)

        # Build a minimal AgentDefinition for this stage
        agent_def = AgentDefinition(
            name=stage.name,
            description=f"Pipeline stage: {stage.name}",
            model="sonnet",
            system_prompt=f"You are the {stage.name} stage of the pipeline.",
            file_path=stage.agent_def_path,
        )

        # Build LoopContext for this stage
        task_description = self._build_task_description(stage, stage_input)
        loop_context = LoopContext(
            current_task=task_description,
            acceptance_criteria=list(stage.acceptance_criteria),
            constraints=list(stage.constraints),
            learnings=[],
            failed_approaches=[],
            iteration_count=0,
            max_iterations=stage.max_iterations,
        )

        try:
            raw_result = self._stage_callback(agent_def, loop_context, self._budget)

            agent_result = AgentResult(
                agent_name=stage.name,
                summary=raw_result.get("summary", ""),
                turns_used=raw_result.get("turns_used", 0),
                tokens_consumed=raw_result.get(
                    "tokens_consumed", {"input": 0, "output": 0, "cache_read": 0}
                ),
                tools_used=raw_result.get("tools_used", []),
                files_modified=raw_result.get("files_modified", []),
                exit_reason=raw_result.get("exit_reason", "complete"),
            )

            # Record usage with cost governor
            self._record_usage(agent_result, stage.name)

            # Extract stage output
            stage_output = raw_result.get("output", {})
            status = (
                PipelineStageStatus.COMPLETE
                if raw_result.get("acceptance_met", True)
                else PipelineStageStatus.FAILED
            )

        except Exception as exc:
            logger.error("Stage '%s' raised an exception: %s", stage.name, exc)
            agent_result = AgentResult(
                agent_name=stage.name,
                summary=f"Error: {exc}",
                turns_used=0,
                tokens_consumed={"input": 0, "output": 0, "cache_read": 0},
                tools_used=[],
                files_modified=[],
                exit_reason="error",
            )
            stage_output = {}
            status = PipelineStageStatus.FAILED

        completed_at = datetime.now(UTC)

        # Fire SubagentStop hook after stage completion (Req 22.4)
        self._fire_subagent_stop(stage.name, agent_result)

        return StageResult(
            stage_name=stage.name,
            status=status,
            agent_result=agent_result,
            output=stage_output,
            started_at=started_at,
            completed_at=completed_at,
            error=agent_result.summary if status == PipelineStageStatus.FAILED else None,
        )

    # ------------------------------------------------------------------
    # Input / output mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _map_input(previous_output: dict, input_mapping: dict) -> dict:
        """Map previous stage output keys to this stage's input keys.

        If ``input_mapping`` is empty, the full previous output is passed
        through unchanged.

        Requirements: 22.2
        """
        if not input_mapping:
            return dict(previous_output)
        mapped: dict = {}
        for src_key, dst_key in input_mapping.items():
            if src_key in previous_output:
                mapped[dst_key] = previous_output[src_key]
        return mapped

    @staticmethod
    def _map_output(stage_output: dict, output_mapping: dict) -> dict:
        """Map this stage's output keys to the keys expected by the next stage.

        If ``output_mapping`` is empty, the full stage output is passed
        through unchanged.

        Requirements: 22.2
        """
        if not output_mapping:
            return dict(stage_output)
        mapped: dict = {}
        for src_key, dst_key in output_mapping.items():
            if src_key in stage_output:
                mapped[dst_key] = stage_output[src_key]
        return mapped

    # ------------------------------------------------------------------
    # Task description builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_task_description(stage: PipelineStage, stage_input: dict) -> str:
        """Build a human-readable task description for the stage's LoopContext."""
        lines = [f"Stage: {stage.name}"]
        if stage_input:
            lines.append("")
            lines.append("## Input from previous stage")
            for key, value in stage_input.items():
                value_str = json.dumps(value) if not isinstance(value, str) else value
                lines.append(f"- {key}: {value_str}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Pipeline state persistence (Req 22.4)
    # ------------------------------------------------------------------

    def _write_pipeline_state(self, state: PipelineState) -> None:
        """Write pipeline state to .brainmass/pipeline-state.json.

        Creates parent directories if they do not exist.

        Requirements: 22.4
        """
        try:
            os.makedirs(os.path.dirname(self._pipeline_state_path), exist_ok=True)
            with open(self._pipeline_state_path, "w", encoding="utf-8") as fh:
                json.dump(state.to_dict(), fh, indent=2)
            logger.debug(
                "Pipeline state written to %s (stage=%s)",
                self._pipeline_state_path,
                state.current_stage,
            )
        except OSError as exc:
            logger.warning("Failed to write pipeline state: %s", exc)

    @staticmethod
    def read_pipeline_state(path: str = DEFAULT_PIPELINE_STATE_PATH) -> PipelineState | None:
        """Read and deserialise pipeline state from disk.

        Returns ``None`` if the file does not exist or cannot be parsed.
        """
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            return PipelineState.from_dict(data)
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            logger.debug("Could not read pipeline state from %s: %s", path, exc)
            return None

    # ------------------------------------------------------------------
    # Hook integration (Req 22.4)
    # ------------------------------------------------------------------

    def _fire_subagent_stop(
        self,
        stage_name: str,
        result: AgentResult,
    ) -> HookResult:
        """Fire SubagentStop hooks after a stage completes.

        Requirements: 22.4
        """
        if self._hook_engine is None:
            return HookResult()

        context = HookContext(
            session_id=self._session_id,
            hook_event_name=HookEvent.SUBAGENT_STOP,
            cwd=self._cwd,
            session_type=self._session_type,
            source=stage_name,
            model="sonnet",
        )
        return self._hook_engine.fire(HookEvent.SUBAGENT_STOP, context)

    # ------------------------------------------------------------------
    # Cost governor integration
    # ------------------------------------------------------------------

    def _record_usage(self, result: AgentResult, stage_name: str) -> None:
        """Record token usage with the cost governor."""
        if self._cost_governor is None:
            return

        tokens = result.tokens_consumed
        if tokens.get("input", 0) == 0 and tokens.get("output", 0) == 0:
            return

        from src.types.core import ModelTier

        agent_id = f"pipeline:{self._config.name}:stage:{stage_name}"
        self._cost_governor.record_usage(
            agent_id=agent_id,
            input_tokens=tokens.get("input", 0),
            output_tokens=tokens.get("output", 0),
            model_tier=ModelTier.SONNET,
            cached_tokens=tokens.get("cache_read", 0),
        )
