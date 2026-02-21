"""Unit tests for src/agents/compound_loop.py.

Tests cover:
- Pipeline config loading from dict and YAML
- Input/output mapping between stages
- Stage execution and result building
- Pipeline state persistence to JSON
- SubagentStop hook firing after each stage
- Stage failure aborts the pipeline
- Cost governor integration
- Full pipeline run with chained stages

Requirements: 22.1, 22.2, 22.3, 22.4
"""

from __future__ import annotations

from unittest.mock import MagicMock

from src.agents.compound_loop import (
    CompoundLoopOrchestrator,
    PipelineState,
    _default_stage_callback,
    load_pipeline_config_from_dict,
)
from src.types.core import (
    AgentBudget,
    AgentDefinition,
    HookEvent,
    HookResult,
    LoopContext,
    PipelineConfig,
    PipelineStage,
    PipelineStageStatus,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(num_stages: int = 3) -> PipelineConfig:
    """Build a minimal PipelineConfig with N stages."""
    stages = []
    for i in range(num_stages):
        name = ["analysis", "planning", "execution"][i] if i < 3 else f"stage{i}"
        stages.append(
            PipelineStage(
                name=name,
                agent_def_path=f".brainmass/agents/{name}.md",
                input_mapping={} if i == 0 else {"result": "input"},
                output_mapping={"result": "result"},
                max_iterations=3,
                acceptance_criteria=[f"{name} done"],
                constraints=[],
            )
        )
    return PipelineConfig(name="test-pipeline", stages=stages, description="Test pipeline")


def _make_callback(outputs: list[dict] | None = None, fail_stage: str | None = None):
    """Return a stage callback that yields predefined outputs."""
    call_count = [0]

    def callback(agent_def: AgentDefinition, loop_ctx: LoopContext, budget: AgentBudget) -> dict:
        idx = call_count[0]
        call_count[0] += 1
        out = outputs[idx] if outputs and idx < len(outputs) else {"result": f"output-{idx}"}
        acceptance_met = agent_def.name != fail_stage
        return {
            "summary": f"{agent_def.name} done",
            "output": out,
            "tokens_consumed": {"input": 100, "output": 50, "cache_read": 0},
            "tools_used": ["read_file"],
            "files_modified": [],
            "exit_reason": "complete",
            "turns_used": 2,
            "acceptance_met": acceptance_met,
        }

    return callback


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class TestLoadPipelineConfigFromDict:
    def test_basic_three_stage_pipeline(self):
        data = {
            "name": "my-pipeline",
            "description": "Analysis → Planning → Execution",
            "stages": [
                {
                    "name": "analysis",
                    "agent_def_path": ".brainmass/agents/analyzer.md",
                    "max_iterations": 3,
                    "acceptance_criteria": ["task list produced"],
                    "constraints": ["read-only"],
                    "input_mapping": {},
                    "output_mapping": {"task_list": "task_list"},
                },
                {
                    "name": "planning",
                    "agent_def_path": ".brainmass/agents/planner.md",
                    "max_iterations": 5,
                    "acceptance_criteria": ["PRD produced"],
                    "input_mapping": {"task_list": "task_list"},
                    "output_mapping": {"prd": "prd"},
                },
            ],
        }
        config = load_pipeline_config_from_dict(data)

        assert config.name == "my-pipeline"
        assert config.description == "Analysis → Planning → Execution"
        assert len(config.stages) == 2

        analysis = config.stages[0]
        assert analysis.name == "analysis"
        assert analysis.max_iterations == 3
        assert analysis.acceptance_criteria == ["task list produced"]
        assert analysis.constraints == ["read-only"]
        assert analysis.input_mapping == {}
        assert analysis.output_mapping == {"task_list": "task_list"}

        planning = config.stages[1]
        assert planning.name == "planning"
        assert planning.input_mapping == {"task_list": "task_list"}

    def test_defaults_applied_when_optional_fields_absent(self):
        data = {
            "name": "minimal",
            "stages": [{"name": "step1", "agent_def_path": "agents/step1.md"}],
        }
        config = load_pipeline_config_from_dict(data)
        stage = config.stages[0]

        assert stage.max_iterations == 5
        assert stage.acceptance_criteria == []
        assert stage.constraints == []
        assert stage.input_mapping == {}
        assert stage.output_mapping == {}
        assert config.description == ""

    def test_empty_stages_list(self):
        config = load_pipeline_config_from_dict({"name": "empty", "stages": []})
        assert config.stages == []


# ---------------------------------------------------------------------------
# YAML config loading
# ---------------------------------------------------------------------------

class TestLoadPipelineConfigFromYaml:
    def test_load_from_yaml_file(self, tmp_path):
        yaml_content = """
name: yaml-pipeline
description: "YAML test"
stages:
  - name: analysis
    agent_def_path: agents/analyzer.md
    max_iterations: 2
    acceptance_criteria:
      - "done"
    input_mapping: {}
    output_mapping:
      result: result
"""
        yaml_file = tmp_path / "pipeline.yaml"
        yaml_file.write_text(yaml_content)

        from src.agents.compound_loop import load_pipeline_config
        config = load_pipeline_config(str(yaml_file))

        assert config.name == "yaml-pipeline"
        assert len(config.stages) == 1
        assert config.stages[0].name == "analysis"
        assert config.stages[0].max_iterations == 2


# ---------------------------------------------------------------------------
# Input / output mapping
# ---------------------------------------------------------------------------

class TestInputOutputMapping:
    def test_map_input_with_explicit_mapping(self):
        prev_output = {"task_list": ["task1", "task2"], "metadata": "ignored"}
        mapping = {"task_list": "tasks"}
        result = CompoundLoopOrchestrator._map_input(prev_output, mapping)
        assert result == {"tasks": ["task1", "task2"]}
        assert "metadata" not in result

    def test_map_input_empty_mapping_passes_all(self):
        prev_output = {"a": 1, "b": 2}
        result = CompoundLoopOrchestrator._map_input(prev_output, {})
        assert result == {"a": 1, "b": 2}

    def test_map_input_missing_key_is_skipped(self):
        prev_output = {"x": 10}
        mapping = {"x": "y", "missing": "z"}
        result = CompoundLoopOrchestrator._map_input(prev_output, mapping)
        assert result == {"y": 10}
        assert "z" not in result

    def test_map_output_with_explicit_mapping(self):
        stage_output = {"prd": "doc", "subtasks": ["t1"]}
        mapping = {"prd": "plan_doc", "subtasks": "tasks"}
        result = CompoundLoopOrchestrator._map_output(stage_output, mapping)
        assert result == {"plan_doc": "doc", "tasks": ["t1"]}

    def test_map_output_empty_mapping_passes_all(self):
        stage_output = {"a": 1, "b": 2}
        result = CompoundLoopOrchestrator._map_output(stage_output, {})
        assert result == {"a": 1, "b": 2}

    def test_map_output_missing_key_is_skipped(self):
        stage_output = {"result": "ok"}
        mapping = {"result": "out", "nonexistent": "x"}
        result = CompoundLoopOrchestrator._map_output(stage_output, mapping)
        assert result == {"out": "ok"}


# ---------------------------------------------------------------------------
# Pipeline state persistence
# ---------------------------------------------------------------------------

class TestPipelineStatePersistence:
    def test_write_and_read_pipeline_state(self, tmp_path):
        state_path = str(tmp_path / ".brainmass" / "pipeline-state.json")
        config = _make_config(2)
        orch = CompoundLoopOrchestrator(
            config=config,
            pipeline_state_path=state_path,
        )

        state = PipelineState(
            pipeline_name="test",
            current_stage="analysis",
            stage_statuses={"analysis": "running", "planning": "pending"},
            stage_outputs={},
            started_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:01:00+00:00",
        )
        orch._write_pipeline_state(state)

        loaded = CompoundLoopOrchestrator.read_pipeline_state(state_path)
        assert loaded is not None
        assert loaded.pipeline_name == "test"
        assert loaded.current_stage == "analysis"
        assert loaded.stage_statuses["analysis"] == "running"

    def test_read_nonexistent_returns_none(self, tmp_path):
        result = CompoundLoopOrchestrator.read_pipeline_state(
            str(tmp_path / "nonexistent.json")
        )
        assert result is None

    def test_read_invalid_json_returns_none(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not valid json")
        result = CompoundLoopOrchestrator.read_pipeline_state(str(bad_file))
        assert result is None

    def test_pipeline_state_to_dict_round_trip(self):
        state = PipelineState(
            pipeline_name="p",
            current_stage="s1",
            stage_statuses={"s1": "complete"},
            stage_outputs={"s1": {"result": "ok"}},
            started_at="2024-01-01T00:00:00+00:00",
            updated_at="2024-01-01T00:01:00+00:00",
            exit_reason="complete",
        )
        d = state.to_dict()
        restored = PipelineState.from_dict(d)
        assert restored.pipeline_name == state.pipeline_name
        assert restored.current_stage == state.current_stage
        assert restored.stage_statuses == state.stage_statuses
        assert restored.stage_outputs == state.stage_outputs
        assert restored.exit_reason == state.exit_reason


# ---------------------------------------------------------------------------
# SubagentStop hook firing
# ---------------------------------------------------------------------------

class TestSubagentStopHooks:
    def test_subagent_stop_fired_after_each_stage(self, tmp_path):
        config = _make_config(3)
        mock_hook_engine = MagicMock()
        mock_hook_engine.fire.return_value = HookResult()

        orch = CompoundLoopOrchestrator(
            config=config,
            stage_callback=_make_callback(),
            hook_engine=mock_hook_engine,
            pipeline_state_path=str(tmp_path / ".brainmass" / "pipeline-state.json"),
        )
        orch.run()

        # One SubagentStop call per stage
        assert mock_hook_engine.fire.call_count == 3
        for c in mock_hook_engine.fire.call_args_list:
            event_arg = c.args[0]
            assert event_arg == HookEvent.SUBAGENT_STOP

    def test_subagent_stop_not_called_without_hook_engine(self, tmp_path):
        config = _make_config(2)
        orch = CompoundLoopOrchestrator(
            config=config,
            stage_callback=_make_callback(),
            hook_engine=None,
            pipeline_state_path=str(tmp_path / ".brainmass" / "pipeline-state.json"),
        )
        # Should not raise
        result = orch.run()
        assert result.exit_reason == "complete"

    def test_hook_context_source_is_stage_name(self, tmp_path):
        config = _make_config(1)
        mock_hook_engine = MagicMock()
        mock_hook_engine.fire.return_value = HookResult()

        orch = CompoundLoopOrchestrator(
            config=config,
            stage_callback=_make_callback(),
            hook_engine=mock_hook_engine,
            pipeline_state_path=str(tmp_path / ".brainmass" / "pipeline-state.json"),
        )
        orch.run()

        ctx = mock_hook_engine.fire.call_args.args[1]
        assert ctx.source == "analysis"


# ---------------------------------------------------------------------------
# Full pipeline run
# ---------------------------------------------------------------------------

class TestPipelineRun:
    def test_three_stage_pipeline_completes(self, tmp_path):
        config = _make_config(3)
        outputs = [
            {"task_list": ["task1", "task2"]},
            {"prd": "PRD doc", "subtasks": ["sub1"]},
            {"status": "all done"},
        ]
        orch = CompoundLoopOrchestrator(
            config=config,
            stage_callback=_make_callback(outputs=outputs),
            pipeline_state_path=str(tmp_path / ".brainmass" / "pipeline-state.json"),
        )
        result = orch.run(initial_input={"task": "Build auth module"})

        assert result.exit_reason == "complete"
        assert result.succeeded
        assert len(result.stage_results) == 3
        assert all(r.status == PipelineStageStatus.COMPLETE for r in result.stage_results)

    def test_stage_failure_aborts_pipeline(self, tmp_path):
        config = _make_config(3)
        orch = CompoundLoopOrchestrator(
            config=config,
            stage_callback=_make_callback(fail_stage="planning"),
            pipeline_state_path=str(tmp_path / ".brainmass" / "pipeline-state.json"),
        )
        result = orch.run()

        assert result.exit_reason == "stage_failed"
        assert not result.succeeded
        # Only analysis and planning ran; execution was skipped
        assert len(result.stage_results) == 2
        assert result.stage_results[1].status == PipelineStageStatus.FAILED

    def test_output_chaining_between_stages(self, tmp_path):
        """Verify that stage N's output becomes stage N+1's input."""
        received_inputs: list[dict] = []

        def tracking_callback(
            agent_def: AgentDefinition,
            loop_ctx: LoopContext,
            budget: AgentBudget,
        ) -> dict:
            # Record what the stage received as its task description
            received_inputs.append({"stage": agent_def.name, "task": loop_ctx.current_task})
            return {
                "summary": "done",
                "output": {"result": f"{agent_def.name}-output"},
                "tokens_consumed": {"input": 0, "output": 0, "cache_read": 0},
                "tools_used": [],
                "files_modified": [],
                "exit_reason": "complete",
                "turns_used": 1,
                "acceptance_met": True,
            }

        config = PipelineConfig(
            name="chain-test",
            stages=[
                PipelineStage(
                    name="stage-a",
                    agent_def_path="",
                    input_mapping={},
                    output_mapping={"result": "prev_result"},
                    max_iterations=1,
                ),
                PipelineStage(
                    name="stage-b",
                    agent_def_path="",
                    input_mapping={"prev_result": "input_value"},
                    output_mapping={},
                    max_iterations=1,
                ),
            ],
        )
        orch = CompoundLoopOrchestrator(
            config=config,
            stage_callback=tracking_callback,
            pipeline_state_path=str(tmp_path / ".brainmass" / "pipeline-state.json"),
        )
        result = orch.run(initial_input={"seed": "hello"})

        assert result.exit_reason == "complete"
        # stage-b should have received stage-a's output in its task description
        stage_b_task = received_inputs[1]["task"]
        assert "stage-a-output" in stage_b_task

    def test_pipeline_state_written_for_each_stage(self, tmp_path):
        state_path = str(tmp_path / ".brainmass" / "pipeline-state.json")
        config = _make_config(2)
        orch = CompoundLoopOrchestrator(
            config=config,
            stage_callback=_make_callback(),
            pipeline_state_path=state_path,
        )
        orch.run()

        state = CompoundLoopOrchestrator.read_pipeline_state(state_path)
        assert state is not None
        assert state.pipeline_name == "test-pipeline"
        assert state.exit_reason == "complete"
        assert all(
            v == PipelineStageStatus.COMPLETE.value
            for v in state.stage_statuses.values()
        )

    def test_pipeline_state_records_stage_outputs(self, tmp_path):
        state_path = str(tmp_path / ".brainmass" / "pipeline-state.json")
        config = _make_config(1)
        orch = CompoundLoopOrchestrator(
            config=config,
            stage_callback=_make_callback(outputs=[{"result": "analysis-done"}]),
            pipeline_state_path=state_path,
        )
        orch.run()

        state = CompoundLoopOrchestrator.read_pipeline_state(state_path)
        assert state is not None
        assert state.stage_outputs["analysis"]["result"] == "analysis-done"

    def test_empty_pipeline_completes_immediately(self, tmp_path):
        config = PipelineConfig(name="empty", stages=[])
        orch = CompoundLoopOrchestrator(
            config=config,
            pipeline_state_path=str(tmp_path / ".brainmass" / "pipeline-state.json"),
        )
        result = orch.run()
        assert result.exit_reason == "complete"
        assert result.stage_results == []

    def test_stage_exception_marks_stage_failed(self, tmp_path):
        def crashing_callback(agent_def, loop_ctx, budget):
            raise RuntimeError("Unexpected error in stage")

        config = _make_config(2)
        orch = CompoundLoopOrchestrator(
            config=config,
            stage_callback=crashing_callback,
            pipeline_state_path=str(tmp_path / ".brainmass" / "pipeline-state.json"),
        )
        result = orch.run()

        assert result.exit_reason == "stage_failed"
        assert result.stage_results[0].status == PipelineStageStatus.FAILED
        assert "Unexpected error" in result.stage_results[0].error

    def test_initial_input_passed_to_first_stage(self, tmp_path):
        received: list[str] = []

        def capturing_callback(agent_def, loop_ctx, budget):
            received.append(loop_ctx.current_task)
            return {
                "summary": "done",
                "output": {},
                "tokens_consumed": {"input": 0, "output": 0, "cache_read": 0},
                "tools_used": [],
                "files_modified": [],
                "exit_reason": "complete",
                "turns_used": 1,
                "acceptance_met": True,
            }

        config = _make_config(1)
        orch = CompoundLoopOrchestrator(
            config=config,
            stage_callback=capturing_callback,
            pipeline_state_path=str(tmp_path / ".brainmass" / "pipeline-state.json"),
        )
        orch.run(initial_input={"task": "Build auth module"})

        assert "Build auth module" in received[0]


# ---------------------------------------------------------------------------
# Cost governor integration
# ---------------------------------------------------------------------------

class TestCostGovernorIntegration:
    def test_record_usage_called_for_each_stage(self, tmp_path):
        mock_governor = MagicMock()
        config = _make_config(2)
        orch = CompoundLoopOrchestrator(
            config=config,
            stage_callback=_make_callback(),
            cost_governor=mock_governor,
            pipeline_state_path=str(tmp_path / ".brainmass" / "pipeline-state.json"),
        )
        orch.run()

        assert mock_governor.record_usage.call_count == 2

    def test_no_record_usage_when_zero_tokens(self, tmp_path):
        mock_governor = MagicMock()

        def zero_token_callback(agent_def, loop_ctx, budget):
            return {
                "summary": "done",
                "output": {},
                "tokens_consumed": {"input": 0, "output": 0, "cache_read": 0},
                "tools_used": [],
                "files_modified": [],
                "exit_reason": "complete",
                "turns_used": 0,
                "acceptance_met": True,
            }

        config = _make_config(1)
        orch = CompoundLoopOrchestrator(
            config=config,
            stage_callback=zero_token_callback,
            cost_governor=mock_governor,
            pipeline_state_path=str(tmp_path / ".brainmass" / "pipeline-state.json"),
        )
        orch.run()

        mock_governor.record_usage.assert_not_called()


# ---------------------------------------------------------------------------
# Default stage callback
# ---------------------------------------------------------------------------

class TestDefaultStageCallback:
    def test_returns_valid_result_dict(self):
        agent_def = AgentDefinition(name="test", description="test agent")
        loop_ctx = LoopContext(
            current_task="do something",
            acceptance_criteria=[],
            constraints=[],
            learnings=[],
            failed_approaches=[],
            iteration_count=0,
            max_iterations=3,
        )
        budget = AgentBudget(
            input_budget_tokens=100_000,
            output_budget_tokens=10_000,
            session_budget_usd=1.0,
        )
        result = _default_stage_callback(agent_def, loop_ctx, budget)

        assert "summary" in result
        assert "output" in result
        assert "tokens_consumed" in result
        assert "acceptance_met" in result
