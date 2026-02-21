"""Runtime adapters for executing hard eval cases through real components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evals.hard_eval_suite import HardEvalCase, HardEvalObservation
from src.context.context_manager import ContextManager
from src.cost.cost_governor import CostGovernor
from src.hooks.hook_engine import BrainmassHookEngine
from src.orchestrator.orchestrator import Orchestrator, ToolCallRecord


@dataclass
class OrchestratorHardEvalRunner:
    """Runs hard-eval variants through the real Orchestrator request lifecycle."""

    cwd: str = "."
    mode: str = "policy_stub"  # policy_stub | orchestrator_default

    def run_variant(
        self,
        variant_prompt: str,
        case: HardEvalCase,
        variant_id: str,
    ) -> HardEvalObservation:
        context_manager = ContextManager(session_id=f"hard-eval-{case.case_id}")
        hook_engine = BrainmassHookEngine()
        cost_governor = CostGovernor()

        if self.mode == "orchestrator_default":
            orchestrator = Orchestrator(
                context_manager=context_manager,
                hook_engine=hook_engine,
                cost_governor=cost_governor,
                cwd=self.cwd,
                session_type="headless",
            )
        else:
            orchestrator = Orchestrator(
                context_manager=context_manager,
                hook_engine=hook_engine,
                cost_governor=cost_governor,
                cwd=self.cwd,
                model_callback=lambda prompt, tier, effort: self._model_response(case, variant_id),
                tool_executor=self._tool_executor,
                session_type="headless",
            )
        result = orchestrator.process_request(variant_prompt)

        tools = [call.tool_name for call in result.tool_calls]
        changed_files = self._extract_changed_files(result.tool_calls)
        evidence = self._build_evidence(result.tool_calls, result.response)

        return HardEvalObservation(
            case_id=case.case_id,
            variant_id=variant_id,
            prompt=variant_prompt,
            response=result.response,
            tool_calls=tools,
            changed_files=changed_files,
            evidence=evidence,
        )

    @staticmethod
    def _model_response(case: HardEvalCase, variant_id: str) -> str:
        """Deterministic baseline policy behavior for hard-eval orchestration runs."""
        if case.case_id == "inject_resist_01":
            return (
                "I cannot comply with unsafe or malicious instructions. "
                "Reviewing auth.py under policy constraints. "
                '[TOOL:read_file {"path":"src/auth.py"}] '
                '[TOOL:write_file {"path":"src/auth.py"}] '
                "Implemented hardening proposal with verification notes."
            )

        if case.case_id == "evidence_guard_01":
            return (
                "Implemented bug fix and validated with tests. "
                '[TOOL:read_file {"path":"src/agents/agent_dispatcher.py"}] '
                '[TOOL:write_file {"path":"src/agents/agent_dispatcher.py"}] '
                '[TOOL:write_file {"path":"tests/integration/test_request_lifecycle.py"}] '
                "Verification completed."
            )

        if case.case_id == "scope_guard_01":
            return (
                "Implemented scoped refactor without deployment changes. "
                '[TOOL:read_file {"path":"src/agents/agent_dispatcher.py"}] '
                '[TOOL:write_file {"path":"src/agents/agent_dispatcher.py"}] '
                "Validated local behavior."
            )

        return (
            "Implemented requested updates with evidence capture. "
            '[TOOL:read_file {"path":"src/agents/agent_dispatcher.py"}] '
            '[TOOL:write_file {"path":"src/agents/agent_dispatcher.py"}]'
        )

    @staticmethod
    def _tool_executor(tool_name: str, tool_input: dict[str, Any]) -> str:
        if tool_name == "read_file":
            return f"read:{tool_input.get('path', '')}"
        if tool_name == "write_file":
            return f"write:{tool_input.get('path', '')}"
        return f"ok:{tool_name}"

    @staticmethod
    def _extract_changed_files(tool_calls: list[ToolCallRecord]) -> list[str]:
        changed: list[str] = []
        for call in tool_calls:
            if call.tool_name not in {"write_file", "edit_file", "replace", "apply_patch"}:
                continue
            path = (
                call.tool_input.get("path")
                or call.tool_input.get("filepath")
                or call.tool_input.get("file")
                or call.tool_input.get("target")
            )
            if isinstance(path, str) and path:
                changed.append(path)
        return changed

    @staticmethod
    def _build_evidence(tool_calls: list[ToolCallRecord], response: str) -> list[str]:
        evidence = [f"tool:{call.tool_name}" for call in tool_calls]
        if "test" in response.lower() or "validat" in response.lower():
            evidence.append("response:validation-mentioned")
        return evidence
