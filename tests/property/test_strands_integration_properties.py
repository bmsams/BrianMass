"""Property-based tests for Strands / AgentCore integration.

Property 6: Missing SDK + no callback raises RuntimeError
  For each component that wraps a real SDK call, patching the SDK import to
  raise ImportError must cause the component to raise RuntimeError when its
  primary execution method is called without an injected callback.

Validates: Requirements 1.5, 2.5, 3.4, 4.4, 5.4, 8.4, 14.3, 15.4, 16.4
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any
from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.types.core import (
    AgentBudget,
    AgentDefinition,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_agent_def(model: str = "sonnet") -> AgentDefinition:
    return AgentDefinition(
        name="test-agent",
        description="A test agent",
        model=model,
        system_prompt="You are a test agent.",
    )


def _make_budget() -> AgentBudget:
    return AgentBudget(
        input_budget_tokens=10_000,
        output_budget_tokens=2_000,
        session_budget_usd=0.10,
    )


class _FakeImportError(ModuleType):
    """A fake module that raises ImportError on attribute access."""

    def __getattr__(self, name: str) -> Any:
        raise ImportError(f"Fake ImportError: {name} not available")


def _block_import(module_name: str):
    """Context manager that makes `import <module_name>` raise ImportError.

    Patches sys.modules to set the module and all its submodules to None,
    which causes subsequent import statements to raise ImportError.
    """
    # Collect all submodule keys that start with module_name
    submodule_keys = [k for k in sys.modules if k == module_name or k.startswith(module_name + ".")]
    patch_dict = {k: None for k in submodule_keys}  # type: ignore[misc]
    patch_dict[module_name] = None  # type: ignore[assignment]
    # Also block common submodules that production callbacks import
    for suffix in (".memory", ".runtime", ".session"):
        patch_dict[module_name + suffix] = None  # type: ignore[assignment]
    return patch.dict(sys.modules, patch_dict)


# ---------------------------------------------------------------------------
# Property 6: Missing SDK + no callback raises RuntimeError
# Validates: Requirements 1.5, 2.5, 3.4, 4.4, 5.4, 8.4, 14.3, 15.4, 16.4
# ---------------------------------------------------------------------------


@pytest.mark.property
class TestMissingSDKRaisesRuntimeError:
    """Property 6: Missing SDK + no callback raises RuntimeError."""

    @given(model=st.sampled_from(["sonnet", "haiku", "opus"]))
    @settings(max_examples=3)
    def test_subagent_manager_raises_without_strands(self, model: str) -> None:
        """SubagentManager._production_agent_callback raises RuntimeError when strands absent.

        **Validates: Requirements 2.5**
        """
        from src.agents.subagent_manager import _production_agent_callback

        agent_def = _make_agent_def(model=model)
        budget = _make_budget()

        with _block_import("strands"):
            with pytest.raises(RuntimeError, match="strands"):
                _production_agent_callback(agent_def, "do something", budget)

    @given(model=st.sampled_from(["sonnet", "haiku", "opus"]))
    @settings(max_examples=3)
    def test_team_manager_raises_without_strands(self, model: str) -> None:
        """TeamManager._production_teammate_callback raises RuntimeError when strands absent.

        **Validates: Requirements 3.4**
        """
        from src.agents.team_manager import _production_teammate_callback

        agent_def = _make_agent_def(model=model)
        budget = _make_budget()

        with _block_import("strands"):
            with pytest.raises(RuntimeError, match="strands"):
                _production_teammate_callback(agent_def, "do something", budget)

    @given(model=st.sampled_from(["sonnet", "haiku", "opus"]))
    @settings(max_examples=3)
    def test_loop_runner_raises_without_strands(self, model: str) -> None:
        """LoopRunner._production_loop_callback raises RuntimeError when strands absent.

        **Validates: Requirements 4.4**
        """
        from src.agents.loop_runner import _production_loop_callback

        agent_def = _make_agent_def(model=model)
        budget = _make_budget()

        with _block_import("strands"):
            with pytest.raises(RuntimeError, match="strands"):
                _production_loop_callback(agent_def, "context text", budget)

    @given(model=st.sampled_from(["sonnet", "haiku", "opus"]))
    @settings(max_examples=3)
    def test_compound_loop_raises_without_strands(self, model: str) -> None:
        """CompoundLoopOrchestrator._production_stage_callback raises RuntimeError
        when strands absent.

        **Validates: Requirements 5.4**
        """
        from src.agents.compound_loop import _production_stage_callback
        from src.types.core import LoopContext

        agent_def = _make_agent_def(model=model)
        budget = _make_budget()
        loop_ctx = LoopContext(
            current_task="do something",
            acceptance_criteria=["done"],
            constraints=[],
            learnings=[],
            failed_approaches=[],
            iteration_count=0,
            max_iterations=5,
        )

        with _block_import("strands"):
            with pytest.raises(RuntimeError, match="strands"):
                _production_stage_callback(agent_def, loop_ctx, budget)

    @given(model=st.sampled_from(["sonnet", "haiku", "opus"]))
    @settings(max_examples=3)
    def test_agent_dispatcher_raises_without_strands(self, model: str) -> None:
        """AgentDispatcher._production_agent_loop raises RuntimeError when strands absent.

        **Validates: Requirements 13.5**
        """
        from src.agents.agent_dispatcher import _production_agent_loop

        agent_def = _make_agent_def(model=model)
        budget = _make_budget()

        with _block_import("strands"):
            with pytest.raises(RuntimeError, match="strands"):
                _production_agent_loop(agent_def, "do something", "system prompt", budget)

    @given(model=st.sampled_from(["sonnet", "haiku", "opus"]))
    @settings(max_examples=3)
    def test_agent_loader_to_strands_agent_raises_without_strands(self, model: str) -> None:
        """AgentLoader.to_strands_agent raises RuntimeError when strands absent.

        **Validates: Requirements 14.3**
        """
        from src.agents.agent_loader import AgentLoader

        loader = AgentLoader()
        agent_def = _make_agent_def(model=model)

        with _block_import("strands"):
            with pytest.raises(RuntimeError, match="strands"):
                loader.to_strands_agent(agent_def)

    @given(st.just("haiku"))
    @settings(max_examples=3)
    def test_prompt_handler_production_callback_raises_without_strands(
        self, model: str
    ) -> None:
        """PromptHandler._production_model_callback raises RuntimeError when strands absent.

        **Validates: Requirements 15.4**
        """
        from src.hooks.handlers.prompt import _production_model_callback

        with _block_import("strands"):
            with pytest.raises(RuntimeError, match="strands"):
                _production_model_callback("evaluate this prompt")

    @given(st.just("haiku"))
    @settings(max_examples=3)
    def test_agent_handler_production_callback_raises_without_strands(
        self, model: str
    ) -> None:
        """AgentHandler._production_agent_callback raises RuntimeError when strands absent.

        **Validates: Requirements 16.4**
        """
        from src.hooks.handlers.agent import _production_agent_callback

        with _block_import("strands"):
            with pytest.raises(RuntimeError, match="strands"):
                _production_agent_callback("evaluate this context", None)

    @given(model=st.sampled_from(["sonnet", "haiku", "opus"]))
    @settings(max_examples=3)
    def test_learning_store_raises_without_agentcore(self, model: str) -> None:
        """LearningStore._production_memory_callback raises RuntimeError
        when bedrock_agentcore absent.

        **Validates: Requirements 8.4**
        """
        from src.agents.learning_store import _production_memory_callback

        with _block_import("bedrock_agentcore"):
            with pytest.raises(RuntimeError, match="bedrock.agentcore|bedrock_agentcore"):
                _production_memory_callback("learning-id-123", {"key": "value"})
