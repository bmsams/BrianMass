"""Unit tests for core type definitions."""

from datetime import UTC, datetime

from src.types import (
    PRICING,
    AgentBudget,
    AgentDefinition,
    AgentResult,
    BudgetStatus,
    ContextCategory,
    ContextHealthMetrics,
    ContextItem,
    EffortLevel,
    HookContext,
    HookDefinition,
    HookEvent,
    HookHandler,
    HookHandlerType,
    HookResult,
    Learning,
    LoopContext,
    MailboxMessage,
    McpServerConfig,
    ModelPricing,
    ModelTier,
    PluginManifest,
    SessionState,
    SkillDefinition,
    TeamTask,
    TopologyType,
)

# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------

class TestModelTier:
    def test_values(self):
        assert ModelTier.OPUS.value == "opus"
        assert ModelTier.SONNET.value == "sonnet"
        assert ModelTier.HAIKU.value == "haiku"

    def test_member_count(self):
        assert len(ModelTier) == 3


class TestEffortLevel:
    def test_values(self):
        assert EffortLevel.QUICK.value == "quick"
        assert EffortLevel.STANDARD.value == "standard"
        assert EffortLevel.DEEP.value == "deep"

    def test_member_count(self):
        assert len(EffortLevel) == 3


class TestContextCategory:
    def test_values(self):
        assert ContextCategory.PRESERVE_VERBATIM.value == "PRESERVE_VERBATIM"
        assert ContextCategory.PRESERVE_STRUCTURED.value == "PRESERVE_STRUCTURED"
        assert ContextCategory.COMPRESS_AGGRESSIVE.value == "COMPRESS_AGGRESSIVE"
        assert ContextCategory.EPHEMERAL.value == "EPHEMERAL"

    def test_member_count(self):
        assert len(ContextCategory) == 4


class TestHookEvent:
    def test_all_12_events(self):
        expected = [
            ("SESSION_START", "SessionStart"),
            ("USER_PROMPT_SUBMIT", "UserPromptSubmit"),
            ("PRE_TOOL_USE", "PreToolUse"),
            ("POST_TOOL_USE", "PostToolUse"),
            ("POST_TOOL_USE_FAILURE", "PostToolUseFailure"),
            ("PERMISSION_REQUEST", "PermissionRequest"),
            ("PRE_COMPACT", "PreCompact"),
            ("NOTIFICATION", "Notification"),
            ("STOP", "Stop"),
            ("SUBAGENT_STOP", "SubagentStop"),
            ("SESSION_END", "SessionEnd"),
            ("SETUP", "Setup"),
        ]
        for name, value in expected:
            assert HookEvent[name].value == value

    def test_member_count(self):
        assert len(HookEvent) == 12


class TestHookHandlerType:
    def test_values(self):
        assert HookHandlerType.COMMAND.value == "command"
        assert HookHandlerType.PROMPT.value == "prompt"
        assert HookHandlerType.AGENT.value == "agent"


class TestBudgetStatus:
    def test_values(self):
        assert BudgetStatus.OK.value == "ok"
        assert BudgetStatus.WARNING.value == "warning"
        assert BudgetStatus.CRITICAL.value == "critical"
        assert BudgetStatus.EXCEEDED.value == "exceeded"


class TestTopologyType:
    def test_values(self):
        assert TopologyType.HIERARCHICAL.value == "hierarchical"
        assert TopologyType.AGENT_TEAMS.value == "agent_teams"
        assert TopologyType.SELF_IMPROVING_LOOP.value == "self_improving_loop"
        assert TopologyType.PIPELINE.value == "pipeline"

    def test_member_count(self):
        assert len(TopologyType) == 4


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------

class TestContextItem:
    def test_creation(self):
        now = datetime.now(tz=UTC)
        item = ContextItem(
            id="ctx-1",
            category=ContextCategory.PRESERVE_VERBATIM,
            content="Error: file not found",
            token_count=10,
            created_at=now,
            last_referenced_at=now,
            reference_count=1,
            source="tool_result",
        )
        assert item.id == "ctx-1"
        assert item.category == ContextCategory.PRESERVE_VERBATIM
        assert item.staleness_score == 0.0

    def test_staleness_default(self):
        now = datetime.now(tz=UTC)
        item = ContextItem(
            id="x", category=ContextCategory.EPHEMERAL, content="",
            token_count=0, created_at=now, last_referenced_at=now,
            reference_count=1, source="system",
        )
        assert item.staleness_score == 0.0


class TestContextHealthMetrics:
    def test_creation(self):
        metrics = ContextHealthMetrics(
            free_percent=45.0,
            total_tokens=110000,
            preserved_tokens=30000,
            compressible_tokens=50000,
            ephemeral_tokens=30000,
            staleness_distribution={"0-5": 100, "5-10": 50, "10+": 20},
            cache_hit_rate=0.75,
        )
        assert metrics.free_percent == 45.0
        assert metrics.total_tokens == 110000


class TestAgentBudget:
    def test_status_ok(self):
        b = AgentBudget(100000, 50000, 10.0, current_cost_usd=7.99)
        assert b.status == BudgetStatus.OK

    def test_status_warning_at_80_percent(self):
        b = AgentBudget(100000, 50000, 10.0, current_cost_usd=8.0)
        assert b.status == BudgetStatus.WARNING

    def test_status_critical_at_95_percent(self):
        b = AgentBudget(100000, 50000, 10.0, current_cost_usd=9.5)
        assert b.status == BudgetStatus.CRITICAL

    def test_status_exceeded_at_100_percent(self):
        b = AgentBudget(100000, 50000, 10.0, current_cost_usd=10.0)
        assert b.status == BudgetStatus.EXCEEDED

    def test_status_exceeded_over_budget(self):
        b = AgentBudget(100000, 50000, 10.0, current_cost_usd=15.0)
        assert b.status == BudgetStatus.EXCEEDED

    def test_status_zero_budget(self):
        b = AgentBudget(100000, 50000, 0.0, current_cost_usd=5.0)
        assert b.status == BudgetStatus.OK

    def test_status_no_spend(self):
        b = AgentBudget(100000, 50000, 10.0, current_cost_usd=0.0)
        assert b.status == BudgetStatus.OK


class TestHookHandler:
    def test_defaults(self):
        h = HookHandler(type=HookHandlerType.COMMAND, command="echo hello")
        assert h.is_async is False
        assert h.timeout == 600000
        assert h.prompt is None
        assert h.agent_config is None


class TestHookDefinition:
    def test_defaults(self):
        d = HookDefinition()
        assert d.matcher is None
        assert d.hooks == []


class TestHookContext:
    def test_creation(self):
        ctx = HookContext(
            session_id="sess-1",
            hook_event_name=HookEvent.PRE_TOOL_USE,
            cwd="/project",
            session_type="interactive",
            tool_name="bash",
        )
        assert ctx.session_id == "sess-1"
        assert ctx.hook_event_name == HookEvent.PRE_TOOL_USE
        assert ctx.tool_input is None


class TestHookResult:
    def test_all_none_defaults(self):
        r = HookResult()
        assert r.permission_decision is None
        assert r.updated_input is None
        assert r.decision is None


class TestAgentDefinition:
    def test_defaults(self):
        a = AgentDefinition(name="code-reviewer", description="Reviews code")
        assert a.model == "inherit"
        assert a.tools is None
        assert a.permission_mode == "default"
        assert a.hooks == {}
        assert a.skills == []
        assert a.mcp_servers == {}
        assert a.source == "project"
        assert a.system_prompt == ""


class TestAgentResult:
    def test_creation(self):
        r = AgentResult(
            agent_name="reviewer",
            summary="Found 3 issues",
            turns_used=5,
            tokens_consumed={"input": 1000, "output": 500, "cache_read": 200},
            tools_used=["read_file", "grep"],
            files_modified=[],
            exit_reason="complete",
        )
        assert r.exit_reason == "complete"
        assert len(r.tools_used) == 2


class TestTeamTask:
    def test_creation(self):
        t = TeamTask(
            id="task-1", title="Fix auth", assignee="agent-1",
            status="pending", dependencies=[], files=["auth.py"],
        )
        assert t.status == "pending"


class TestMailboxMessage:
    def test_creation(self):
        now = datetime.now(tz=UTC)
        m = MailboxMessage(
            sender="lead", recipient="agent-1",
            type="task_assignment", payload={"task_id": "1"}, timestamp=now,
        )
        assert m.type == "task_assignment"


class TestLoopContext:
    def test_creation(self):
        lc = LoopContext(
            current_task="Implement auth",
            acceptance_criteria=["Tests pass"],
            constraints=["No external deps"],
            learnings=[],
            failed_approaches=[],
            iteration_count=0,
            max_iterations=10,
        )
        assert lc.max_iterations == 10


class TestLearning:
    def test_defaults(self):
        l = Learning(pattern="retry on 429", resolution="add backoff", confidence=0.9, source_iteration=3)
        assert l.embedding is None


class TestPluginManifest:
    def test_creation(self):
        p = PluginManifest(
            name="my-plugin", description="A plugin", version="1.0.0",
            author={"name": "Dev"},
        )
        assert p.homepage is None
        assert p.license is None


class TestSkillDefinition:
    def test_defaults(self):
        s = SkillDefinition(name="owasp", description="OWASP checks")
        assert s.disable_model_invocation is False
        assert s.content == ""


class TestModelPricing:
    def test_creation(self):
        p = ModelPricing(5.0, 25.0, 0.5, 6.25)
        assert p.input_per_million == 5.0


class TestPricingConstants:
    def test_all_tiers_present(self):
        assert ModelTier.OPUS in PRICING
        assert ModelTier.SONNET in PRICING
        assert ModelTier.HAIKU in PRICING

    def test_opus_pricing(self):
        p = PRICING[ModelTier.OPUS]
        assert p.input_per_million == 5.00
        assert p.output_per_million == 25.00
        assert p.cached_input_per_million == 0.50
        assert p.cache_write_per_million == 6.25

    def test_sonnet_pricing(self):
        p = PRICING[ModelTier.SONNET]
        assert p.input_per_million == 3.00
        assert p.output_per_million == 15.00
        assert p.cached_input_per_million == 0.30
        assert p.cache_write_per_million == 3.75

    def test_haiku_pricing(self):
        p = PRICING[ModelTier.HAIKU]
        assert p.input_per_million == 0.80
        assert p.output_per_million == 4.00
        assert p.cached_input_per_million == 0.08
        assert p.cache_write_per_million == 1.00


class TestMcpServerConfig:
    def test_defaults(self):
        c = McpServerConfig(command="npx mcp-server")
        assert c.args == []
        assert c.env == {}
        assert c.scope == "project"


class TestSessionState:
    def test_creation(self):
        s = SessionState(
            conversation_history=[],
            tool_permissions={"bash": True},
            active_workers=[],
            pending_approvals=[],
            compaction_state={},
            context_manager_state={},
            cost_tracking={},
            hook_registrations={},
            trace_id="trace-abc",
        )
        assert s.trace_id == "trace-abc"
