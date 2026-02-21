"""Core type definitions for Brainmass v3 Enterprise System.

Defines all enums, dataclasses, and constants used across the system.
Requirements: 1.4, 4.6, 28.1
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ModelTier(Enum):
    """Three model tiers with different capability/cost trade-offs."""
    OPUS = "opus"
    SONNET = "sonnet"
    HAIKU = "haiku"


class EffortLevel(Enum):
    """Reasoning depth levels mapped to budget_tokens."""
    QUICK = "quick"        # budget_tokens: 2000
    STANDARD = "standard"  # budget_tokens: 10000
    DEEP = "deep"          # budget_tokens: 50000+


class ContextCategory(Enum):
    """Semantic triage classifications for context items."""
    PRESERVE_VERBATIM = "PRESERVE_VERBATIM"
    PRESERVE_STRUCTURED = "PRESERVE_STRUCTURED"
    COMPRESS_AGGRESSIVE = "COMPRESS_AGGRESSIVE"
    EPHEMERAL = "EPHEMERAL"


class HookEvent(Enum):
    """12 lifecycle events fired by the Hook Engine."""
    SESSION_START = "SessionStart"
    USER_PROMPT_SUBMIT = "UserPromptSubmit"
    PRE_TOOL_USE = "PreToolUse"
    POST_TOOL_USE = "PostToolUse"
    POST_TOOL_USE_FAILURE = "PostToolUseFailure"
    PERMISSION_REQUEST = "PermissionRequest"
    PRE_COMPACT = "PreCompact"
    NOTIFICATION = "Notification"
    STOP = "Stop"
    SUBAGENT_STOP = "SubagentStop"
    SESSION_END = "SessionEnd"
    SETUP = "Setup"


class HookHandlerType(Enum):
    """Three handler types for hook execution."""
    COMMAND = "command"
    PROMPT = "prompt"
    AGENT = "agent"


class BudgetStatus(Enum):
    """Budget utilization status thresholds."""
    OK = "ok"              # < 80%
    WARNING = "warning"    # >= 80%
    CRITICAL = "critical"  # >= 95%
    EXCEEDED = "exceeded"  # >= 100%


class TopologyType(Enum):
    """Multi-agent orchestration topologies."""
    HIERARCHICAL = "hierarchical"
    AGENT_TEAMS = "agent_teams"
    SELF_IMPROVING_LOOP = "self_improving_loop"
    PIPELINE = "pipeline"


# ---------------------------------------------------------------------------
# Context and Session Models
# ---------------------------------------------------------------------------

@dataclass
class ContextItem:
    """A single item in the agent's context window."""
    id: str
    category: ContextCategory
    content: str
    token_count: int
    created_at: datetime
    last_referenced_at: datetime
    reference_count: int
    source: str  # 'user' | 'assistant' | 'tool_call' | 'tool_result' | 'system'
    staleness_score: float = 0.0  # (turns_since_ref) * (1 / reference_count)


@dataclass
class ContextHealthMetrics:
    """Health metrics exposed by the Context Manager."""
    free_percent: float
    total_tokens: int
    preserved_tokens: int
    compressible_tokens: int
    ephemeral_tokens: int
    staleness_distribution: dict[str, int]
    cache_hit_rate: float


# ---------------------------------------------------------------------------
# Budget Model
# ---------------------------------------------------------------------------

@dataclass
class AgentBudget:
    """Per-agent token and cost budget with status tracking."""
    input_budget_tokens: int
    output_budget_tokens: int
    session_budget_usd: float
    team_budget_usd: float | None = None
    current_input_tokens: int = 0
    current_output_tokens: int = 0
    current_cost_usd: float = 0.0

    @property
    def status(self) -> BudgetStatus:
        """Compute budget status from current cost vs session budget."""
        if self.session_budget_usd <= 0:
            return BudgetStatus.OK
        pct = self.current_cost_usd / self.session_budget_usd
        if pct >= 1.0:
            return BudgetStatus.EXCEEDED
        if pct >= 0.95:
            return BudgetStatus.CRITICAL
        if pct >= 0.80:
            return BudgetStatus.WARNING
        return BudgetStatus.OK


# ---------------------------------------------------------------------------
# Hook Models
# ---------------------------------------------------------------------------

@dataclass
class HookHandler:
    """A single hook handler configuration."""
    type: HookHandlerType
    command: str | None = None       # for 'command' type
    prompt: str | None = None        # for 'prompt' type
    agent_config: dict | None = None  # for 'agent' type
    is_async: bool = False
    timeout: int = 600000  # 10 minutes in ms


@dataclass
class HookDefinition:
    """A hook definition with matcher and handlers."""
    matcher: str | None = None  # regex, '*' = match all
    hooks: list[HookHandler] = field(default_factory=list)


@dataclass
class HookContext:
    """Context passed to hook handlers during execution."""
    session_id: str
    hook_event_name: HookEvent
    cwd: str
    session_type: str  # 'interactive' | 'headless'
    tool_name: str | None = None
    tool_input: dict | None = None
    tool_response: str | None = None
    source: str | None = None  # for SessionStart: 'new' | 'resume' | 'teleport'
    model: str | None = None


@dataclass
class HookResult:
    """Result returned from hook handler execution."""
    permission_decision: str | None = None  # 'allow' | 'deny'
    permission_decision_reason: str | None = None
    updated_input: dict | None = None
    additional_context: str | None = None
    decision: str | None = None  # 'block' | 'continue' (for Stop event)
    reason: str | None = None


# ---------------------------------------------------------------------------
# Agent Models
# ---------------------------------------------------------------------------

@dataclass
class AgentDefinition:
    """Custom agent definition parsed from .md files with YAML frontmatter."""
    name: str                    # lowercase-with-hyphens
    description: str             # action-oriented with examples
    model: str = "inherit"       # sonnet | opus | haiku | inherit
    tools: list[str] | None = None
    disallowed_tools: list[str] | None = None
    permission_mode: str = "default"  # default | bypassPermissions | plan
    color: str | None = None
    max_turns: int | None = None
    hooks: dict[str, list[HookDefinition]] = field(default_factory=dict)
    skills: list[str] = field(default_factory=list)
    mcp_servers: dict[str, dict] = field(default_factory=dict)
    memory: str | None = None
    system_prompt: str = ""
    source: str = "project"      # project | user | plugin | cli
    plugin_namespace: str | None = None
    file_path: str = ""


@dataclass
class AgentResult:
    """Structured result returned when an agent completes execution."""
    agent_name: str
    summary: str
    turns_used: int
    tokens_consumed: dict  # {input, output, cache_read}
    tools_used: list[str]
    files_modified: list[str]
    exit_reason: str  # 'complete' | 'maxTurns' | 'budget' | 'error' | 'stopped'


# ---------------------------------------------------------------------------
# Team and Loop Models
# ---------------------------------------------------------------------------

@dataclass
class TeamTask:
    """A task in the Agent Teams shared task list."""
    id: str
    title: str
    assignee: str | None
    status: str  # 'pending' | 'claimed' | 'blocked' | 'complete'
    dependencies: list[str]
    files: list[str]


@dataclass
class MailboxMessage:
    """A message in the Agent Teams mailbox IPC system."""
    sender: str
    recipient: str
    type: str  # 'task_assignment' | 'finding' | 'question' | 'status_update'
    payload: dict
    timestamp: datetime


@dataclass
class LoopContext:
    """Structured context file for self-improving loop iterations."""
    current_task: str
    acceptance_criteria: list[str]
    constraints: list[str]
    learnings: list[dict]  # {pattern, resolution, confidence, source_iteration}
    failed_approaches: list[dict]  # {iteration, approach, why_failed}
    iteration_count: int
    max_iterations: int


@dataclass
class Learning:
    """A single learning entry in the cross-session Learning Store."""
    pattern: str
    resolution: str
    confidence: float
    source_iteration: int
    embedding: list[float] | None = None


# ---------------------------------------------------------------------------
# Pipeline / Compound Loop Models
# ---------------------------------------------------------------------------

class PipelineStageStatus(Enum):
    """Status of a single pipeline stage execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PipelineStage:
    """Configuration for a single stage in a compound loop pipeline.

    Requirements: 22.2
    """
    name: str                          # e.g. "analysis", "planning", "execution"
    agent_def_path: str                # path to the agent .md definition file
    input_mapping: dict                # keys from previous stage output → this stage's input keys
    output_mapping: dict               # this stage's output keys → keys passed to next stage
    max_iterations: int = 5            # max loop iterations for this stage
    acceptance_criteria: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)


@dataclass
class PipelineConfig:
    """Full pipeline configuration loaded from YAML.

    Requirements: 22.2, 22.4
    """
    name: str
    stages: list[PipelineStage]
    description: str = ""


# ---------------------------------------------------------------------------
# Plugin and Skill Models
# ---------------------------------------------------------------------------

@dataclass
class PluginManifest:
    """Plugin manifest from .brainmass-plugin/plugin.json."""
    name: str
    description: str
    version: str
    author: dict  # {name: str}
    homepage: str | None = None
    license: str | None = None


@dataclass
class SkillDefinition:
    """Skill definition parsed from SKILL.md frontmatter."""
    name: str
    description: str
    disable_model_invocation: bool = False
    hooks: dict[str, list[HookDefinition]] = field(default_factory=dict)
    allowed_tools: list[str] | None = None
    content: str = ""  # SKILL.md body


# ---------------------------------------------------------------------------
# Pricing Model and Constants
# ---------------------------------------------------------------------------

@dataclass
class ModelPricing:
    """Pricing rates per million tokens for a model tier."""
    input_per_million: float
    output_per_million: float
    cached_input_per_million: float
    cache_write_per_million: float


PRICING: dict[ModelTier, ModelPricing] = {
    ModelTier.OPUS: ModelPricing(5.00, 25.00, 0.50, 6.25),
    ModelTier.SONNET: ModelPricing(3.00, 15.00, 0.30, 3.75),
    ModelTier.HAIKU: ModelPricing(0.80, 4.00, 0.08, 1.00),
}


# ---------------------------------------------------------------------------
# Configuration Models
# ---------------------------------------------------------------------------

@dataclass
class McpServerConfig:
    """MCP server configuration from .mcp.json."""
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    scope: str = "project"  # 'project' | 'user'


@dataclass
class SessionState:
    """Complete session state for serialization/teleportation."""
    conversation_history: list[dict]
    tool_permissions: dict[str, bool]
    active_workers: list[str]
    pending_approvals: list[dict]
    compaction_state: dict
    context_manager_state: dict
    cost_tracking: dict
    hook_registrations: dict
    trace_id: str


# ---------------------------------------------------------------------------
# Enterprise Settings
# ---------------------------------------------------------------------------

@dataclass
class EnterpriseSettings:
    """Organization-level policy settings that override all user/project settings.

    Requirements: 23.1, 23.2, 23.3, 23.4
    """

    allow_managed_hooks_only: bool = False
    """When True, only enterprise_managed scope hooks are allowed (Req 23.2)."""

    allowed_models: list[str] = field(default_factory=list)
    """Allowlist of model IDs; empty list means all models are permitted."""

    max_session_budget_usd: float | None = None
    """Maximum per-session cost in USD; None means no limit."""

    max_team_budget_usd: float | None = None
    """Maximum per-team cost in USD; None means no limit."""

    required_skills: list[str] = field(default_factory=list)
    """Skills that must be loaded for every session."""

    blocked_tools: list[str] = field(default_factory=list)
    """Tool names that are unconditionally blocked."""

    audit_log_endpoint: str | None = None
    """HTTP endpoint for audit log delivery; None disables audit logging."""

    git_config_repo_url: str | None = None
    """Git repository URL for policy distribution (Req 23.3)."""
