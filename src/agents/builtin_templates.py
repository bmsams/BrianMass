"""Built-in agent templates.

Requirements: 8.12
"""

from __future__ import annotations

from src.types.core import AgentDefinition


def get_builtin_agent_templates() -> list[AgentDefinition]:
    """Return the five built-in agent templates."""

    return [
        AgentDefinition(
            name="code-reviewer",
            description="Review code quality, correctness, and maintainability.",
            model="sonnet",
            tools=["Read", "Glob", "Grep", "Bash"],
            color="purple",
            max_turns=60,
            system_prompt="You are a senior code reviewer focused on actionable findings.",
            source="cli",
        ),
        AgentDefinition(
            name="security-auditor",
            description="Audit code for security vulnerabilities using OWASP guidance.",
            model="opus",
            disallowed_tools=["Write", "Edit", "MultiEdit"],
            skills=["owasp"],
            color="red",
            max_turns=80,
            system_prompt="You are a security auditor with strict read-only behavior.",
            source="cli",
        ),
        AgentDefinition(
            name="implementer-tester",
            description="Implement features and write tests with fast iteration loops.",
            model="sonnet",
            tools=["Read", "Write", "Edit", "MultiEdit", "Bash", "Grep", "Glob"],
            color="green",
            max_turns=120,
            system_prompt=(
                "You implement code and tests. Run validation after each major change."
            ),
            source="cli",
        ),
        AgentDefinition(
            name="researcher",
            description="Perform lightweight codebase research and planning in read-only mode.",
            model="haiku",
            disallowed_tools=["Write", "Edit", "MultiEdit", "Bash"],
            permission_mode="plan",
            color="cyan",
            max_turns=40,
            system_prompt="You gather facts and produce concise plans with citations.",
            source="cli",
        ),
        AgentDefinition(
            name="architect",
            description="Design architecture and ADRs for complex systems and migrations.",
            model="opus",
            skills=["adr"],
            mcp_servers={"github": {"enabled": True}},
            color="blue",
            max_turns=100,
            system_prompt="You produce architecture decisions with clear tradeoffs.",
            source="cli",
        ),
    ]

