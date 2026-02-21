"""Unit tests for the Skill Registry.

Tests discovery, SKILL.md parsing, auto-invocation matching, slash commands,
invocation logging, context consumption tracking, hot-reload, and plugin
namespacing.

Requirements: 10.1, 10.2, 10.3, 10.4, 10.5
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from src.skills.skill_registry import (
    SkillRegistry,
    _tokenize,
    compute_keyword_confidence,
    parse_skill_md,
)
from src.types.core import SkillDefinition

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_skill_md(directory: Path, name: str, content: str) -> Path:
    """Write a SKILL.md file inside *directory* and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "SKILL.md"
    path.write_text(content, encoding="utf-8")
    return path


SIMPLE_SKILL_MD = textwrap.dedent("""\
    ---
    name: code-review
    description: Review code for quality and security issues
    ---
    You are a code review assistant. Analyze code for bugs and style issues.
""")

DISABLED_SKILL_MD = textwrap.dedent("""\
    ---
    name: internal-tool
    description: Internal tool that should not be auto-invoked
    disable-model-invocation: true
    ---
    Internal use only.
""")

SKILL_WITH_TOOLS_MD = textwrap.dedent("""\
    ---
    name: linter
    description: Run linting checks on source files
    allowed_tools: eslint, pylint, flake8
    ---
    Lint the codebase.
""")

SKILL_WITH_HOOKS_MD = textwrap.dedent("""\
    ---
    name: formatter
    description: Auto-format code on save
    hooks: PreToolUse, PostToolUse
    ---
    Format code automatically.
""")


# ---------------------------------------------------------------------------
# parse_skill_md
# ---------------------------------------------------------------------------

class TestParseSkillMd:
    """Tests for SKILL.md frontmatter parsing."""

    def test_parse_simple_skill(self, tmp_path: Path):
        path = _write_skill_md(tmp_path, "code-review", SIMPLE_SKILL_MD)
        skill = parse_skill_md(str(path))
        assert skill.name == "code-review"
        assert "quality" in skill.description
        assert skill.disable_model_invocation is False
        assert skill.allowed_tools is None
        assert "code review assistant" in skill.content.lower()

    def test_parse_disabled_skill(self, tmp_path: Path):
        path = _write_skill_md(tmp_path, "internal", DISABLED_SKILL_MD)
        skill = parse_skill_md(str(path))
        assert skill.name == "internal-tool"
        assert skill.disable_model_invocation is True

    def test_parse_allowed_tools(self, tmp_path: Path):
        path = _write_skill_md(tmp_path, "linter", SKILL_WITH_TOOLS_MD)
        skill = parse_skill_md(str(path))
        assert skill.allowed_tools == ["eslint", "pylint", "flake8"]

    def test_parse_hooks(self, tmp_path: Path):
        path = _write_skill_md(tmp_path, "formatter", SKILL_WITH_HOOKS_MD)
        skill = parse_skill_md(str(path))
        assert "PreToolUse" in skill.hooks
        assert "PostToolUse" in skill.hooks

    def test_missing_frontmatter_raises(self, tmp_path: Path):
        path = tmp_path / "SKILL.md"
        path.write_text("No frontmatter here", encoding="utf-8")
        with pytest.raises(ValueError, match="No frontmatter"):
            parse_skill_md(str(path))

    def test_unclosed_frontmatter_raises(self, tmp_path: Path):
        path = tmp_path / "SKILL.md"
        path.write_text("---\nname: broken\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Unclosed frontmatter"):
            parse_skill_md(str(path))

    def test_missing_name_raises(self, tmp_path: Path):
        content = "---\ndescription: no name\n---\nbody"
        path = tmp_path / "SKILL.md"
        path.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match="Missing required field 'name'"):
            parse_skill_md(str(path))

    def test_missing_description_raises(self, tmp_path: Path):
        content = "---\nname: no-desc\n---\nbody"
        path = tmp_path / "SKILL.md"
        path.write_text(content, encoding="utf-8")
        with pytest.raises(ValueError, match="Missing required field 'description'"):
            parse_skill_md(str(path))

    def test_utf8_bom_handled(self, tmp_path: Path):
        content = "\ufeff" + SIMPLE_SKILL_MD
        path = tmp_path / "SKILL.md"
        path.write_text(content, encoding="utf-8")
        skill = parse_skill_md(str(path))
        assert skill.name == "code-review"

    def test_quoted_values_stripped(self, tmp_path: Path):
        content = '---\nname: "my-skill"\ndescription: \'A cool skill\'\n---\nbody'
        path = tmp_path / "SKILL.md"
        path.write_text(content, encoding="utf-8")
        skill = parse_skill_md(str(path))
        assert skill.name == "my-skill"
        assert skill.description == "A cool skill"

    def test_empty_body(self, tmp_path: Path):
        content = "---\nname: empty\ndescription: empty body skill\n---\n"
        path = tmp_path / "SKILL.md"
        path.write_text(content, encoding="utf-8")
        skill = parse_skill_md(str(path))
        assert skill.content == ""


# ---------------------------------------------------------------------------
# Keyword Matching
# ---------------------------------------------------------------------------

class TestKeywordMatching:
    """Tests for compute_keyword_confidence."""

    def test_exact_match_high_confidence(self):
        skill = SkillDefinition(name="code-review", description="review code for quality")
        score = compute_keyword_confidence("review code quality", skill)
        assert score > 0.5

    def test_no_overlap_zero_confidence(self):
        skill = SkillDefinition(name="database", description="manage database migrations")
        score = compute_keyword_confidence("paint a picture", skill)
        assert score == 0.0

    def test_partial_overlap(self):
        skill = SkillDefinition(name="test-runner", description="run unit tests and integration tests")
        score = compute_keyword_confidence("run unit tests", skill)
        assert 0.0 < score <= 1.0

    def test_empty_task_returns_zero(self):
        skill = SkillDefinition(name="anything", description="does stuff")
        assert compute_keyword_confidence("", skill) == 0.0

    def test_case_insensitive(self):
        skill = SkillDefinition(name="Linter", description="Run ESLint checks")
        score = compute_keyword_confidence("run eslint checks", skill)
        assert score > 0.0

    def test_tokenize_filters_short_words(self):
        tokens = _tokenize("I am a dev")
        assert "i" not in tokens
        assert "am" in tokens
        assert "dev" in tokens


# ---------------------------------------------------------------------------
# SkillRegistry — Discovery & Registration
# ---------------------------------------------------------------------------

class TestSkillRegistryDiscovery:
    """Tests for skill discovery from directories."""

    def test_discover_from_project_dir(self, tmp_path: Path):
        project_dir = tmp_path / ".brainmass" / "skills" / "review"
        _write_skill_md(project_dir, "code-review", SIMPLE_SKILL_MD)

        registry = SkillRegistry(
            project_skills_dir=str(tmp_path / ".brainmass" / "skills"),
            user_skills_dir=str(tmp_path / "nonexistent"),
        )
        count = registry.discover()
        assert count == 1
        assert "code-review" in registry.skills

    def test_discover_from_user_dir(self, tmp_path: Path):
        user_dir = tmp_path / "user_skills" / "review"
        _write_skill_md(user_dir, "code-review", SIMPLE_SKILL_MD)

        registry = SkillRegistry(
            user_skills_dir=str(tmp_path / "user_skills"),
            project_skills_dir=str(tmp_path / "nonexistent"),
        )
        count = registry.discover()
        assert count == 1

    def test_discover_from_both_dirs(self, tmp_path: Path):
        user_dir = tmp_path / "user" / "skill1"
        project_dir = tmp_path / "project" / "skill2"
        _write_skill_md(user_dir, "skill1", SIMPLE_SKILL_MD)
        _write_skill_md(project_dir, "skill2", DISABLED_SKILL_MD)

        registry = SkillRegistry(
            user_skills_dir=str(tmp_path / "user"),
            project_skills_dir=str(tmp_path / "project"),
        )
        count = registry.discover()
        assert count == 2

    def test_discover_nonexistent_dirs(self, tmp_path: Path):
        registry = SkillRegistry(
            user_skills_dir=str(tmp_path / "nope1"),
            project_skills_dir=str(tmp_path / "nope2"),
        )
        assert registry.discover() == 0

    def test_discover_skips_invalid_files(self, tmp_path: Path):
        skill_dir = tmp_path / "skills" / "bad"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("no frontmatter", encoding="utf-8")

        registry = SkillRegistry(
            project_skills_dir=str(tmp_path / "skills"),
            user_skills_dir=str(tmp_path / "nope"),
        )
        assert registry.discover() == 0


# ---------------------------------------------------------------------------
# SkillRegistry — Register / Unregister / Get
# ---------------------------------------------------------------------------

class TestSkillRegistryRegistration:
    """Tests for manual skill registration."""

    def test_register_and_get(self):
        registry = SkillRegistry()
        skill = SkillDefinition(name="my-skill", description="does things")
        registry.register(skill)
        assert registry.get("my-skill") is skill

    def test_unregister(self):
        registry = SkillRegistry()
        skill = SkillDefinition(name="my-skill", description="does things")
        registry.register(skill)
        assert registry.unregister("my-skill") is True
        assert registry.get("my-skill") is None

    def test_unregister_nonexistent(self):
        registry = SkillRegistry()
        assert registry.unregister("ghost") is False

    def test_register_overwrites(self):
        registry = SkillRegistry()
        skill1 = SkillDefinition(name="s", description="v1")
        skill2 = SkillDefinition(name="s", description="v2")
        registry.register(skill1)
        registry.register(skill2)
        assert registry.get("s").description == "v2"


# ---------------------------------------------------------------------------
# SkillRegistry — Slash Commands
# ---------------------------------------------------------------------------

class TestSlashCommands:
    """Tests for slash command registration and invocation."""

    def test_slash_command_registered_on_register(self):
        registry = SkillRegistry()
        skill = SkillDefinition(name="lint", description="run linter")
        registry.register(skill)
        assert "/lint" in registry.slash_commands

    def test_invoke_slash_command(self):
        registry = SkillRegistry()
        skill = SkillDefinition(name="lint", description="run linter")
        registry.register(skill)
        result = registry.invoke_slash_command("/lint")
        assert result is skill

    def test_invoke_unknown_slash_command(self):
        registry = SkillRegistry()
        assert registry.invoke_slash_command("/unknown") is None

    def test_slash_command_removed_on_unregister(self):
        registry = SkillRegistry()
        skill = SkillDefinition(name="lint", description="run linter")
        registry.register(skill)
        registry.unregister("lint")
        assert "/lint" not in registry.slash_commands


# ---------------------------------------------------------------------------
# SkillRegistry — Auto-Invocation
# ---------------------------------------------------------------------------

class TestAutoInvocation:
    """Tests for auto-invocation matching."""

    def _make_registry(self) -> SkillRegistry:
        registry = SkillRegistry(confidence_threshold=0.3)
        registry.register(SkillDefinition(
            name="code-review",
            description="review code for quality and security issues",
        ))
        registry.register(SkillDefinition(
            name="test-runner",
            description="run unit tests and integration tests",
        ))
        registry.register(SkillDefinition(
            name="disabled-skill",
            description="this skill is disabled",
            disable_model_invocation=True,
        ))
        return registry

    def test_auto_invoke_selects_best_match(self):
        registry = self._make_registry()
        result = registry.auto_invoke("review code quality")
        assert result is not None
        assert result.name == "code-review"

    def test_auto_invoke_returns_none_below_threshold(self):
        registry = self._make_registry()
        result = registry.auto_invoke("paint a landscape")
        assert result is None

    def test_auto_invoke_skips_disabled_skills(self):
        registry = self._make_registry()
        result = registry.auto_invoke("this skill is disabled")
        # Even though the text matches, the skill is disabled
        assert result is None or result.name != "disabled-skill"

    def test_match_returns_sorted_by_confidence(self):
        registry = self._make_registry()
        matches = registry.match("run unit tests")
        # test-runner should rank higher
        enabled_matches = [m for m in matches if m.confidence > 0]
        if len(enabled_matches) >= 2:
            assert enabled_matches[0].confidence >= enabled_matches[1].confidence

    def test_match_includes_disabled_with_zero_confidence(self):
        registry = self._make_registry()
        matches = registry.match("anything")
        disabled = [m for m in matches if m.skill_name == "disabled-skill"]
        assert len(disabled) == 1
        assert disabled[0].confidence == 0.0
        assert "disabled" in disabled[0].reason


# ---------------------------------------------------------------------------
# SkillRegistry — Invocation Logging
# ---------------------------------------------------------------------------

class TestInvocationLogging:
    """Tests for invocation logging (Req 10.4)."""

    def test_auto_invoke_creates_log_entry(self):
        registry = SkillRegistry(confidence_threshold=0.3)
        registry.register(SkillDefinition(name="s", description="do stuff"))
        registry.auto_invoke("do stuff")
        logs = registry.get_invocation_logs()
        assert len(logs) == 1
        assert logs[0].task_description == "do stuff"

    def test_log_records_candidates(self):
        registry = SkillRegistry(confidence_threshold=0.3)
        registry.register(SkillDefinition(name="a", description="alpha"))
        registry.register(SkillDefinition(name="b", description="beta"))
        registry.auto_invoke("alpha task")
        log = registry.invocation_logs[0]
        assert len(log.candidates) == 2

    def test_log_records_selected_skill(self):
        registry = SkillRegistry(confidence_threshold=0.0)
        registry.register(SkillDefinition(name="s", description="do stuff"))
        registry.auto_invoke("do stuff")
        log = registry.invocation_logs[0]
        assert log.selected_skill == "s"

    def test_log_records_rejection_reason(self):
        registry = SkillRegistry(confidence_threshold=0.99)
        registry.register(SkillDefinition(name="s", description="manage database migrations"))
        registry.auto_invoke("review code quality")
        log = registry.invocation_logs[0]
        assert log.selected_skill is None
        assert "threshold" in log.selection_reason

    def test_clear_invocation_logs(self):
        registry = SkillRegistry()
        registry.register(SkillDefinition(name="s", description="d"))
        registry.auto_invoke("d")
        registry.clear_invocation_logs()
        assert len(registry.invocation_logs) == 0


# ---------------------------------------------------------------------------
# SkillRegistry — Context Consumption Tracking
# ---------------------------------------------------------------------------

class TestContextConsumption:
    """Tests for context consumption tracking (Req 10.4)."""

    def test_record_and_get_usage(self):
        registry = SkillRegistry()
        skill = SkillDefinition(name="s", description="d")
        registry.register(skill)
        registry.record_context_usage("s", 500)
        usage = registry.get_context_usage("s")
        assert usage is not None
        assert usage.total_tokens == 500
        assert usage.invocation_count == 1

    def test_cumulative_tracking(self):
        registry = SkillRegistry()
        registry.register(SkillDefinition(name="s", description="d"))
        registry.record_context_usage("s", 100)
        registry.record_context_usage("s", 200)
        usage = registry.get_context_usage("s")
        assert usage.total_tokens == 300
        assert usage.invocation_count == 2

    def test_get_all_context_usage(self):
        registry = SkillRegistry()
        registry.register(SkillDefinition(name="a", description="d"))
        registry.register(SkillDefinition(name="b", description="d"))
        registry.record_context_usage("a", 100)
        registry.record_context_usage("b", 200)
        all_usage = registry.get_all_context_usage()
        assert "a" in all_usage
        assert "b" in all_usage

    def test_usage_for_unregistered_skill(self):
        registry = SkillRegistry()
        registry.record_context_usage("unknown", 100)
        usage = registry.get_context_usage("unknown")
        assert usage is not None
        assert usage.total_tokens == 100


# ---------------------------------------------------------------------------
# SkillRegistry — Hot-Reload
# ---------------------------------------------------------------------------

class TestHotReload:
    """Tests for hot-reload via reload() (Req 10.4)."""

    def test_reload_clears_and_rediscovers(self, tmp_path: Path):
        skill_dir = tmp_path / "skills" / "s1"
        _write_skill_md(skill_dir, "s1", SIMPLE_SKILL_MD)

        registry = SkillRegistry(
            project_skills_dir=str(tmp_path / "skills"),
            user_skills_dir=str(tmp_path / "nope"),
        )
        registry.discover()
        assert len(registry.skills) == 1

        # Add a second skill
        skill_dir2 = tmp_path / "skills" / "s2"
        _write_skill_md(skill_dir2, "s2", DISABLED_SKILL_MD)

        count = registry.reload()
        assert count == 2
        assert "code-review" in registry.skills
        assert "internal-tool" in registry.skills

    def test_reload_removes_deleted_skills(self, tmp_path: Path):
        skill_dir = tmp_path / "skills" / "s1"
        path = _write_skill_md(skill_dir, "s1", SIMPLE_SKILL_MD)

        registry = SkillRegistry(
            project_skills_dir=str(tmp_path / "skills"),
            user_skills_dir=str(tmp_path / "nope"),
        )
        registry.discover()
        assert len(registry.skills) == 1

        # Delete the skill file
        path.unlink()
        skill_dir.rmdir()

        count = registry.reload()
        assert count == 0
        assert len(registry.skills) == 0


# ---------------------------------------------------------------------------
# SkillRegistry — Plugin Namespacing
# ---------------------------------------------------------------------------

class TestPluginNamespacing:
    """Tests for plugin skill namespacing (Req 10.5)."""

    def test_plugin_skill_namespaced(self):
        registry = SkillRegistry()
        skill = SkillDefinition(name="lint", description="run linter")
        registry.register_plugin_skill("my-plugin", skill)
        assert "my-plugin:lint" in registry.skills
        assert registry.get("my-plugin:lint") is not None

    def test_plugin_slash_command_namespaced(self):
        registry = SkillRegistry()
        skill = SkillDefinition(name="lint", description="run linter")
        registry.register_plugin_skill("my-plugin", skill)
        assert "/my-plugin:lint" in registry.slash_commands

    def test_plugin_skill_does_not_collide_with_user_skill(self):
        registry = SkillRegistry()
        user_skill = SkillDefinition(name="lint", description="user linter")
        plugin_skill = SkillDefinition(name="lint", description="plugin linter")
        registry.register(user_skill)
        registry.register_plugin_skill("ext", plugin_skill)
        assert registry.get("lint").description == "user linter"
        assert registry.get("ext:lint").description == "plugin linter"
