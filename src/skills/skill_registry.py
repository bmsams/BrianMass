"""Skill Registry for skill discovery, invocation, and lifecycle management.

Discovers skills from ~/.brainmass/skills and .brainmass/skills directories,
parses SKILL.md frontmatter, supports auto-invocation matching, slash command
registration, hot-reload, invocation logging, and context consumption tracking.

Requirements: 10.1, 10.2, 10.3, 10.4, 10.5
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from src.observability.instrumentation import BrainmassTracer
from src.types.core import HookDefinition, SkillDefinition

logger = logging.getLogger(__name__)

# Default confidence threshold for auto-invocation
DEFAULT_CONFIDENCE_THRESHOLD = 0.3


# ---------------------------------------------------------------------------
# Invocation Logging
# ---------------------------------------------------------------------------

@dataclass
class SkillMatch:
    """A single skill match result from auto-invocation matching."""
    skill_name: str
    confidence: float
    selected: bool = False
    reason: str = ""


@dataclass
class InvocationLog:
    """Log entry for a skill invocation decision."""
    task_description: str
    candidates: list[SkillMatch] = field(default_factory=list)
    selected_skill: str | None = None
    selection_reason: str = ""


# ---------------------------------------------------------------------------
# Context Consumption Tracking
# ---------------------------------------------------------------------------

@dataclass
class SkillContextUsage:
    """Tracks token consumption for a single skill."""
    skill_name: str
    invocation_count: int = 0
    total_tokens: int = 0


# ---------------------------------------------------------------------------
# SKILL.md Frontmatter Parser (manual, no python-frontmatter dependency)
# ---------------------------------------------------------------------------


def parse_skill_md(file_path: str) -> SkillDefinition:
    """Parse a SKILL.md file into a SkillDefinition.

    Manually splits on ``---`` delimiters to extract YAML frontmatter,
    then parses the YAML fields. The Markdown body becomes the skill content.

    Raises ``ValueError`` if the file has no valid frontmatter or is missing
    required fields (name, description).
    """
    text = Path(file_path).read_text(encoding="utf-8")

    # Strip UTF-8 BOM if present
    if text.startswith("\ufeff"):
        text = text[1:]

    # Frontmatter must start with '---' on the first line
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"No frontmatter found in {file_path}")

    # Find the closing '---'
    end_idx: int | None = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break

    if end_idx is None:
        raise ValueError(f"Unclosed frontmatter in {file_path}")

    fm_lines = lines[1:end_idx]
    body = "\n".join(lines[end_idx + 1 :]).strip()

    # Simple YAML-like key: value parser (handles the fields we need)
    fm = _parse_simple_yaml(fm_lines)

    # Required fields
    name = fm.get("name")
    description = fm.get("description")
    if not name:
        raise ValueError(f"Missing required field 'name' in {file_path}")
    if not description:
        raise ValueError(f"Missing required field 'description' in {file_path}")

    disable_model_invocation = _parse_bool(fm.get("disable-model-invocation", "false"))

    allowed_tools: list[str] | None = None
    raw_tools = fm.get("allowed_tools")
    if raw_tools:
        allowed_tools = [t.strip() for t in raw_tools.split(",") if t.strip()]

    hooks = _parse_hooks(fm.get("hooks"))

    return SkillDefinition(
        name=str(name).strip(),
        description=str(description).strip(),
        disable_model_invocation=disable_model_invocation,
        hooks=hooks,
        allowed_tools=allowed_tools,
        content=body,
    )


def _parse_simple_yaml(lines: list[str]) -> dict[str, str]:
    """Minimal YAML parser for flat key: value pairs."""
    result: dict[str, str] = {}
    current_key: str | None = None
    for line in lines:
        if not line.strip() or line.strip().startswith("#"):
            continue
        # Check for key: value
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_-]*)\s*:\s*(.*)", line)
        if match:
            current_key = match.group(1)
            value = match.group(2).strip()
            # Strip surrounding quotes
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            result[current_key] = value
        elif current_key and line.startswith("  "):
            # Continuation of previous value
            result[current_key] += " " + line.strip()
    return result


def _parse_bool(value: str) -> bool:
    """Parse a YAML-ish boolean string."""
    return str(value).lower() in ("true", "yes", "1")


def _parse_hooks(raw: str | None) -> dict[str, list[HookDefinition]]:
    """Parse hooks from a simple comma-separated string or return empty dict.

    For the simple frontmatter parser, hooks are represented as a
    comma-separated list of event names. Full hook definitions with
    handlers would come from the settings system.
    """
    if not raw:
        return {}
    hooks: dict[str, list[HookDefinition]] = {}
    for event_name in raw.split(","):
        event_name = event_name.strip()
        if event_name:
            hooks[event_name] = [HookDefinition()]
    return hooks


# ---------------------------------------------------------------------------
# Keyword Matching for Auto-Invocation
# ---------------------------------------------------------------------------

def compute_keyword_confidence(task_description: str, skill: SkillDefinition) -> float:
    """Compute a confidence score for how well a skill matches a task.

    Uses simple keyword overlap between the task description and the skill's
    name + description. Returns a float in [0.0, 1.0].
    """
    task_words = _tokenize(task_description)
    if not task_words:
        return 0.0

    skill_words = _tokenize(skill.name + " " + skill.description + " " + skill.content)
    if not skill_words:
        return 0.0

    overlap = task_words & skill_words
    if not overlap:
        return 0.0

    # Jaccard-like score weighted toward task coverage
    task_coverage = len(overlap) / len(task_words)
    skill_coverage = len(overlap) / len(skill_words)
    # Weighted average: task coverage matters more
    return 0.7 * task_coverage + 0.3 * skill_coverage


def _tokenize(text: str) -> set[str]:
    """Lowercase and split text into a set of word tokens (>=2 chars)."""
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 2}



# ---------------------------------------------------------------------------
# Skill Registry
# ---------------------------------------------------------------------------

class SkillRegistry:
    """Central registry for skill discovery, invocation, and lifecycle.

    Responsibilities:
    - Discover skills from ~/.brainmass/skills and .brainmass/skills (Req 10.1)
    - Auto-invocation matching with confidence threshold (Req 10.2)
    - Slash command registration (Req 10.2)
    - Skill-scoped hooks (Req 10.3)
    - Invocation logging and context consumption tracking (Req 10.4)
    - Hot-reload via reload() (Req 10.4)
    - Plugin skill namespacing (Req 10.5)
    """

    def __init__(
        self,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        user_skills_dir: str | None = None,
        project_skills_dir: str | None = None,
        tracer: BrainmassTracer | None = None,
    ) -> None:
        self._confidence_threshold = confidence_threshold

        # Resolve default directories
        home = Path.home()
        self._user_skills_dir = Path(user_skills_dir) if user_skills_dir else home / ".brainmass" / "skills"
        self._project_skills_dir = Path(project_skills_dir) if project_skills_dir else Path(".brainmass") / "skills"

        # Registered skills: name -> SkillDefinition
        self._skills: dict[str, SkillDefinition] = {}

        # Slash commands: /skill-name -> skill name
        self._slash_commands: dict[str, str] = {}

        # Invocation logs
        self._invocation_logs: list[InvocationLog] = []

        # Context consumption tracking: skill_name -> SkillContextUsage
        self._context_usage: dict[str, SkillContextUsage] = {}

        # Observability
        self._tracer = tracer

    # -- Properties -----------------------------------------------------------

    @property
    def confidence_threshold(self) -> float:
        return self._confidence_threshold

    @confidence_threshold.setter
    def confidence_threshold(self, value: float) -> None:
        self._confidence_threshold = value

    @property
    def skills(self) -> dict[str, SkillDefinition]:
        """Return a copy of the registered skills."""
        return dict(self._skills)

    @property
    def slash_commands(self) -> dict[str, str]:
        """Return a copy of the slash command registry."""
        return dict(self._slash_commands)

    @property
    def invocation_logs(self) -> list[InvocationLog]:
        """Return the invocation log history."""
        return list(self._invocation_logs)

    # -- Discovery & Registration ---------------------------------------------

    def discover(self) -> int:
        """Scan skill directories and register all discovered SKILL.md files.

        Returns the number of skills discovered.
        """
        count = 0
        for directory in (self._user_skills_dir, self._project_skills_dir):
            count += self._scan_directory(directory)
        return count

    def _scan_directory(self, directory: Path) -> int:
        """Scan a single directory for SKILL.md files."""
        if not directory.is_dir():
            return 0
        count = 0
        for skill_file in directory.rglob("SKILL.md"):
            try:
                skill_def = parse_skill_md(str(skill_file))
                self.register(skill_def)
                count += 1
            except (ValueError, OSError) as exc:
                logger.warning("Failed to load skill from %s: %s", skill_file, exc)
        return count

    def register(self, skill: SkillDefinition) -> None:
        """Register a skill definition and its slash command."""
        self._skills[skill.name] = skill
        # Register slash command: /skill-name
        cmd = f"/{skill.name}"
        self._slash_commands[cmd] = skill.name
        # Initialize context usage tracking
        if skill.name not in self._context_usage:
            self._context_usage[skill.name] = SkillContextUsage(skill_name=skill.name)

    def unregister(self, skill_name: str) -> bool:
        """Remove a skill from the registry. Returns True if it existed."""
        if skill_name not in self._skills:
            return False
        del self._skills[skill_name]
        cmd = f"/{skill_name}"
        self._slash_commands.pop(cmd, None)
        return True

    def get(self, name: str) -> SkillDefinition | None:
        """Look up a skill by name."""
        return self._skills.get(name)

    # -- Plugin Namespacing (Req 10.5) ----------------------------------------

    def register_plugin_skill(self, plugin_name: str, skill: SkillDefinition) -> None:
        """Register a plugin skill with namespace prefix plugin-name:skill-name."""
        namespaced_name = f"{plugin_name}:{skill.name}"
        namespaced_skill = SkillDefinition(
            name=namespaced_name,
            description=skill.description,
            disable_model_invocation=skill.disable_model_invocation,
            hooks=skill.hooks,
            allowed_tools=skill.allowed_tools,
            content=skill.content,
        )
        self.register(namespaced_skill)

    # -- Auto-Invocation Matching (Req 10.2) ----------------------------------

    def match(self, task_description: str) -> list[SkillMatch]:
        """Match a task description against all registered skills.

        Returns a list of SkillMatch objects sorted by confidence (descending).
        Only skills with disable_model_invocation=False are considered.
        """
        matches: list[SkillMatch] = []
        for skill in self._skills.values():
            if skill.disable_model_invocation:
                matches.append(SkillMatch(
                    skill_name=skill.name,
                    confidence=0.0,
                    selected=False,
                    reason="disabled via disable-model-invocation",
                ))
                continue

            confidence = compute_keyword_confidence(task_description, skill)
            matches.append(SkillMatch(
                skill_name=skill.name,
                confidence=confidence,
                selected=False,
                reason="",
            ))

        matches.sort(key=lambda m: m.confidence, reverse=True)
        return matches

    def auto_invoke(self, task_description: str) -> SkillDefinition | None:
        """Attempt to auto-invoke a skill for the given task description.

        Returns the best matching skill if its confidence exceeds the threshold,
        otherwise None. Logs the invocation decision.
        """
        matches = self.match(task_description)

        log_entry = InvocationLog(
            task_description=task_description,
            candidates=matches,
        )

        best: SkillMatch | None = None
        for m in matches:
            if not m.reason and m.confidence >= self._confidence_threshold:
                best = m
                break

        if best is not None:
            best.selected = True
            best.reason = f"confidence {best.confidence:.3f} >= threshold {self._confidence_threshold:.3f}"
            log_entry.selected_skill = best.skill_name
            log_entry.selection_reason = best.reason
        else:
            log_entry.selection_reason = "no skill met confidence threshold"
            # Annotate rejection reasons for non-disabled skills
            for m in matches:
                if not m.reason:
                    m.reason = f"confidence {m.confidence:.3f} < threshold {self._confidence_threshold:.3f}"

        self._invocation_logs.append(log_entry)

        # Emit skill span for every candidate considered (Req 16.2)
        if self._tracer is not None:
            for m in matches:
                self._tracer.record_skill_span(
                    skill_name=m.skill_name,
                    confidence=m.confidence,
                    selection_reason=m.reason,
                    matched=m.selected,
                )

        if best is not None:
            return self._skills.get(best.skill_name)
        return None

    # -- Slash Command Invocation (Req 10.2) ----------------------------------

    def invoke_slash_command(self, command: str) -> SkillDefinition | None:
        """Look up a skill by slash command (e.g. '/my-skill').

        Returns the SkillDefinition if found, otherwise None.
        """
        skill_name = self._slash_commands.get(command)
        if skill_name is None:
            return None
        return self._skills.get(skill_name)

    # -- Context Consumption Tracking (Req 10.4) ------------------------------

    def record_context_usage(self, skill_name: str, tokens: int) -> None:
        """Record token consumption for a skill invocation."""
        usage = self._context_usage.get(skill_name)
        if usage is None:
            usage = SkillContextUsage(skill_name=skill_name)
            self._context_usage[skill_name] = usage
        usage.invocation_count += 1
        usage.total_tokens += tokens

    def get_context_usage(self, skill_name: str) -> SkillContextUsage | None:
        """Return context usage stats for a skill."""
        return self._context_usage.get(skill_name)

    def get_all_context_usage(self) -> dict[str, SkillContextUsage]:
        """Return context usage stats for all skills."""
        return dict(self._context_usage)

    # -- Hot-Reload (Req 10.4) ------------------------------------------------

    def reload(self) -> int:
        """Re-scan skill directories and update the registry.

        Clears existing skills and re-discovers from disk.
        Returns the number of skills loaded.
        """
        self._skills.clear()
        self._slash_commands.clear()
        return self.discover()

    # -- Invocation Logging (Req 10.4) ----------------------------------------

    def get_invocation_logs(self) -> list[InvocationLog]:
        """Return all invocation log entries."""
        return list(self._invocation_logs)

    def clear_invocation_logs(self) -> None:
        """Clear all invocation log entries."""
        self._invocation_logs.clear()
