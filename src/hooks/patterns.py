"""Community-proven hook patterns — pre-built templates for common use cases.

Provides five composable hook pattern classes:

1. SecurityGuardPattern — blocks dangerous bash commands (PreToolUse)
2. ContextBackupPattern — backs up PRESERVE_VERBATIM items before compaction
3. AutoFormatPattern — runs a formatter on Write/Edit tool output (PostToolUse)
4. LoopContextTemplate — generates/validates loop-context.json structures
5. TeamTaskListTemplate — generates/validates tasks.json structures

All patterns are SDK-independent and use pluggable callbacks for any external
dependencies, making them fully testable in isolation.

Requirements: 27.1, 27.2, 27.3, 27.4, 27.5
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from src.types.core import (
    ContextCategory,
    ContextItem,
    HookContext,
    HookResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Security Guard Pattern (Req 27.2)
# ---------------------------------------------------------------------------

class SecurityGuardPattern:
    """PreToolUse hook that blocks dangerous bash commands.

    Evaluates tool input for the ``Bash`` / ``bash`` / ``shell`` tool and
    checks the command string against a configurable list of dangerous
    regex patterns.  Returns a deny ``HookResult`` when a match is found,
    or an allow result when the command is safe.

    Usage::

        guard = SecurityGuardPattern()
        result = guard.evaluate(hook_context)
        if result.permission_decision == "deny":
            print(result.permission_decision_reason)

    The pattern list can be extended at construction time via the
    *extra_patterns* parameter.
    """

    # Default dangerous command patterns (compiled at class level for speed).
    DANGEROUS_PATTERNS: list[str] = [
        # Destructive filesystem operations
        r"rm\s+-rf\s+/",
        r"rm\s+-fr\s+/",
        r"rm\s+--no-preserve-root",
        # SQL injection / data destruction
        r"(?i)DROP\s+TABLE",
        r"(?i)DROP\s+DATABASE",
        r"(?i)TRUNCATE\s+TABLE",
        # Remote code execution via pipe
        r"curl\s+.*\|\s*(?:ba)?sh",
        r"wget\s+.*\|\s*(?:ba)?sh",
        r"curl\s+.*\|\s*python",
        r"wget\s+.*\|\s*python",
        # Overly permissive file permissions
        r"chmod\s+777",
        r"chmod\s+-R\s+777",
        # Force push (can destroy remote history)
        r"git\s+push\s+--force",
        r"git\s+push\s+-f\b",
        # Disk / partition destruction
        r"mkfs\.",
        r"dd\s+.*of=/dev/",
        # Fork bomb
        r":\(\)\s*\{\s*:\|:\s*&\s*\}\s*;",
    ]

    def __init__(
        self,
        extra_patterns: list[str] | None = None,
        tool_names: set[str] | None = None,
    ) -> None:
        """Initialise the security guard.

        Parameters
        ----------
        extra_patterns:
            Additional regex patterns to add to the default dangerous list.
        tool_names:
            Set of tool names to monitor (default: ``{"Bash", "bash", "shell"}``).
        """
        all_patterns = list(self.DANGEROUS_PATTERNS)
        if extra_patterns:
            all_patterns.extend(extra_patterns)

        self._compiled: list[re.Pattern[str]] = []
        for pat in all_patterns:
            try:
                self._compiled.append(re.compile(pat))
            except re.error as exc:
                logger.warning(
                    "SecurityGuardPattern: invalid pattern %r — skipping: %s",
                    pat, exc,
                )

        self._tool_names: set[str] = tool_names or {"Bash", "bash", "shell"}

    # -- public API --------------------------------------------------------

    def evaluate(self, hook_context: HookContext) -> HookResult:
        """Evaluate a PreToolUse hook context for dangerous commands.

        Parameters
        ----------
        hook_context:
            The ``HookContext`` from a ``PreToolUse`` event.

        Returns
        -------
        HookResult
            With ``permission_decision="deny"`` and a reason when a
            dangerous pattern is detected, otherwise ``permission_decision="allow"``.
        """
        # Only inspect monitored tool names
        if hook_context.tool_name not in self._tool_names:
            return HookResult(permission_decision="allow")

        command = self._extract_command(hook_context)
        if command is None:
            return HookResult(permission_decision="allow")

        # Check each pattern against the command
        for pattern in self._compiled:
            match = pattern.search(command)
            if match:
                reason = (
                    f"Blocked dangerous command matching pattern "
                    f"/{pattern.pattern}/: {command[:200]}"
                )
                logger.warning("SecurityGuardPattern: %s", reason)
                return HookResult(
                    permission_decision="deny",
                    permission_decision_reason=reason,
                )

        return HookResult(permission_decision="allow")

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _extract_command(hook_context: HookContext) -> str | None:
        """Extract the command string from tool_input."""
        if hook_context.tool_input is None:
            return None
        # Support common input shapes: {"command": "..."} or {"cmd": "..."}
        return (
            hook_context.tool_input.get("command")
            or hook_context.tool_input.get("cmd")
            or hook_context.tool_input.get("input")
        )


# ---------------------------------------------------------------------------
# 2. Context Backup Pattern (Req 27.1)
# ---------------------------------------------------------------------------

class ContextBackupPattern:
    """PreCompact hook that backs up PRESERVE_VERBATIM items before compaction.

    Scans the current context items for entries with
    ``category == PRESERVE_VERBATIM`` and saves them as a timestamped
    JSON backup file under ``backup_dir``.

    Usage::

        backup = ContextBackupPattern(backup_dir=".brainmass/backups")
        metadata = backup.evaluate(context_items, cwd="/project")
        # metadata = {"backup_path": "...", "item_count": 3, "timestamp": "..."}
    """

    def __init__(self, backup_dir: str = ".brainmass/backups") -> None:
        """Initialise the context backup pattern.

        Parameters
        ----------
        backup_dir:
            Directory (relative to *cwd*) where backups are written.
        """
        self._backup_dir = backup_dir

    # -- public API --------------------------------------------------------

    def evaluate(
        self,
        context_items: list[ContextItem],
        cwd: str = ".",
    ) -> dict:
        """Back up PRESERVE_VERBATIM items and return metadata.

        Parameters
        ----------
        context_items:
            All current context items (from the Context Manager).
        cwd:
            Working directory used to resolve *backup_dir*.

        Returns
        -------
        dict
            Metadata about the backup::

                {
                    "backup_path": str,   # absolute path to the backup file
                    "item_count": int,     # number of items backed up
                    "timestamp": str,      # ISO-8601 UTC timestamp
                    "total_tokens": int,   # total token count of backed-up items
                }

            Returns ``{"item_count": 0}`` when there are no items to back up.
        """
        # Filter for PRESERVE_VERBATIM items only
        preserved = [
            item for item in context_items
            if item.category == ContextCategory.PRESERVE_VERBATIM
        ]

        if not preserved:
            logger.debug("ContextBackupPattern: no PRESERVE_VERBATIM items to back up")
            return {"item_count": 0}

        timestamp = datetime.now(UTC)
        timestamp_str = timestamp.strftime("%Y%m%dT%H%M%SZ")

        # Serialise items to JSON-safe dicts
        payload = {
            "timestamp": timestamp.isoformat(),
            "item_count": len(preserved),
            "items": [
                {
                    "id": item.id,
                    "category": item.category.value,
                    "content": item.content,
                    "token_count": item.token_count,
                    "source": item.source,
                    "created_at": item.created_at.isoformat(),
                    "reference_count": item.reference_count,
                }
                for item in preserved
            ],
        }

        # Write backup file (atomic via temp file + rename)
        backup_dir_abs = os.path.join(cwd, self._backup_dir)
        Path(backup_dir_abs).mkdir(parents=True, exist_ok=True)

        filename = f"context_backup_{timestamp_str}.json"
        target_path = os.path.join(backup_dir_abs, filename)

        fd, tmp_path = tempfile.mkstemp(
            dir=backup_dir_abs, suffix=".tmp", prefix=".backup-"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            os.replace(tmp_path, target_path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        total_tokens = sum(item.token_count for item in preserved)
        logger.info(
            "ContextBackupPattern: backed up %d items (%d tokens) to %s",
            len(preserved), total_tokens, target_path,
        )

        return {
            "backup_path": os.path.abspath(target_path),
            "item_count": len(preserved),
            "timestamp": timestamp.isoformat(),
            "total_tokens": total_tokens,
        }


# ---------------------------------------------------------------------------
# 3. Auto-Format Pattern (Req 27.3)
# ---------------------------------------------------------------------------

class AutoFormatPattern:
    """PostToolUse hook that runs a formatter after Write/Edit operations.

    Only triggers on tool names matching ``Write`` or ``Edit``.  Extracts
    the file path from ``tool_input`` and passes it to a pluggable
    ``formatter_callback`` which performs the actual formatting.

    Usage::

        def my_formatter(filepath: str) -> str:
            # Run black, prettier, etc.
            return f"Formatted {filepath}"

        auto_fmt = AutoFormatPattern(formatter_callback=my_formatter)
        result = auto_fmt.evaluate(hook_context)
    """

    # Tool names that trigger formatting.
    _FORMAT_TOOL_NAMES: set[str] = {"Write", "Edit", "write", "edit"}

    def __init__(
        self,
        formatter_callback: Callable[[str], str],
        tool_names: set[str] | None = None,
    ) -> None:
        """Initialise the auto-format pattern.

        Parameters
        ----------
        formatter_callback:
            Callable that receives a file path and returns a status string.
            This is where the actual formatting logic (black, prettier, etc.)
            would be invoked.
        tool_names:
            Override the default set of tool names that trigger formatting.
        """
        self._formatter_callback = formatter_callback
        if tool_names is not None:
            self._FORMAT_TOOL_NAMES = tool_names

    # -- public API --------------------------------------------------------

    def evaluate(self, hook_context: HookContext) -> HookResult:
        """Evaluate a PostToolUse hook context and run the formatter if applicable.

        Parameters
        ----------
        hook_context:
            The ``HookContext`` from a ``PostToolUse`` event.

        Returns
        -------
        HookResult
            With ``additional_context`` describing the formatting outcome.
            Returns an empty ``HookResult`` when the tool is not a Write/Edit.
        """
        if hook_context.tool_name not in self._FORMAT_TOOL_NAMES:
            return HookResult()

        filepath = self._extract_filepath(hook_context)
        if filepath is None:
            logger.debug(
                "AutoFormatPattern: could not extract file path from %s tool input",
                hook_context.tool_name,
            )
            return HookResult()

        try:
            status = self._formatter_callback(filepath)
            logger.info("AutoFormatPattern: formatted %s — %s", filepath, status)
            return HookResult(
                additional_context=f"Auto-formatted {filepath}: {status}",
            )
        except Exception as exc:
            logger.warning(
                "AutoFormatPattern: formatter failed for %s: %s", filepath, exc,
            )
            return HookResult(
                additional_context=f"Auto-format failed for {filepath}: {exc}",
            )

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _extract_filepath(hook_context: HookContext) -> str | None:
        """Extract the target file path from tool_input or tool_response."""
        # Try tool_input first (Write/Edit usually have a path/file_path key)
        if hook_context.tool_input:
            for key in ("file_path", "path", "filepath", "filename"):
                val = hook_context.tool_input.get(key)
                if val and isinstance(val, str):
                    return val

        # Fallback: extract from tool_response (first line often contains path)
        if hook_context.tool_response:
            # Heuristic: look for a path-like string in the first 500 chars
            response_head = hook_context.tool_response[:500]
            match = re.search(r'["\']?([/\\][\w./\\-]+\.\w+)["\']?', response_head)
            if match:
                return match.group(1)

        return None


# ---------------------------------------------------------------------------
# 4. Loop Context Template (Req 27.4)
# ---------------------------------------------------------------------------

# Required keys in a valid loop context structure.
_LOOP_CONTEXT_REQUIRED_KEYS = frozenset({
    "current_task",
    "acceptance_criteria",
    "constraints",
    "learnings",
    "failed_approaches",
    "iteration_count",
    "max_iterations",
})


class LoopContextTemplate:
    """Template for generating and validating self-improving loop context files.

    Generates a ``loop-context.json`` structure compatible with the
    :class:`~src.agents.context_file` module and the
    :class:`~src.agents.loop_runner.LoopRunner`.

    Usage::

        template = LoopContextTemplate()
        context = template.generate(
            task="Implement the auth module",
            criteria=["All tests pass", "No lint errors"],
            constraints=["Do not modify config.py"],
        )
        assert template.validate(context)
    """

    # -- public API --------------------------------------------------------

    def generate(
        self,
        task: str,
        criteria: list[str],
        constraints: list[str] | None = None,
        max_iterations: int = 10,
    ) -> dict:
        """Generate a loop-context.json structure.

        Parameters
        ----------
        task:
            Description of the current task.
        criteria:
            Acceptance criteria that must be met to exit the loop.
        constraints:
            Hard constraints the agent must respect.
        max_iterations:
            Maximum number of loop iterations (default 10).

        Returns
        -------
        dict
            A JSON-serialisable loop context structure.
        """
        return {
            "current_task": task,
            "acceptance_criteria": list(criteria),
            "constraints": list(constraints) if constraints else [],
            "learnings": [],
            "failed_approaches": [],
            "iteration_count": 0,
            "max_iterations": max_iterations,
        }

    def validate(self, context_dict: dict) -> bool:
        """Validate that *context_dict* is a well-formed loop context.

        Parameters
        ----------
        context_dict:
            The dict to validate.

        Returns
        -------
        bool
            ``True`` if valid, ``False`` otherwise.
        """
        errors = self._validate_structure(context_dict)
        if errors:
            logger.debug("LoopContextTemplate: validation errors: %s", errors)
            return False
        return True

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _validate_structure(data: dict) -> list[str]:
        """Return validation error strings (empty list == valid)."""
        errors: list[str] = []

        if not isinstance(data, dict):
            return ["Root must be a dict"]

        missing = _LOOP_CONTEXT_REQUIRED_KEYS - set(data.keys())
        if missing:
            errors.append(f"Missing required keys: {sorted(missing)}")

        if "current_task" in data and not isinstance(data["current_task"], str):
            errors.append("'current_task' must be a string")

        for list_key in ("acceptance_criteria", "constraints", "learnings", "failed_approaches"):
            if list_key in data and not isinstance(data[list_key], list):
                errors.append(f"'{list_key}' must be a list")

        for int_key in ("iteration_count", "max_iterations"):
            if int_key in data and not isinstance(data[int_key], int):
                errors.append(f"'{int_key}' must be an integer")

        if "iteration_count" in data and isinstance(data["iteration_count"], int):
            if data["iteration_count"] < 0:
                errors.append("'iteration_count' must be >= 0")

        if "max_iterations" in data and isinstance(data["max_iterations"], int):
            if data["max_iterations"] < 1:
                errors.append("'max_iterations' must be >= 1")

        return errors


# ---------------------------------------------------------------------------
# 5. Team Task List Template (Req 27.5)
# ---------------------------------------------------------------------------

# Valid task statuses matching src.types.core.TeamTask
_VALID_TASK_STATUSES = {"pending", "claimed", "blocked", "complete"}

# Required keys in each task entry
_TASK_REQUIRED_KEYS = frozenset({"id", "title", "status"})


class TeamTaskListTemplate:
    """Template for generating and validating Agent Teams task list structures.

    Generates a ``tasks.json`` structure compatible with the
    :class:`~src.agents.team_manager.SharedTaskList`.

    Usage::

        template = TeamTaskListTemplate()
        task_list = template.generate(
            team_name="feature-team",
            tasks=[
                {"id": "t1", "title": "Build auth API", "assignee": "backend-agent"},
                {"id": "t2", "title": "Build auth UI", "dependencies": ["t1"]},
            ],
        )
        assert template.validate(task_list)
    """

    # -- public API --------------------------------------------------------

    def generate(
        self,
        team_name: str,
        tasks: list[dict],
    ) -> dict:
        """Generate a tasks.json structure for an Agent Team.

        Parameters
        ----------
        team_name:
            Name of the team (e.g. ``"feature-team"``).
        tasks:
            List of task dicts.  Each must have ``id`` and ``title``.
            Optional keys: ``assignee``, ``status`` (default ``"pending"``),
            ``dependencies`` (default ``[]``), ``files`` (default ``[]``).

        Returns
        -------
        dict
            A JSON-serialisable tasks.json structure::

                {
                    "team_name": str,
                    "created_at": str,  # ISO-8601
                    "tasks": [
                        {
                            "id": str,
                            "title": str,
                            "assignee": str | null,
                            "status": str,
                            "dependencies": list[str],
                            "files": list[str],
                        },
                        ...
                    ],
                }
        """
        normalised_tasks = []
        for task in tasks:
            normalised_tasks.append({
                "id": task["id"],
                "title": task["title"],
                "assignee": task.get("assignee"),
                "status": task.get("status", "pending"),
                "dependencies": list(task.get("dependencies", [])),
                "files": list(task.get("files", [])),
            })

        return {
            "team_name": team_name,
            "created_at": datetime.now(UTC).isoformat(),
            "tasks": normalised_tasks,
        }

    def validate(self, task_list_dict: dict) -> bool:
        """Validate that *task_list_dict* is a well-formed tasks.json.

        Parameters
        ----------
        task_list_dict:
            The dict to validate.

        Returns
        -------
        bool
            ``True`` if valid, ``False`` otherwise.
        """
        errors = self._validate_structure(task_list_dict)
        if errors:
            logger.debug("TeamTaskListTemplate: validation errors: %s", errors)
            return False
        return True

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _validate_structure(data: dict) -> list[str]:
        """Return validation error strings (empty list == valid)."""
        errors: list[str] = []

        if not isinstance(data, dict):
            return ["Root must be a dict"]

        if "team_name" not in data:
            errors.append("Missing required key: 'team_name'")
        elif not isinstance(data["team_name"], str):
            errors.append("'team_name' must be a string")

        if "tasks" not in data:
            errors.append("Missing required key: 'tasks'")
            return errors

        if not isinstance(data["tasks"], list):
            errors.append("'tasks' must be a list")
            return errors

        seen_ids: set[str] = set()
        for i, task in enumerate(data["tasks"]):
            if not isinstance(task, dict):
                errors.append(f"Task at index {i} must be a dict")
                continue

            # Check required keys
            task_missing = _TASK_REQUIRED_KEYS - set(task.keys())
            if task_missing:
                errors.append(
                    f"Task at index {i} missing keys: {sorted(task_missing)}"
                )

            # Check id uniqueness
            task_id = task.get("id")
            if task_id is not None:
                if task_id in seen_ids:
                    errors.append(f"Duplicate task id: '{task_id}'")
                seen_ids.add(task_id)

            # Check status validity
            status = task.get("status")
            if status is not None and status not in _VALID_TASK_STATUSES:
                errors.append(
                    f"Task '{task_id}': invalid status '{status}'. "
                    f"Must be one of {_VALID_TASK_STATUSES}"
                )

            # Check dependencies is a list
            deps = task.get("dependencies")
            if deps is not None and not isinstance(deps, list):
                errors.append(f"Task '{task_id}': 'dependencies' must be a list")

            # Check files is a list
            files = task.get("files")
            if files is not None and not isinstance(files, list):
                errors.append(f"Task '{task_id}': 'files' must be a list")

        # Check dependency references point to existing task IDs
        for task in data["tasks"]:
            if not isinstance(task, dict):
                continue
            for dep_id in task.get("dependencies", []):
                if dep_id not in seen_ids:
                    errors.append(
                        f"Task '{task.get('id')}': dependency '{dep_id}' "
                        f"references a non-existent task"
                    )

        return errors
