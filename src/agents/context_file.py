"""Context file management for self-improving loop iterations.

Manages .brainmass/loop-context.json — the structured context file that
carries state between loop iterations (Requirement 7.2).

Each iteration of the "Ralph Wiggum" pattern reads the context file to
understand the current task, constraints, accumulated learnings, and
failed approaches, then writes back updated state after completing work.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from src.types.core import LoopContext

DEFAULT_CONTEXT_PATH = os.path.join(".brainmass", "loop-context.json")

# Required top-level keys in a valid context file.
_REQUIRED_KEYS = frozenset(
    {
        "current_task",
        "acceptance_criteria",
        "constraints",
        "learnings",
        "failed_approaches",
        "iteration_count",
        "max_iterations",
    }
)


def _validate_raw(data: dict) -> list[str]:
    """Return a list of validation error strings (empty == valid)."""
    errors: list[str] = []

    missing = _REQUIRED_KEYS - set(data.keys())
    if missing:
        errors.append(f"Missing required keys: {sorted(missing)}")

    if "current_task" in data and not isinstance(data["current_task"], str):
        errors.append("'current_task' must be a string")

    for list_field in ("acceptance_criteria", "constraints", "learnings", "failed_approaches"):
        if list_field in data and not isinstance(data[list_field], list):
            errors.append(f"'{list_field}' must be a list")

    for int_field in ("iteration_count", "max_iterations"):
        if int_field in data and not isinstance(data[int_field], int):
            errors.append(f"'{int_field}' must be an integer")

    return errors


def serialize(context: LoopContext) -> dict:
    """Convert a LoopContext dataclass to a plain dict suitable for JSON."""
    return {
        "current_task": context.current_task,
        "acceptance_criteria": list(context.acceptance_criteria),
        "constraints": list(context.constraints),
        "learnings": [dict(l) for l in context.learnings],
        "failed_approaches": [dict(f) for f in context.failed_approaches],
        "iteration_count": context.iteration_count,
        "max_iterations": context.max_iterations,
    }


def deserialize(data: dict) -> LoopContext:
    """Build a LoopContext from a validated dict.

    Raises ``ValueError`` if *data* fails validation.
    """
    errors = _validate_raw(data)
    if errors:
        raise ValueError(f"Invalid loop context data: {'; '.join(errors)}")

    return LoopContext(
        current_task=data["current_task"],
        acceptance_criteria=list(data["acceptance_criteria"]),
        constraints=list(data["constraints"]),
        learnings=[dict(l) for l in data["learnings"]],
        failed_approaches=[dict(f) for f in data["failed_approaches"]],
        iteration_count=data["iteration_count"],
        max_iterations=data["max_iterations"],
    )


def save(context: LoopContext, path: str = DEFAULT_CONTEXT_PATH) -> None:
    """Persist *context* to *path* as JSON.

    * Creates parent directories if they don't exist.
    * Uses atomic write (write to temp file in the same directory, then rename)
      to avoid partial writes on crash.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    payload = json.dumps(serialize(context), indent=2)

    # Atomic write: temp file in the same dir → rename.
    fd, tmp_path = tempfile.mkstemp(
        dir=str(target.parent), suffix=".tmp", prefix=".loop-ctx-"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        # os.replace is atomic on both POSIX and Windows (same volume).
        os.replace(tmp_path, str(target))
    except BaseException:
        # Clean up the temp file on any failure.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load(path: str = DEFAULT_CONTEXT_PATH) -> LoopContext | None:
    """Load a LoopContext from *path*.

    Returns ``None`` when the file does not exist.
    Raises ``ValueError`` on invalid / corrupt data.
    Raises ``json.JSONDecodeError`` on malformed JSON.
    """
    target = Path(path)
    if not target.exists():
        return None

    raw = target.read_text(encoding="utf-8")
    data = json.loads(raw)

    if not isinstance(data, dict):
        raise ValueError("Loop context file must contain a JSON object")

    return deserialize(data)
