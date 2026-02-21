"""Built-in agent templates as Markdown + YAML frontmatter files.

This package contains the five standard agent templates shipped with
BrainMass.  Each ``.md`` file is a self-contained agent definition whose
YAML frontmatter declares configuration and whose Markdown body provides
the system prompt.

Requirements: 8.12 (Task 17.4)
"""

from __future__ import annotations

from pathlib import Path

TEMPLATES_DIR: Path = Path(__file__).resolve().parent
"""Absolute path to the directory containing the built-in template files."""

TEMPLATE_FILES: list[str] = [
    "code-reviewer.md",
    "security-auditor.md",
    "implementer-tester.md",
    "researcher.md",
    "architect.md",
]
"""Ordered list of built-in template filenames."""


def get_template_paths() -> list[Path]:
    """Return resolved paths for every built-in template file."""
    return [TEMPLATES_DIR / fname for fname in TEMPLATE_FILES]
