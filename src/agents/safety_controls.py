"""Safety controls for self-improving loop runner.

Provides four composable safety mechanisms:
1. ErrorMonitor — auto-pause on repeated errors (3+ same error)
2. DiffSizeChecker — abort on oversized diffs or out-of-scope files
3. StopFileChecker — check for .auto/stop sentinel file
4. AcceptanceCriteriaGate — validate acceptance criteria before marking complete

Requirements: 7.5
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Error Monitor
# ---------------------------------------------------------------------------

@dataclass
class ErrorMonitor:
    """Tracks error messages and auto-pauses when the same error appears 3+ times.

    The threshold is configurable (default 3). Errors are tracked by their
    exact message string. Call ``record_error`` after each iteration and
    check ``should_pause`` to decide whether to halt the loop.
    """

    threshold: int = 3
    _error_counts: Counter = field(default_factory=Counter)
    _paused: bool = False
    _pause_reason: str | None = None

    # -- public API --------------------------------------------------------

    def record_error(self, error_message: str) -> None:
        """Record an error message from the current iteration."""
        if not error_message:
            return
        self._error_counts[error_message] += 1
        count = self._error_counts[error_message]
        if count >= self.threshold:
            self._paused = True
            self._pause_reason = (
                f"Error repeated {count} times (threshold={self.threshold}): "
                f"{error_message[:200]}"
            )
            logger.warning("ErrorMonitor: %s", self._pause_reason)

    @property
    def should_pause(self) -> bool:
        """Return True if the loop should be paused."""
        return self._paused

    @property
    def pause_reason(self) -> str | None:
        """Human-readable reason for the pause, or None."""
        return self._pause_reason

    def reset(self) -> None:
        """Clear all tracked errors and un-pause."""
        self._error_counts.clear()
        self._paused = False
        self._pause_reason = None

    @property
    def error_counts(self) -> dict[str, int]:
        """Return a snapshot of current error counts."""
        return dict(self._error_counts)


# ---------------------------------------------------------------------------
# Diff Size Checker
# ---------------------------------------------------------------------------

@dataclass
class DiffSizeChecker:
    """Checks diff size against configurable limits and validates file scope.

    - ``max_diff_lines``: maximum number of changed lines allowed (default 500).
    - ``allowed_paths``: optional list of path prefixes that are in scope.
      If set, any file outside these prefixes triggers an abort.
    """

    max_diff_lines: int = 500
    allowed_paths: list[str] | None = None

    # -- public API --------------------------------------------------------

    def check(
        self,
        changed_lines: int,
        files_modified: list[str],
    ) -> DiffCheckResult:
        """Validate a diff against size and scope constraints.

        Returns a ``DiffCheckResult`` indicating whether the diff is acceptable.
        """
        # Check diff size
        if changed_lines > self.max_diff_lines:
            reason = (
                f"Diff size {changed_lines} lines exceeds limit of "
                f"{self.max_diff_lines} lines"
            )
            logger.warning("DiffSizeChecker: %s", reason)
            return DiffCheckResult(ok=False, reason=reason)

        # Check file scope
        if self.allowed_paths is not None:
            out_of_scope = self._find_out_of_scope(files_modified)
            if out_of_scope:
                reason = (
                    f"Files outside allowed scope: {out_of_scope}"
                )
                logger.warning("DiffSizeChecker: %s", reason)
                return DiffCheckResult(
                    ok=False,
                    reason=reason,
                    out_of_scope_files=out_of_scope,
                )

        return DiffCheckResult(ok=True)

    # -- internals ---------------------------------------------------------

    def _find_out_of_scope(self, files: list[str]) -> list[str]:
        """Return files that don't match any allowed path prefix."""
        if self.allowed_paths is None:
            return []
        out = []
        for f in files:
            normalised = f.replace("\\", "/")
            if not any(
                normalised.startswith(p.replace("\\", "/"))
                for p in self.allowed_paths
            ):
                out.append(f)
        return out


@dataclass
class DiffCheckResult:
    """Result of a diff size / scope check."""

    ok: bool
    reason: str | None = None
    out_of_scope_files: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Stop File Checker
# ---------------------------------------------------------------------------

class StopFileChecker:
    """Checks for the existence of a ``.auto/stop`` sentinel file.

    The sentinel path is relative to the working directory. When the file
    exists the loop should halt gracefully at the next iteration boundary.
    """

    def __init__(self, cwd: str = ".") -> None:
        self._cwd = cwd

    @property
    def stop_file_path(self) -> str:
        """Absolute path to the sentinel file."""
        return os.path.join(self._cwd, ".auto", "stop")

    def should_stop(self) -> bool:
        """Return True if the stop sentinel file exists."""
        exists = os.path.isfile(self.stop_file_path)
        if exists:
            logger.info("StopFileChecker: stop file detected at %s", self.stop_file_path)
        return exists


# ---------------------------------------------------------------------------
# Acceptance Criteria Gate
# ---------------------------------------------------------------------------

@dataclass
class AcceptanceCriteriaGate:
    """Validates that all acceptance criteria are met before marking complete.

    Criteria are represented as a list of strings. The gate receives a
    mapping of criterion → bool (met or not) and decides whether the
    overall gate passes.
    """

    criteria: list[str] = field(default_factory=list)

    def check(self, results: dict[str, bool]) -> AcceptanceGateResult:
        """Check whether all acceptance criteria pass.

        Args:
            results: mapping of criterion text → True/False.

        Returns:
            AcceptanceGateResult with pass/fail and details.
        """
        if not self.criteria:
            return AcceptanceGateResult(passed=True)

        met: list[str] = []
        unmet: list[str] = []

        for criterion in self.criteria:
            if results.get(criterion, False):
                met.append(criterion)
            else:
                unmet.append(criterion)

        passed = len(unmet) == 0
        if not passed:
            logger.info(
                "AcceptanceCriteriaGate: %d/%d criteria met, %d unmet",
                len(met),
                len(self.criteria),
                len(unmet),
            )

        return AcceptanceGateResult(
            passed=passed,
            met=met,
            unmet=unmet,
        )


@dataclass
class AcceptanceGateResult:
    """Result of an acceptance criteria gate check."""

    passed: bool
    met: list[str] = field(default_factory=list)
    unmet: list[str] = field(default_factory=list)
