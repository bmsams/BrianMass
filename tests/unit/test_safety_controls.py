"""Unit tests for safety controls module.

Tests the four composable safety mechanisms:
- ErrorMonitor: auto-pause on repeated errors
- DiffSizeChecker: diff size limits and file scope validation
- StopFileChecker: .auto/stop sentinel file detection
- AcceptanceCriteriaGate: acceptance criteria validation

Requirements: 7.5
"""

from __future__ import annotations

import os

from src.agents.safety_controls import (
    AcceptanceCriteriaGate,
    AcceptanceGateResult,
    DiffCheckResult,
    DiffSizeChecker,
    ErrorMonitor,
    StopFileChecker,
)

# =========================================================================
# ErrorMonitor
# =========================================================================

class TestErrorMonitor:
    """Tests for ErrorMonitor — auto-pause on 3+ same errors."""

    def test_no_errors_no_pause(self):
        monitor = ErrorMonitor()
        assert monitor.should_pause is False
        assert monitor.pause_reason is None

    def test_single_error_no_pause(self):
        monitor = ErrorMonitor()
        monitor.record_error("connection timeout")
        assert monitor.should_pause is False

    def test_two_errors_no_pause(self):
        monitor = ErrorMonitor()
        monitor.record_error("connection timeout")
        monitor.record_error("connection timeout")
        assert monitor.should_pause is False

    def test_three_same_errors_triggers_pause(self):
        monitor = ErrorMonitor()
        for _ in range(3):
            monitor.record_error("connection timeout")
        assert monitor.should_pause is True
        assert "connection timeout" in monitor.pause_reason
        assert "3" in monitor.pause_reason

    def test_different_errors_no_pause(self):
        monitor = ErrorMonitor()
        monitor.record_error("error A")
        monitor.record_error("error B")
        monitor.record_error("error C")
        assert monitor.should_pause is False

    def test_custom_threshold(self):
        monitor = ErrorMonitor(threshold=2)
        monitor.record_error("fail")
        assert monitor.should_pause is False
        monitor.record_error("fail")
        assert monitor.should_pause is True

    def test_threshold_one(self):
        monitor = ErrorMonitor(threshold=1)
        monitor.record_error("boom")
        assert monitor.should_pause is True

    def test_empty_error_ignored(self):
        monitor = ErrorMonitor(threshold=1)
        monitor.record_error("")
        assert monitor.should_pause is False

    def test_reset_clears_state(self):
        monitor = ErrorMonitor()
        for _ in range(3):
            monitor.record_error("fail")
        assert monitor.should_pause is True
        monitor.reset()
        assert monitor.should_pause is False
        assert monitor.pause_reason is None
        assert monitor.error_counts == {}

    def test_error_counts_snapshot(self):
        monitor = ErrorMonitor()
        monitor.record_error("a")
        monitor.record_error("a")
        monitor.record_error("b")
        assert monitor.error_counts == {"a": 2, "b": 1}

    def test_pause_reason_truncates_long_message(self):
        monitor = ErrorMonitor(threshold=1)
        long_msg = "x" * 300
        monitor.record_error(long_msg)
        assert monitor.should_pause is True
        # Reason should contain a truncated version (max 200 chars of the error)
        assert len(monitor.pause_reason) < 350

    def test_more_than_threshold_stays_paused(self):
        monitor = ErrorMonitor(threshold=3)
        for _ in range(5):
            monitor.record_error("same error")
        assert monitor.should_pause is True
        assert "5" in monitor.pause_reason


# =========================================================================
# DiffSizeChecker
# =========================================================================

class TestDiffSizeChecker:
    """Tests for DiffSizeChecker — diff size limits and scope validation."""

    def test_within_limits_passes(self):
        checker = DiffSizeChecker(max_diff_lines=500)
        result = checker.check(changed_lines=100, files_modified=["src/main.py"])
        assert result.ok is True
        assert result.reason is None

    def test_exact_limit_passes(self):
        checker = DiffSizeChecker(max_diff_lines=500)
        result = checker.check(changed_lines=500, files_modified=[])
        assert result.ok is True

    def test_exceeds_limit_fails(self):
        checker = DiffSizeChecker(max_diff_lines=500)
        result = checker.check(changed_lines=501, files_modified=[])
        assert result.ok is False
        assert "501" in result.reason
        assert "500" in result.reason

    def test_zero_lines_passes(self):
        checker = DiffSizeChecker(max_diff_lines=500)
        result = checker.check(changed_lines=0, files_modified=[])
        assert result.ok is True

    def test_custom_limit(self):
        checker = DiffSizeChecker(max_diff_lines=10)
        result = checker.check(changed_lines=11, files_modified=[])
        assert result.ok is False

    def test_no_allowed_paths_all_files_ok(self):
        checker = DiffSizeChecker(allowed_paths=None)
        result = checker.check(
            changed_lines=10,
            files_modified=["anywhere/file.py", "/etc/passwd"],
        )
        assert result.ok is True

    def test_allowed_paths_in_scope(self):
        checker = DiffSizeChecker(allowed_paths=["src/", "tests/"])
        result = checker.check(
            changed_lines=10,
            files_modified=["src/main.py", "tests/test_main.py"],
        )
        assert result.ok is True

    def test_allowed_paths_out_of_scope(self):
        checker = DiffSizeChecker(allowed_paths=["src/"])
        result = checker.check(
            changed_lines=10,
            files_modified=["src/main.py", "docs/readme.md"],
        )
        assert result.ok is False
        assert "docs/readme.md" in result.out_of_scope_files

    def test_allowed_paths_all_out_of_scope(self):
        checker = DiffSizeChecker(allowed_paths=["src/"])
        result = checker.check(
            changed_lines=10,
            files_modified=["docs/a.md", "config/b.yaml"],
        )
        assert result.ok is False
        assert len(result.out_of_scope_files) == 2

    def test_diff_size_checked_before_scope(self):
        """If diff size exceeds limit, scope check is skipped."""
        checker = DiffSizeChecker(max_diff_lines=10, allowed_paths=["src/"])
        result = checker.check(
            changed_lines=100,
            files_modified=["docs/readme.md"],
        )
        assert result.ok is False
        assert "100" in result.reason
        # out_of_scope_files should be empty since we aborted on size
        assert result.out_of_scope_files == []

    def test_backslash_paths_normalised(self):
        checker = DiffSizeChecker(allowed_paths=["src/"])
        result = checker.check(
            changed_lines=5,
            files_modified=["src\\main.py"],
        )
        assert result.ok is True

    def test_empty_files_list(self):
        checker = DiffSizeChecker(allowed_paths=["src/"])
        result = checker.check(changed_lines=5, files_modified=[])
        assert result.ok is True


# =========================================================================
# StopFileChecker
# =========================================================================

class TestStopFileChecker:
    """Tests for StopFileChecker — .auto/stop sentinel detection."""

    def test_no_stop_file(self, tmp_path):
        checker = StopFileChecker(cwd=str(tmp_path))
        assert checker.should_stop() is False

    def test_stop_file_exists(self, tmp_path):
        auto_dir = tmp_path / ".auto"
        auto_dir.mkdir()
        (auto_dir / "stop").write_text("stop")
        checker = StopFileChecker(cwd=str(tmp_path))
        assert checker.should_stop() is True

    def test_stop_file_path_property(self, tmp_path):
        checker = StopFileChecker(cwd=str(tmp_path))
        expected = os.path.join(str(tmp_path), ".auto", "stop")
        assert checker.stop_file_path == expected

    def test_directory_not_file(self, tmp_path):
        """A directory named 'stop' should not trigger the check."""
        auto_dir = tmp_path / ".auto"
        auto_dir.mkdir()
        (auto_dir / "stop").mkdir()
        checker = StopFileChecker(cwd=str(tmp_path))
        assert checker.should_stop() is False

    def test_default_cwd(self):
        checker = StopFileChecker()
        assert checker.stop_file_path == os.path.join(".", ".auto", "stop")

    def test_empty_stop_file(self, tmp_path):
        """An empty stop file should still trigger."""
        auto_dir = tmp_path / ".auto"
        auto_dir.mkdir()
        (auto_dir / "stop").write_text("")
        checker = StopFileChecker(cwd=str(tmp_path))
        assert checker.should_stop() is True


# =========================================================================
# AcceptanceCriteriaGate
# =========================================================================

class TestAcceptanceCriteriaGate:
    """Tests for AcceptanceCriteriaGate — acceptance criteria validation."""

    def test_no_criteria_passes(self):
        gate = AcceptanceCriteriaGate(criteria=[])
        result = gate.check({})
        assert result.passed is True

    def test_all_criteria_met(self):
        gate = AcceptanceCriteriaGate(criteria=["tests pass", "lint clean"])
        result = gate.check({"tests pass": True, "lint clean": True})
        assert result.passed is True
        assert result.met == ["tests pass", "lint clean"]
        assert result.unmet == []

    def test_some_criteria_unmet(self):
        gate = AcceptanceCriteriaGate(criteria=["tests pass", "lint clean"])
        result = gate.check({"tests pass": True, "lint clean": False})
        assert result.passed is False
        assert result.met == ["tests pass"]
        assert result.unmet == ["lint clean"]

    def test_all_criteria_unmet(self):
        gate = AcceptanceCriteriaGate(criteria=["a", "b"])
        result = gate.check({"a": False, "b": False})
        assert result.passed is False
        assert result.unmet == ["a", "b"]

    def test_missing_criteria_treated_as_unmet(self):
        gate = AcceptanceCriteriaGate(criteria=["a", "b"])
        result = gate.check({"a": True})
        assert result.passed is False
        assert result.unmet == ["b"]

    def test_empty_results_all_unmet(self):
        gate = AcceptanceCriteriaGate(criteria=["a", "b"])
        result = gate.check({})
        assert result.passed is False
        assert len(result.unmet) == 2

    def test_extra_results_ignored(self):
        gate = AcceptanceCriteriaGate(criteria=["a"])
        result = gate.check({"a": True, "b": True, "c": False})
        assert result.passed is True
        assert result.met == ["a"]

    def test_single_criterion_met(self):
        gate = AcceptanceCriteriaGate(criteria=["deploy works"])
        result = gate.check({"deploy works": True})
        assert result.passed is True

    def test_single_criterion_unmet(self):
        gate = AcceptanceCriteriaGate(criteria=["deploy works"])
        result = gate.check({"deploy works": False})
        assert result.passed is False


# =========================================================================
# DiffCheckResult / AcceptanceGateResult dataclass defaults
# =========================================================================

class TestResultDataclasses:
    """Verify default field values on result dataclasses."""

    def test_diff_check_result_defaults(self):
        r = DiffCheckResult(ok=True)
        assert r.reason is None
        assert r.out_of_scope_files == []

    def test_acceptance_gate_result_defaults(self):
        r = AcceptanceGateResult(passed=False)
        assert r.met == []
        assert r.unmet == []
