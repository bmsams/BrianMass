"""Unit tests for the file-lock protocol (src/agents/file_lock.py).

Requirements: 6.4
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from src.agents.file_lock import (
    STALE_LOCK_SECONDS,
    FileLockManager,
)


@pytest.fixture()
def lock_dir(tmp_path: Path) -> Path:
    """Return a temporary lock directory."""
    d = tmp_path / "locks"
    d.mkdir()
    return d


@pytest.fixture()
def mgr(lock_dir: Path) -> FileLockManager:
    """Return a FileLockManager using a temp directory."""
    return FileLockManager(lock_dir=str(lock_dir))


# ---------------------------------------------------------------------------
# acquire_lock
# ---------------------------------------------------------------------------

class TestAcquireLock:
    def test_acquire_creates_lock_file(self, mgr: FileLockManager, lock_dir: Path):
        assert mgr.acquire_lock("src/main.py", "agent-1") is True
        lock_files = list(lock_dir.glob("*.lock"))
        assert len(lock_files) == 1

    def test_acquire_stores_correct_data(self, mgr: FileLockManager):
        mgr.acquire_lock("src/main.py", "agent-1")
        info = mgr.get_lock_info("src/main.py")
        assert info is not None
        assert info.teammate_id == "agent-1"
        assert info.filepath == "src/main.py"

    def test_acquire_idempotent_same_teammate(self, mgr: FileLockManager):
        assert mgr.acquire_lock("f.py", "agent-1") is True
        assert mgr.acquire_lock("f.py", "agent-1") is True

    def test_acquire_fails_when_held_by_other(self, mgr: FileLockManager):
        mgr.acquire_lock("f.py", "agent-1")
        assert mgr.acquire_lock("f.py", "agent-2") is False

    def test_acquire_breaks_stale_lock(self, mgr: FileLockManager):
        mgr.acquire_lock("f.py", "agent-1")
        # Simulate staleness by patching time.time
        stale_time = time.time() + STALE_LOCK_SECONDS + 1
        with patch("src.agents.file_lock.time") as mock_time:
            mock_time.time.return_value = stale_time
            assert mgr.acquire_lock("f.py", "agent-2") is True
        info = mgr.get_lock_info("f.py")
        assert info is not None
        assert info.teammate_id == "agent-2"

    def test_acquire_creates_lock_dir_if_missing(self, tmp_path: Path):
        lock_dir = tmp_path / "nonexistent" / "locks"
        mgr = FileLockManager(lock_dir=str(lock_dir))
        assert mgr.acquire_lock("f.py", "agent-1") is True
        assert lock_dir.exists()

    def test_acquire_multiple_files(self, mgr: FileLockManager):
        assert mgr.acquire_lock("a.py", "agent-1") is True
        assert mgr.acquire_lock("b.py", "agent-1") is True
        assert mgr.acquire_lock("c.py", "agent-2") is True


# ---------------------------------------------------------------------------
# release_lock
# ---------------------------------------------------------------------------

class TestReleaseLock:
    def test_release_removes_lock(self, mgr: FileLockManager):
        mgr.acquire_lock("f.py", "agent-1")
        assert mgr.release_lock("f.py", "agent-1") is True
        assert mgr.is_locked("f.py") is False

    def test_release_nonexistent_returns_false(self, mgr: FileLockManager):
        assert mgr.release_lock("f.py", "agent-1") is False

    def test_release_wrong_teammate_returns_false(self, mgr: FileLockManager):
        mgr.acquire_lock("f.py", "agent-1")
        assert mgr.release_lock("f.py", "agent-2") is False
        # Lock should still be held
        assert mgr.is_locked("f.py") is True

    def test_release_after_release_returns_false(self, mgr: FileLockManager):
        mgr.acquire_lock("f.py", "agent-1")
        mgr.release_lock("f.py", "agent-1")
        assert mgr.release_lock("f.py", "agent-1") is False


# ---------------------------------------------------------------------------
# release_all_locks
# ---------------------------------------------------------------------------

class TestReleaseAllLocks:
    def test_release_all_for_teammate(self, mgr: FileLockManager):
        mgr.acquire_lock("a.py", "agent-1")
        mgr.acquire_lock("b.py", "agent-1")
        mgr.acquire_lock("c.py", "agent-2")
        released = mgr.release_all_locks("agent-1")
        assert released == 2
        assert mgr.is_locked("a.py") is False
        assert mgr.is_locked("b.py") is False
        assert mgr.is_locked("c.py") is True

    def test_release_all_no_locks(self, mgr: FileLockManager):
        assert mgr.release_all_locks("agent-1") == 0

    def test_release_all_nonexistent_dir(self, tmp_path: Path):
        mgr = FileLockManager(lock_dir=str(tmp_path / "nope"))
        assert mgr.release_all_locks("agent-1") == 0


# ---------------------------------------------------------------------------
# is_locked / get_lock_info
# ---------------------------------------------------------------------------

class TestLockQueries:
    def test_is_locked_true(self, mgr: FileLockManager):
        mgr.acquire_lock("f.py", "agent-1")
        assert mgr.is_locked("f.py") is True

    def test_is_locked_false_when_no_lock(self, mgr: FileLockManager):
        assert mgr.is_locked("f.py") is False

    def test_is_locked_false_when_stale(self, mgr: FileLockManager):
        mgr.acquire_lock("f.py", "agent-1")
        stale_time = time.time() + STALE_LOCK_SECONDS + 1
        with patch("src.agents.file_lock.time") as mock_time:
            mock_time.time.return_value = stale_time
            assert mgr.is_locked("f.py") is False

    def test_get_lock_info_returns_none_when_stale(self, mgr: FileLockManager):
        mgr.acquire_lock("f.py", "agent-1")
        stale_time = time.time() + STALE_LOCK_SECONDS + 1
        with patch("src.agents.file_lock.time") as mock_time:
            mock_time.time.return_value = stale_time
            assert mgr.get_lock_info("f.py") is None

    def test_get_lock_info_returns_none_when_absent(self, mgr: FileLockManager):
        assert mgr.get_lock_info("f.py") is None


# ---------------------------------------------------------------------------
# get_all_locks
# ---------------------------------------------------------------------------

class TestGetAllLocks:
    def test_returns_all_non_stale(self, mgr: FileLockManager):
        mgr.acquire_lock("a.py", "agent-1")
        mgr.acquire_lock("b.py", "agent-2")
        locks = mgr.get_all_locks()
        assert len(locks) == 2
        ids = {l.teammate_id for l in locks}
        assert ids == {"agent-1", "agent-2"}

    def test_empty_when_no_locks(self, mgr: FileLockManager):
        assert mgr.get_all_locks() == []

    def test_empty_dir_not_created(self, tmp_path: Path):
        mgr = FileLockManager(lock_dir=str(tmp_path / "nope"))
        assert mgr.get_all_locks() == []


# ---------------------------------------------------------------------------
# Path normalisation
# ---------------------------------------------------------------------------

class TestPathNormalisation:
    def test_forward_slashes(self, mgr: FileLockManager):
        mgr.acquire_lock("src/agents/file.py", "agent-1")
        assert mgr.is_locked("src/agents/file.py") is True

    def test_backslashes_normalised(self, mgr: FileLockManager):
        mgr.acquire_lock("src\\agents\\file.py", "agent-1")
        # Should be accessible via forward slashes too
        assert mgr.is_locked("src/agents/file.py") is True


# ---------------------------------------------------------------------------
# Corrupt lock file handling
# ---------------------------------------------------------------------------

class TestCorruptLockFile:
    def test_corrupt_json_treated_as_absent(self, mgr: FileLockManager, lock_dir: Path):
        lock_file = lock_dir / "f.py.lock"
        lock_file.write_text("not json", encoding="utf-8")
        assert mgr.is_locked("f.py") is False
        # Should be able to acquire over corrupt lock
        assert mgr.acquire_lock("f.py", "agent-1") is True

    def test_missing_fields_treated_as_absent(self, mgr: FileLockManager, lock_dir: Path):
        lock_file = lock_dir / "f.py.lock"
        lock_file.write_text(json.dumps({"teammate_id": "x"}), encoding="utf-8")
        assert mgr.is_locked("f.py") is False
