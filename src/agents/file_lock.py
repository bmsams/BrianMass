"""File-lock protocol for Agent Teams coordination.

Before editing a file, a teammate checks for .brainmass/locks/{filepath}.lock,
creates the lock with teammate ID and timestamp if absent, breaks stale locks
(>5 minutes), and releases all locks on task completion.

The lock directory is configurable (defaults to .brainmass/locks) to support
testing with temporary directories.

Requirements: 6.4
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Stale lock threshold in seconds (5 minutes)
STALE_LOCK_SECONDS = 5 * 60


@dataclass
class LockInfo:
    """Metadata stored inside a lock file."""
    teammate_id: str
    timestamp: float  # time.time() epoch seconds
    filepath: str


class FileLockManager:
    """Manages file locks for Agent Teams coordination.

    Locks are stored as JSON files at ``{lock_dir}/{safe_path}.lock`` where
    ``safe_path`` replaces path separators with ``__`` to flatten the
    directory structure.

    Requirement 6.4: file-lock-based coordination with stale lock breaking.
    """

    def __init__(self, lock_dir: str = ".brainmass/locks") -> None:
        self._lock_dir = Path(lock_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def acquire_lock(self, filepath: str, teammate_id: str) -> bool:
        """Attempt to acquire a lock on *filepath* for *teammate_id*.

        Returns ``True`` if the lock was acquired (or was already held by
        the same teammate). Returns ``False`` if another teammate holds a
        non-stale lock.

        Stale locks (older than 5 minutes) are automatically broken before
        the new lock is created.
        """
        lock_path = self._lock_path(filepath)
        existing = self._read_lock(lock_path)

        if existing is not None:
            # Already held by the same teammate — idempotent success
            if existing.teammate_id == teammate_id:
                return True

            # Check staleness
            if self._is_stale(existing):
                logger.info(
                    "Breaking stale lock on '%s' held by '%s' (age %.0fs)",
                    filepath,
                    existing.teammate_id,
                    time.time() - existing.timestamp,
                )
                self._delete_lock(lock_path)
            else:
                # Another teammate holds a valid lock
                logger.debug(
                    "Lock on '%s' held by '%s' — cannot acquire for '%s'",
                    filepath,
                    existing.teammate_id,
                    teammate_id,
                )
                return False

        # Create the lock
        self._write_lock(lock_path, LockInfo(
            teammate_id=teammate_id,
            timestamp=time.time(),
            filepath=filepath,
        ))
        return True

    def release_lock(self, filepath: str, teammate_id: str) -> bool:
        """Release the lock on *filepath* if held by *teammate_id*.

        Returns ``True`` if the lock was released. Returns ``False`` if the
        lock does not exist or is held by a different teammate.
        """
        lock_path = self._lock_path(filepath)
        existing = self._read_lock(lock_path)

        if existing is None:
            return False

        if existing.teammate_id != teammate_id:
            logger.debug(
                "Cannot release lock on '%s': held by '%s', not '%s'",
                filepath,
                existing.teammate_id,
                teammate_id,
            )
            return False

        self._delete_lock(lock_path)
        return True

    def release_all_locks(self, teammate_id: str) -> int:
        """Release all locks held by *teammate_id*.

        Returns the number of locks released. Intended to be called on task
        completion to clean up.
        """
        released = 0
        if not self._lock_dir.exists():
            return released

        for lock_file in self._lock_dir.iterdir():
            if not lock_file.suffix == ".lock":
                continue
            info = self._read_lock(lock_file)
            if info is not None and info.teammate_id == teammate_id:
                self._delete_lock(lock_file)
                released += 1

        return released

    def is_locked(self, filepath: str) -> bool:
        """Check whether *filepath* is currently locked (non-stale)."""
        lock_path = self._lock_path(filepath)
        existing = self._read_lock(lock_path)
        if existing is None:
            return False
        if self._is_stale(existing):
            return False
        return True

    def get_lock_info(self, filepath: str) -> LockInfo | None:
        """Return the lock info for *filepath*, or ``None`` if unlocked.

        Stale locks are treated as absent (returns ``None``).
        """
        lock_path = self._lock_path(filepath)
        existing = self._read_lock(lock_path)
        if existing is None:
            return None
        if self._is_stale(existing):
            return None
        return existing

    def get_all_locks(self) -> list[LockInfo]:
        """Return all non-stale locks currently held."""
        locks: list[LockInfo] = []
        if not self._lock_dir.exists():
            return locks

        for lock_file in self._lock_dir.iterdir():
            if not lock_file.suffix == ".lock":
                continue
            info = self._read_lock(lock_file)
            if info is not None and not self._is_stale(info):
                locks.append(info)

        return locks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _lock_path(self, filepath: str) -> Path:
        """Convert a filepath to its corresponding lock file path.

        Path separators are replaced with ``__`` to flatten the structure.
        """
        # Normalise the filepath to use forward slashes, then replace
        safe = filepath.replace("\\", "/").replace("/", "__")
        return self._lock_dir / f"{safe}.lock"

    def _read_lock(self, lock_path: Path) -> LockInfo | None:
        """Read and parse a lock file. Returns ``None`` if missing or corrupt."""
        if not lock_path.exists():
            return None
        try:
            data = json.loads(lock_path.read_text(encoding="utf-8"))
            return LockInfo(
                teammate_id=data["teammate_id"],
                timestamp=float(data["timestamp"]),
                filepath=data["filepath"],
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Corrupt lock file '%s': %s", lock_path, exc)
            return None

    def _write_lock(self, lock_path: Path, info: LockInfo) -> None:
        """Write a lock file as JSON."""
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "teammate_id": info.teammate_id,
            "timestamp": info.timestamp,
            "filepath": info.filepath,
        }
        lock_path.write_text(json.dumps(data), encoding="utf-8")

    def _delete_lock(self, lock_path: Path) -> None:
        """Delete a lock file if it exists."""
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass

    @staticmethod
    def _is_stale(info: LockInfo) -> bool:
        """Return ``True`` if the lock is older than the stale threshold."""
        return (time.time() - info.timestamp) > STALE_LOCK_SECONDS
