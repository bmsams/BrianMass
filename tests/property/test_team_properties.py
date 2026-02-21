"""Property-based tests for Agent Teams coordination.

Properties covered:
- Property 11: File-lock mutual exclusion
"""

from __future__ import annotations

import tempfile
import time
from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.agents.file_lock import STALE_LOCK_SECONDS, FileLockManager

PATH_SEGMENT = st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=10)
FILE_PATH = st.lists(PATH_SEGMENT, min_size=1, max_size=4).map(lambda p: "/".join(p) + ".py")
TEAMMATE_ID = st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789-", min_size=1, max_size=12)


@pytest.mark.property
@settings(max_examples=100)
@given(
    filepath=FILE_PATH,
    teammates=st.tuples(TEAMMATE_ID, TEAMMATE_ID).filter(lambda t: t[0] != t[1]),
)
def test_property_11_file_lock_mutual_exclusion(
    filepath: str,
    teammates: tuple[str, str],
) -> None:
    """Feature: claude-code-v3-enterprise, Property 11."""
    first, second = teammates
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = FileLockManager(lock_dir=tmpdir)

        assert manager.acquire_lock(filepath, first) is True
        assert manager.acquire_lock(filepath, second) is False
        info = manager.get_lock_info(filepath)
        assert info is not None
        assert info.teammate_id == first

        assert manager.release_lock(filepath, first) is True
        assert manager.acquire_lock(filepath, second) is True


@pytest.mark.property
@settings(max_examples=100)
@given(
    filepath=FILE_PATH,
    stale_offset=st.integers(
        min_value=STALE_LOCK_SECONDS + 1,
        max_value=STALE_LOCK_SECONDS * 5,
    ),
    teammates=st.tuples(TEAMMATE_ID, TEAMMATE_ID).filter(lambda t: t[0] != t[1]),
)
def test_property_11_stale_locks_are_breakable(
    filepath: str,
    stale_offset: int,
    teammates: tuple[str, str],
) -> None:
    """Feature: claude-code-v3-enterprise, Property 11."""
    first, second = teammates
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = FileLockManager(lock_dir=tmpdir)
        assert manager.acquire_lock(filepath, first) is True

        stale_time = time.time() + stale_offset
        with patch("src.agents.file_lock.time") as mocked_time:
            mocked_time.time.return_value = stale_time
            assert manager.acquire_lock(filepath, second) is True

        info = manager.get_lock_info(filepath)
        assert info is not None
        assert info.teammate_id == second

