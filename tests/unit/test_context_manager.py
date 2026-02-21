"""Unit tests for the Context Manager.

Tests cover add_item, staleness scoring, compaction, session persistence,
health metrics, learning store, and configurable thresholds.

Requirements: 2.3, 2.4, 2.5, 2.7, 2.8, 2.9, 2.10, 2.12
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.context.context_manager import AgentCoreMemoryStore, CompactionResult, ContextManager
from src.types.core import ContextCategory

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_session_dir(tmp_path: Path) -> str:
    """Return a temporary directory path for session files."""
    d = tmp_path / ".brainmass"
    d.mkdir()
    return str(d)


@pytest.fixture
def cm(tmp_session_dir: str) -> ContextManager:
    """A fresh ContextManager with 200K window and temp session dir."""
    return ContextManager(
        session_id="test-session",
        window_size=200_000,
        session_dir=tmp_session_dir,
    )


# ---------------------------------------------------------------------------
# add_item
# ---------------------------------------------------------------------------

class TestAddItem:
    """Tests for ContextManager.add_item()."""

    def test_adds_item_to_list(self, cm: ContextManager):
        item = cm.add_item("some content", "user", 100)
        assert len(cm.items) == 1
        assert cm.items[0] is item

    def test_classifies_via_triage(self, cm: ContextManager):
        # Stack trace → PRESERVE_VERBATIM
        item = cm.add_item("Error: fail at line 5", "assistant", 50)
        assert item.category == ContextCategory.PRESERVE_VERBATIM

    def test_classifies_structured(self, cm: ContextManager):
        item = cm.add_item("DECISION: Use PostgreSQL", "user", 30)
        assert item.category == ContextCategory.PRESERVE_STRUCTURED

    def test_classifies_ephemeral(self, cm: ContextManager):
        item = cm.add_item("Searching for files...", "tool_call", 20)
        assert item.category == ContextCategory.EPHEMERAL

    def test_classifies_compress(self, cm: ContextManager):
        item = cm.add_item("Let me think about this approach.", "user", 40)
        assert item.category == ContextCategory.COMPRESS_AGGRESSIVE

    def test_custom_id(self, cm: ContextManager):
        item = cm.add_item("content", "user", 10, item_id="my-id")
        assert item.id == "my-id"

    def test_auto_generated_id(self, cm: ContextManager):
        item = cm.add_item("content", "user", 10)
        assert item.id  # non-empty

    def test_initial_staleness_zero(self, cm: ContextManager):
        item = cm.add_item("content", "user", 10)
        assert item.staleness_score == 0.0

    def test_initial_reference_count_one(self, cm: ContextManager):
        item = cm.add_item("content", "user", 10)
        assert item.reference_count == 1

    def test_token_count_stored(self, cm: ContextManager):
        item = cm.add_item("content", "user", 42)
        assert item.token_count == 42


# ---------------------------------------------------------------------------
# Staleness scoring
# ---------------------------------------------------------------------------

class TestStaleness:
    """Tests for staleness score calculation (Req 2.7)."""

    def test_staleness_formula(self, cm: ContextManager):
        item = cm.add_item("content", "user", 10, item_id="a")
        # Turn 0: staleness = 0 * (1/1) = 0
        cm.update_staleness(0)
        assert item.staleness_score == 0.0

        # Advance to turn 3 without referencing
        cm.update_staleness(3)
        # turns_since = 3 - 0 = 3, ref_count = 1 → score = 3.0
        assert item.staleness_score == 3.0

    def test_reference_reduces_staleness(self, cm: ContextManager):
        item = cm.add_item("content", "user", 10, item_id="b")
        cm.update_staleness(5)
        assert item.staleness_score == 5.0

        # Reference the item at turn 5
        cm.current_turn = 5
        cm.reference_item("b")
        cm.update_staleness(7)
        # turns_since = 7 - 5 = 2, ref_count = 2 → score = 2 * 0.5 = 1.0
        assert item.staleness_score == 1.0

    def test_multiple_references_lower_staleness(self, cm: ContextManager):
        item = cm.add_item("content", "user", 10, item_id="c")
        cm.current_turn = 0
        cm.reference_item("c")  # ref_count → 2
        cm.reference_item("c")  # ref_count → 3
        cm.update_staleness(3)
        # turns_since = 3 - 0 = 3, ref_count = 3 → score = 3 * (1/3) = 1.0
        assert item.staleness_score == 1.0

    def test_advance_turn_updates_staleness(self, cm: ContextManager):
        item = cm.add_item("content", "user", 10)
        cm.advance_turn()
        assert cm.current_turn == 1
        assert item.staleness_score == 1.0

    def test_reference_nonexistent_item_no_error(self, cm: ContextManager):
        cm.reference_item("nonexistent")  # should not raise


# ---------------------------------------------------------------------------
# Compaction
# ---------------------------------------------------------------------------

class TestCompaction:
    """Tests for compaction logic (Req 2.4, 2.5)."""

    def test_should_compact_below_threshold(self, cm: ContextManager):
        cm.add_item("x", "user", 100)
        assert not cm.should_compact()

    def test_should_compact_at_threshold(self, cm: ContextManager):
        # 200K * 83.5% = 167,000
        cm.add_item("x", "user", 167_000)
        assert cm.should_compact()

    def test_compact_preserves_verbatim(self, cm: ContextManager):
        cm.add_item("Error: fail at line 5", "assistant", 100, item_id="v1")
        cm.add_item("Just chatting", "user", 200, item_id="c1")
        cm.add_item("Searching...", "tool_call", 50, item_id="e1")

        result = cm.compact()
        assert result.items_preserved >= 1
        # Verbatim item should still be in items
        ids = [i.id for i in cm.items]
        assert "v1" in ids

    def test_compact_drops_ephemeral(self, cm: ContextManager):
        cm.add_item("Searching...", "tool_call", 50, item_id="e1")
        result = cm.compact()
        assert result.items_dropped >= 1
        ids = [i.id for i in cm.items]
        assert "e1" not in ids

    def test_compact_compresses_discussion(self, cm: ContextManager):
        cm.add_item("Let me think about this approach.", "user", 1000, item_id="d1")
        result = cm.compact()
        assert result.items_compressed >= 1
        # Token count should be reduced
        item = next(i for i in cm.items if i.id == "d1")
        assert item.token_count < 1000

    def test_compact_returns_result(self, cm: ContextManager):
        cm.add_item("Error: x", "assistant", 100)
        cm.add_item("chat", "user", 200)
        cm.add_item("search", "tool_call", 50)
        result = cm.compact()
        assert isinstance(result, CompactionResult)
        assert result.tokens_freed > 0

    def test_compact_saves_to_session_file(self, cm: ContextManager):
        cm.add_item("Error: fail at line 5", "assistant", 100)
        result = cm.compact()
        assert Path(result.session_file_path).exists()

    def test_compact_preserves_structured(self, cm: ContextManager):
        cm.add_item("DECISION: Use Redis", "user", 80, item_id="s1")
        cm.compact()
        ids = [i.id for i in cm.items]
        assert "s1" in ids

    def test_compact_drops_stale_tool_results(self, cm: ContextManager):
        """Stale tool_result items with high staleness get dropped during compaction."""
        item = cm.add_item("Some old result", "tool_result", 100, item_id="stale1")
        # Make it compressible (not ephemeral — it has no pattern match but source is tool_result)
        # Actually tool_result with no pattern → EPHEMERAL, so let's use a structured one
        # that becomes stale. Let's manually set category to COMPRESS_AGGRESSIVE for this test.
        item.category = ContextCategory.COMPRESS_AGGRESSIVE
        item.staleness_score = 10.0  # very stale
        item.source = "tool_result"

        cm.compact()
        ids = [i.id for i in cm.items]
        assert "stale1" not in ids


# ---------------------------------------------------------------------------
# Threshold adaptation
# ---------------------------------------------------------------------------

class TestThresholdAdaptation:
    """Tests for adaptive compaction thresholds (Req 2.4, 2.5, 2.12)."""

    def test_200k_threshold(self, tmp_session_dir: str):
        cm = ContextManager("s", window_size=200_000, session_dir=tmp_session_dir)
        assert cm._effective_threshold() == 83.5

    def test_1m_threshold(self, tmp_session_dir: str):
        cm = ContextManager("s", window_size=1_000_000, session_dir=tmp_session_dir)
        assert cm._effective_threshold() == 85.0

    def test_env_override(self, tmp_session_dir: str, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("BRAINMASS_AUTOCOMPACT_PCT_OVERRIDE", "90")
        cm = ContextManager("s", window_size=200_000, session_dir=tmp_session_dir)
        assert cm._effective_threshold() == 90.0

    def test_env_override_invalid_ignored(self, tmp_session_dir: str, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("BRAINMASS_AUTOCOMPACT_PCT_OVERRIDE", "not_a_number")
        cm = ContextManager("s", window_size=200_000, session_dir=tmp_session_dir)
        assert cm._effective_threshold() == 83.5

    def test_env_override_out_of_range_ignored(self, tmp_session_dir: str, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("BRAINMASS_AUTOCOMPACT_PCT_OVERRIDE", "0")
        cm = ContextManager("s", window_size=200_000, session_dir=tmp_session_dir)
        assert cm._effective_threshold() == 83.5

    def test_env_override_boundary_100(self, tmp_session_dir: str, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("BRAINMASS_AUTOCOMPACT_PCT_OVERRIDE", "100")
        cm = ContextManager("s", window_size=200_000, session_dir=tmp_session_dir)
        assert cm._effective_threshold() == 100.0

    def test_env_override_boundary_1(self, tmp_session_dir: str, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("BRAINMASS_AUTOCOMPACT_PCT_OVERRIDE", "1")
        cm = ContextManager("s", window_size=200_000, session_dir=tmp_session_dir)
        assert cm._effective_threshold() == 1.0


# ---------------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------------

class TestSessionPersistence:
    """Tests for save_session / load_session (Req 2.3, 2.9)."""

    def test_save_creates_file(self, cm: ContextManager):
        cm.add_item("Error: x", "assistant", 100)
        path = cm.save_session()
        assert Path(path).exists()

    def test_save_only_preserved_items(self, cm: ContextManager):
        cm.add_item("Error: x", "assistant", 100, item_id="v1")
        cm.add_item("chat", "user", 50, item_id="c1")
        cm.add_item("search", "tool_call", 30, item_id="e1")
        path = cm.save_session()

        data = json.loads(Path(path).read_text())
        saved_ids = [i["id"] for i in data["items"]]
        assert "v1" in saved_ids
        assert "c1" not in saved_ids  # COMPRESS_AGGRESSIVE
        assert "e1" not in saved_ids  # EPHEMERAL

    def test_load_restores_items(self, cm: ContextManager, tmp_session_dir: str):
        cm.add_item("Error: x", "assistant", 100, item_id="v1")
        cm.save_session()

        # Create a new manager and load
        cm2 = ContextManager("test-session", session_dir=tmp_session_dir)
        loaded = cm2.load_session()
        assert loaded == 1
        assert cm2.items[0].id == "v1"
        assert cm2.items[0].category == ContextCategory.PRESERVE_VERBATIM

    def test_load_no_file_returns_zero(self, tmp_session_dir: str):
        cm = ContextManager("s", session_dir=tmp_session_dir)
        # Remove the session file if it exists
        sf = Path(tmp_session_dir) / "session-state.json"
        if sf.exists():
            sf.unlink()
        assert cm.load_session() == 0

    def test_load_avoids_duplicates(self, cm: ContextManager, tmp_session_dir: str):
        cm.add_item("Error: x", "assistant", 100, item_id="v1")
        cm.save_session()
        # Load into same manager — should not duplicate
        loaded = cm.load_session()
        assert loaded == 0
        assert len([i for i in cm.items if i.id == "v1"]) == 1

    def test_custom_path(self, cm: ContextManager, tmp_path: Path):
        cm.add_item("Error: x", "assistant", 100)
        custom = str(tmp_path / "custom-session.json")
        path = cm.save_session(custom)
        assert Path(path).exists()

        cm2 = ContextManager("s", session_dir=str(tmp_path))
        loaded = cm2.load_session(custom)
        assert loaded == 1

    def test_round_trip_preserves_content(self, cm: ContextManager, tmp_session_dir: str):
        original_content = "Error: connection refused at socket.connect"
        cm.add_item(original_content, "assistant", 100, item_id="rt1")
        cm.save_session()

        cm2 = ContextManager("test-session", session_dir=tmp_session_dir)
        cm2.load_session()
        assert cm2.items[0].content == original_content


# ---------------------------------------------------------------------------
# Health metrics
# ---------------------------------------------------------------------------

class TestHealthMetrics:
    """Tests for get_health_metrics() (Req 2.8)."""

    def test_empty_context(self, cm: ContextManager):
        metrics = cm.get_health_metrics()
        assert metrics.free_percent == 100.0
        assert metrics.total_tokens == 0
        assert metrics.preserved_tokens == 0
        assert metrics.compressible_tokens == 0
        assert metrics.ephemeral_tokens == 0

    def test_metrics_sum(self, cm: ContextManager):
        cm.add_item("Error: x", "assistant", 100)       # VERBATIM
        cm.add_item("DECISION: y", "user", 50)           # STRUCTURED
        cm.add_item("thinking...", "user", 200)           # COMPRESS
        cm.add_item("search result", "tool_call", 30)    # EPHEMERAL

        metrics = cm.get_health_metrics()
        assert metrics.total_tokens == 380
        assert metrics.preserved_tokens == 150  # 100 + 50
        assert metrics.compressible_tokens == 200
        assert metrics.ephemeral_tokens == 30

    def test_free_percent(self, cm: ContextManager):
        cm.add_item("x", "user", 100_000)
        metrics = cm.get_health_metrics()
        # (1 - 100000/200000) * 100 = 50.0
        assert metrics.free_percent == 50.0

    def test_staleness_distribution(self, cm: ContextManager):
        # Add items at different turns to create varied staleness
        cm.current_turn = 0
        cm.add_item("a", "user", 10, item_id="stale_item")

        cm.current_turn = 8
        cm.add_item("b", "user", 10, item_id="warm_item")

        cm.current_turn = 10
        cm.add_item("c", "user", 10, item_id="fresh_item")

        cm.update_staleness(10)
        # stale_item: (10-0)*(1/1) = 10.0 → stale
        # warm_item: (10-8)*(1/1) = 2.0 → warm
        # fresh_item: (10-10)*(1/1) = 0.0 → fresh

        metrics = cm.get_health_metrics()
        assert metrics.staleness_distribution["fresh"] >= 1
        assert metrics.staleness_distribution["warm"] >= 1
        assert metrics.staleness_distribution["stale"] >= 1

    def test_cache_hit_rate(self, cm: ContextManager):
        cm.record_cache_hit()
        cm.record_cache_hit()
        cm.record_cache_miss()
        metrics = cm.get_health_metrics()
        assert abs(metrics.cache_hit_rate - 2 / 3) < 0.01

    def test_cache_hit_rate_no_data(self, cm: ContextManager):
        metrics = cm.get_health_metrics()
        assert metrics.cache_hit_rate == 0.0


# ---------------------------------------------------------------------------
# Learning store
# ---------------------------------------------------------------------------

class TestLearningStore:
    """Tests for store_learning / query_learnings."""

    def test_store_and_query(self, cm: ContextManager):
        cm.store_learning("pattern1", "resolution1", 0.9)
        results = cm.query_learnings("anything")
        assert len(results) == 1
        assert results[0]["content"]["pattern"] == "pattern1"

    def test_store_multiple(self, cm: ContextManager):
        cm.store_learning("p1", "r1", 0.8)
        cm.store_learning("p2", "r2", 0.9)
        results = cm.query_learnings("q", top_k=10)
        assert len(results) == 2

    def test_store_returns_id(self, cm: ContextManager):
        mid = cm.store_learning("p", "r", 0.5)
        assert mid  # non-empty string


class TestAgentCoreMemoryStoreAdapter:
    def test_legacy_client_interface(self):
        class LegacyClient:
            def create_memory(self, namespace, content):
                return {"id": "legacy-id", "namespace": namespace, "content": content}

            def query(self, namespace, query, top_k=5):
                return [{"content": {"pattern": "p", "resolution": "r"}}]

        store = AgentCoreMemoryStore(client=LegacyClient())
        mid = store.create_memory("learning-store", {"pattern": "p"})
        rows = store.query("learning-store", "p", top_k=3)

        assert mid == "legacy-id"
        assert rows[0]["content"]["pattern"] == "p"

    def test_current_client_interface(self):
        class NewClient:
            def __init__(self):
                self.events = []

            def create_or_get_memory(self, name):
                return {"id": "mem-123", "name": name}

            def create_event(self, memory_id, actor_id, session_id, messages):
                self.events.append(
                    {
                        "memory_id": memory_id,
                        "actor_id": actor_id,
                        "session_id": session_id,
                        "messages": messages,
                    }
                )
                return {"event_id": "evt-123"}

            def retrieve_memories(self, memory_id, namespace, query, top_k=3):
                return [{"content": {"pattern": "p", "resolution": "r"}}]

        client = NewClient()
        store = AgentCoreMemoryStore(client=client)
        mid = store.create_memory("learning-store", {"pattern": "p"})
        rows = store.query("learning-store", "p", top_k=3)

        assert mid == "evt-123"
        assert client.events[0]["memory_id"] == "mem-123"
        assert rows[0]["content"]["resolution"] == "r"
