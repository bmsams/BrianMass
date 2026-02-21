"""Unit tests for the mailbox IPC system (src/agents/mailbox.py).

Requirements: 6.2, 6.5
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from src.agents.mailbox import VALID_MESSAGE_TYPES, Mailbox
from src.types.core import MailboxMessage


@pytest.fixture()
def base_dir(tmp_path: Path) -> Path:
    """Return a temporary mailbox base directory."""
    d = tmp_path / "mailboxes"
    d.mkdir()
    return d


@pytest.fixture()
def mailbox(base_dir: Path) -> Mailbox:
    """Return a Mailbox using a temp directory."""
    return Mailbox(base_dir=str(base_dir))


# ---------------------------------------------------------------------------
# create_inbox
# ---------------------------------------------------------------------------


class TestCreateInbox:
    def test_creates_directory(self, mailbox: Mailbox, base_dir: Path):
        mailbox.create_inbox("agent-1")
        assert (base_dir / "agent-1").is_dir()

    def test_idempotent(self, mailbox: Mailbox, base_dir: Path):
        mailbox.create_inbox("agent-1")
        mailbox.create_inbox("agent-1")
        assert (base_dir / "agent-1").is_dir()

    def test_returns_inbox_path(self, mailbox: Mailbox, base_dir: Path):
        path = mailbox.create_inbox("agent-1")
        assert path == base_dir / "agent-1"

    def test_multiple_inboxes(self, mailbox: Mailbox, base_dir: Path):
        mailbox.create_inbox("agent-1")
        mailbox.create_inbox("agent-2")
        assert (base_dir / "agent-1").is_dir()
        assert (base_dir / "agent-2").is_dir()


# ---------------------------------------------------------------------------
# send_message
# ---------------------------------------------------------------------------


class TestSendMessage:
    def test_creates_json_file(self, mailbox: Mailbox, base_dir: Path):
        mailbox.send_message("lead", "agent-1", "task_assignment", {"task": "review"})
        inbox = base_dir / "agent-1"
        files = list(inbox.glob("*.json"))
        assert len(files) == 1

    def test_returns_mailbox_message(self, mailbox: Mailbox):
        msg = mailbox.send_message("lead", "agent-1", "finding", {"issue": "bug"})
        assert isinstance(msg, MailboxMessage)
        assert msg.sender == "lead"
        assert msg.recipient == "agent-1"
        assert msg.type == "finding"
        assert msg.payload == {"issue": "bug"}
        assert isinstance(msg.timestamp, datetime)

    def test_creates_inbox_if_missing(self, mailbox: Mailbox, base_dir: Path):
        mailbox.send_message("lead", "new-agent", "status_update", {})
        assert (base_dir / "new-agent").is_dir()

    def test_invalid_type_raises(self, mailbox: Mailbox):
        with pytest.raises(ValueError, match="Invalid message type"):
            mailbox.send_message("lead", "agent-1", "invalid_type", {})

    def test_all_valid_types(self, mailbox: Mailbox):
        for msg_type in VALID_MESSAGE_TYPES:
            msg = mailbox.send_message("lead", "agent-1", msg_type, {})
            assert msg.type == msg_type

    def test_multiple_messages_to_same_inbox(self, mailbox: Mailbox, base_dir: Path):
        mailbox.send_message("lead", "agent-1", "task_assignment", {"n": 1})
        mailbox.send_message("lead", "agent-1", "task_assignment", {"n": 2})
        files = list((base_dir / "agent-1").glob("*.json"))
        assert len(files) == 2


# ---------------------------------------------------------------------------
# broadcast
# ---------------------------------------------------------------------------


class TestBroadcast:
    def test_sends_to_all_teammates(self, mailbox: Mailbox):
        teammates = ["agent-1", "agent-2", "agent-3"]
        msgs = mailbox.broadcast("lead", teammates, "task_assignment", {"go": True})
        assert len(msgs) == 3
        for msg, tid in zip(msgs, teammates):
            assert msg.recipient == tid
            assert msg.sender == "lead"

    def test_each_teammate_receives_message(self, mailbox: Mailbox):
        teammates = ["agent-1", "agent-2"]
        mailbox.broadcast("lead", teammates, "status_update", {"status": "go"})
        for tid in teammates:
            msgs = mailbox.read_messages(tid)
            assert len(msgs) == 1
            assert msgs[0].payload == {"status": "go"}

    def test_broadcast_empty_list(self, mailbox: Mailbox):
        msgs = mailbox.broadcast("lead", [], "finding", {})
        assert msgs == []


# ---------------------------------------------------------------------------
# read_messages
# ---------------------------------------------------------------------------


class TestReadMessages:
    def test_fifo_order(self, mailbox: Mailbox):
        mailbox.send_message("a", "agent-1", "finding", {"n": 1})
        mailbox.send_message("b", "agent-1", "finding", {"n": 2})
        mailbox.send_message("c", "agent-1", "finding", {"n": 3})
        msgs = mailbox.read_messages("agent-1")
        assert [m.payload["n"] for m in msgs] == [1, 2, 3]

    def test_filter_by_type(self, mailbox: Mailbox):
        mailbox.send_message("a", "agent-1", "finding", {"n": 1})
        mailbox.send_message("b", "agent-1", "question", {"n": 2})
        mailbox.send_message("c", "agent-1", "finding", {"n": 3})
        msgs = mailbox.read_messages("agent-1", msg_type="finding")
        assert len(msgs) == 2
        assert all(m.type == "finding" for m in msgs)

    def test_empty_inbox(self, mailbox: Mailbox):
        mailbox.create_inbox("agent-1")
        assert mailbox.read_messages("agent-1") == []

    def test_nonexistent_inbox(self, mailbox: Mailbox):
        assert mailbox.read_messages("ghost") == []

    def test_read_does_not_consume(self, mailbox: Mailbox):
        mailbox.send_message("a", "agent-1", "finding", {})
        mailbox.read_messages("agent-1")
        assert len(mailbox.read_messages("agent-1")) == 1


# ---------------------------------------------------------------------------
# consume_messages
# ---------------------------------------------------------------------------


class TestConsumeMessages:
    def test_consume_removes_messages(self, mailbox: Mailbox):
        mailbox.send_message("a", "agent-1", "finding", {})
        msgs = mailbox.consume_messages("agent-1")
        assert len(msgs) == 1
        assert mailbox.read_messages("agent-1") == []

    def test_consume_fifo_order(self, mailbox: Mailbox):
        mailbox.send_message("a", "agent-1", "finding", {"n": 1})
        mailbox.send_message("b", "agent-1", "finding", {"n": 2})
        msgs = mailbox.consume_messages("agent-1")
        assert [m.payload["n"] for m in msgs] == [1, 2]

    def test_consume_with_type_filter(self, mailbox: Mailbox):
        mailbox.send_message("a", "agent-1", "finding", {"n": 1})
        mailbox.send_message("b", "agent-1", "question", {"n": 2})
        mailbox.send_message("c", "agent-1", "finding", {"n": 3})
        consumed = mailbox.consume_messages("agent-1", msg_type="finding")
        assert len(consumed) == 2
        # The question should still be in the inbox
        remaining = mailbox.read_messages("agent-1")
        assert len(remaining) == 1
        assert remaining[0].type == "question"

    def test_consume_empty_inbox(self, mailbox: Mailbox):
        mailbox.create_inbox("agent-1")
        assert mailbox.consume_messages("agent-1") == []

    def test_consume_nonexistent_inbox(self, mailbox: Mailbox):
        assert mailbox.consume_messages("ghost") == []


# ---------------------------------------------------------------------------
# count_messages
# ---------------------------------------------------------------------------


class TestCountMessages:
    def test_count(self, mailbox: Mailbox):
        mailbox.send_message("a", "agent-1", "finding", {})
        mailbox.send_message("b", "agent-1", "question", {})
        assert mailbox.count_messages("agent-1") == 2

    def test_count_with_filter(self, mailbox: Mailbox):
        mailbox.send_message("a", "agent-1", "finding", {})
        mailbox.send_message("b", "agent-1", "question", {})
        assert mailbox.count_messages("agent-1", msg_type="finding") == 1

    def test_count_empty(self, mailbox: Mailbox):
        assert mailbox.count_messages("agent-1") == 0


# ---------------------------------------------------------------------------
# clear_inbox
# ---------------------------------------------------------------------------


class TestClearInbox:
    def test_clears_all_messages(self, mailbox: Mailbox):
        mailbox.send_message("a", "agent-1", "finding", {})
        mailbox.send_message("b", "agent-1", "question", {})
        deleted = mailbox.clear_inbox("agent-1")
        assert deleted == 2
        assert mailbox.read_messages("agent-1") == []

    def test_clear_empty_inbox(self, mailbox: Mailbox):
        mailbox.create_inbox("agent-1")
        assert mailbox.clear_inbox("agent-1") == 0

    def test_clear_nonexistent_inbox(self, mailbox: Mailbox):
        assert mailbox.clear_inbox("ghost") == 0


# ---------------------------------------------------------------------------
# Corrupt message handling
# ---------------------------------------------------------------------------


class TestCorruptMessages:
    def test_corrupt_json_skipped_on_read(self, mailbox: Mailbox, base_dir: Path):
        mailbox.create_inbox("agent-1")
        inbox = base_dir / "agent-1"
        (inbox / "00000000T000000_000000_bad.json").write_text("not json", encoding="utf-8")
        mailbox.send_message("a", "agent-1", "finding", {"ok": True})
        msgs = mailbox.read_messages("agent-1")
        assert len(msgs) == 1
        assert msgs[0].payload == {"ok": True}

    def test_corrupt_json_removed_on_consume(self, mailbox: Mailbox, base_dir: Path):
        mailbox.create_inbox("agent-1")
        inbox = base_dir / "agent-1"
        corrupt_file = inbox / "00000000T000000_000000_bad.json"
        corrupt_file.write_text("not json", encoding="utf-8")
        mailbox.consume_messages("agent-1")
        assert not corrupt_file.exists()

    def test_missing_fields_skipped(self, mailbox: Mailbox, base_dir: Path):
        mailbox.create_inbox("agent-1")
        inbox = base_dir / "agent-1"
        import json
        (inbox / "00000000T000000_000000_partial.json").write_text(
            json.dumps({"sender": "x"}), encoding="utf-8"
        )
        assert mailbox.read_messages("agent-1") == []


# ---------------------------------------------------------------------------
# Message serialization round-trip
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_round_trip(self, mailbox: Mailbox):
        mailbox.send_message("lead", "agent-1", "task_assignment", {"key": "value", "num": 42})
        msgs = mailbox.read_messages("agent-1")
        assert len(msgs) == 1
        msg = msgs[0]
        assert msg.sender == "lead"
        assert msg.recipient == "agent-1"
        assert msg.type == "task_assignment"
        assert msg.payload == {"key": "value", "num": 42}
        assert msg.timestamp.tzinfo is not None  # timezone-aware

    def test_nested_payload(self, mailbox: Mailbox):
        payload = {"files": ["a.py", "b.py"], "meta": {"priority": "high"}}
        mailbox.send_message("lead", "agent-1", "task_assignment", payload)
        msgs = mailbox.read_messages("agent-1")
        assert msgs[0].payload == payload
