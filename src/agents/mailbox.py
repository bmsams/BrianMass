"""File-system-based mailbox IPC for Agent Teams.

Each teammate has an inbox directory under a configurable base directory.
Messages are JSON files with unique IDs, ordered by timestamp for FIFO
consumption. The team lead can broadcast messages to all teammates.

Requirements: 6.2, 6.5
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path

from src.types.core import MailboxMessage

logger = logging.getLogger(__name__)

# Valid message types per requirement 6.5
VALID_MESSAGE_TYPES = frozenset({
    "task_assignment",
    "finding",
    "question",
    "status_update",
})


class Mailbox:
    """File-system-based message queue for Agent Teams IPC.

    Messages are stored as JSON files in per-teammate inbox directories::

        {base_dir}/{teammate_id}/
            {timestamp}_{seq}_{uuid}.json

    FIFO ordering is achieved by sorting on the filename (timestamp + sequence prefix).

    Requirement 6.2: peer-to-peer communication via mailbox system.
    Requirement 6.5: inbox/mailbox system using file-system-based IPC.
    """

    def __init__(self, base_dir: str = ".brainmass/mailboxes") -> None:
        self._base_dir = Path(base_dir)
        self._seq: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_inbox(self, teammate_id: str) -> Path:
        """Create an inbox directory for *teammate_id*.

        Returns the inbox path. Idempotent — safe to call multiple times.
        """
        inbox = self._inbox_path(teammate_id)
        inbox.mkdir(parents=True, exist_ok=True)
        logger.debug("Inbox ready for '%s' at %s", teammate_id, inbox)
        return inbox

    def send_message(
        self,
        sender: str,
        recipient: str,
        msg_type: str,
        payload: dict,
    ) -> MailboxMessage:
        """Send a message to *recipient*'s inbox.

        Creates the recipient's inbox directory if it does not exist.
        Returns the ``MailboxMessage`` that was written.

        Raises ``ValueError`` if *msg_type* is not one of the valid types.
        """
        if msg_type not in VALID_MESSAGE_TYPES:
            raise ValueError(
                f"Invalid message type '{msg_type}'. "
                f"Must be one of: {', '.join(sorted(VALID_MESSAGE_TYPES))}"
            )

        now = datetime.now(UTC)
        message = MailboxMessage(
            sender=sender,
            recipient=recipient,
            type=msg_type,
            payload=payload,
            timestamp=now,
        )

        inbox = self._inbox_path(recipient)
        inbox.mkdir(parents=True, exist_ok=True)

        filename = self._message_filename(now, self._seq)
        self._seq += 1
        filepath = inbox / filename
        filepath.write_text(
            json.dumps(self._serialize_message(message), indent=2),
            encoding="utf-8",
        )
        logger.debug(
            "Message from '%s' to '%s' (type=%s) written to %s",
            sender, recipient, msg_type, filepath,
        )
        return message

    def broadcast(
        self,
        sender: str,
        teammates: list[str],
        msg_type: str,
        payload: dict,
    ) -> list[MailboxMessage]:
        """Broadcast a message from *sender* (team lead) to all *teammates*.

        Returns the list of messages sent (one per teammate).
        """
        messages: list[MailboxMessage] = []
        for teammate_id in teammates:
            msg = self.send_message(sender, teammate_id, msg_type, payload)
            messages.append(msg)
        logger.info(
            "Broadcast from '%s' to %d teammates (type=%s)",
            sender, len(teammates), msg_type,
        )
        return messages

    def read_messages(
        self,
        teammate_id: str,
        msg_type: str | None = None,
    ) -> list[MailboxMessage]:
        """Read all messages from *teammate_id*'s inbox in FIFO order.

        If *msg_type* is provided, only messages of that type are returned.
        Messages are **not** consumed (use ``consume_messages`` to read and
        delete).
        """
        inbox = self._inbox_path(teammate_id)
        if not inbox.exists():
            return []

        messages: list[MailboxMessage] = []
        for filepath in sorted(inbox.iterdir()):
            if not filepath.suffix == ".json":
                continue
            msg = self._read_message_file(filepath)
            if msg is None:
                continue
            if msg_type is not None and msg.type != msg_type:
                continue
            messages.append(msg)
        return messages

    def consume_messages(
        self,
        teammate_id: str,
        msg_type: str | None = None,
    ) -> list[MailboxMessage]:
        """Read and delete messages from *teammate_id*'s inbox (FIFO).

        If *msg_type* is provided, only messages of that type are consumed;
        other messages remain in the inbox.
        """
        inbox = self._inbox_path(teammate_id)
        if not inbox.exists():
            return []

        messages: list[MailboxMessage] = []
        for filepath in sorted(inbox.iterdir()):
            if not filepath.suffix == ".json":
                continue
            msg = self._read_message_file(filepath)
            if msg is None:
                # Remove corrupt files
                self._safe_delete(filepath)
                continue
            if msg_type is not None and msg.type != msg_type:
                continue
            messages.append(msg)
            self._safe_delete(filepath)
        return messages

    def count_messages(
        self,
        teammate_id: str,
        msg_type: str | None = None,
    ) -> int:
        """Return the number of messages in *teammate_id*'s inbox."""
        return len(self.read_messages(teammate_id, msg_type))

    def clear_inbox(self, teammate_id: str) -> int:
        """Delete all messages from *teammate_id*'s inbox.

        Returns the number of messages deleted.
        """
        inbox = self._inbox_path(teammate_id)
        if not inbox.exists():
            return 0

        deleted = 0
        for filepath in inbox.iterdir():
            if filepath.suffix == ".json":
                self._safe_delete(filepath)
                deleted += 1
        return deleted

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _inbox_path(self, teammate_id: str) -> Path:
        """Return the inbox directory path for a teammate."""
        return self._base_dir / teammate_id

    @staticmethod
    def _message_filename(timestamp: datetime, seq: int = 0) -> str:
        """Generate a unique, sortable filename for a message.

        Format: ``{iso_timestamp}_{seq:010d}_{uuid4}.json``
        The ISO timestamp + monotonic sequence ensures strict FIFO ordering
        even when multiple messages arrive within the same clock tick.
        """
        ts = timestamp.strftime("%Y%m%dT%H%M%S_%f")
        uid = uuid.uuid4().hex[:12]
        return f"{ts}_{seq:010d}_{uid}.json"

    @staticmethod
    def _serialize_message(msg: MailboxMessage) -> dict:
        """Convert a MailboxMessage to a JSON-serialisable dict."""
        return {
            "sender": msg.sender,
            "recipient": msg.recipient,
            "type": msg.type,
            "payload": msg.payload,
            "timestamp": msg.timestamp.isoformat(),
        }

    @staticmethod
    def _deserialize_message(data: dict) -> MailboxMessage:
        """Convert a dict back to a MailboxMessage."""
        return MailboxMessage(
            sender=data["sender"],
            recipient=data["recipient"],
            type=data["type"],
            payload=data["payload"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )

    def _read_message_file(self, filepath: Path) -> MailboxMessage | None:
        """Read and parse a message file. Returns ``None`` if corrupt."""
        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
            return self._deserialize_message(data)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            logger.warning("Corrupt message file '%s': %s", filepath, exc)
            return None

    @staticmethod
    def _safe_delete(filepath: Path) -> None:
        """Delete a file, ignoring FileNotFoundError."""
        try:
            filepath.unlink()
        except FileNotFoundError:
            pass
