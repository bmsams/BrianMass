"""Session Teleporter - cross-surface session serialization and continuity.

Serializes all session state into a portable blob and transfers it across
surfaces (web, mobile, desktop, CLI) with full state preservation.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from src.types.core import SessionState

logger = logging.getLogger(__name__)

VALID_SURFACES = frozenset({"web", "mobile", "desktop", "cli"})


class SessionStorageAdapter(Protocol):
    """Storage adapter for serialized session blobs."""

    def put(self, key: str, data: bytes) -> None:
        ...

    def get(self, key: str) -> bytes:
        ...


class SessionMemoryAdapter(Protocol):
    """Memory adapter for session continuity metadata."""

    def write_session_metadata(self, session_key: str, metadata: dict) -> None:
        ...

    def read_session_metadata(self, session_key: str) -> dict:
        ...


class InMemorySessionStorageAdapter:
    """In-memory storage adapter used by tests and local-dev paths."""

    def __init__(self) -> None:
        self._store: dict[str, bytes] = {}

    def put(self, key: str, data: bytes) -> None:
        self._store[key] = data

    def get(self, key: str) -> bytes:
        if key not in self._store:
            raise KeyError(f"Session key not found: {key}")
        return self._store[key]


class InMemorySessionMemoryAdapter:
    """In-memory metadata adapter used by tests and local-dev paths."""

    def __init__(self) -> None:
        self._store: dict[str, dict] = {}

    def write_session_metadata(self, session_key: str, metadata: dict) -> None:
        self._store[session_key] = dict(metadata)

    def read_session_metadata(self, session_key: str) -> dict:
        return dict(self._store.get(session_key, {}))


class S3SessionStorageAdapter:
    """S3-backed blob storage adapter for production session persistence."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "sessions/",
        s3_client: object | None = None,
    ) -> None:
        self._bucket = bucket
        self._prefix = prefix
        if s3_client is not None:
            self._client = s3_client
        else:
            try:
                import boto3  # type: ignore
            except Exception as exc:  # pragma: no cover - exercised in runtime env
                raise RuntimeError("boto3 is required for S3 session storage adapter.") from exc
            self._client = boto3.client("s3")

    def _obj_key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    def put(self, key: str, data: bytes) -> None:
        self._client.put_object(Bucket=self._bucket, Key=self._obj_key(key), Body=data)

    def get(self, key: str) -> bytes:
        try:
            resp = self._client.get_object(Bucket=self._bucket, Key=self._obj_key(key))
        except Exception as exc:
            raise KeyError(f"Session key not found: {key}") from exc
        body = resp["Body"].read()
        return bytes(body)


class AgentCoreSessionMemoryAdapter:
    """AgentCore Memory-backed metadata adapter."""

    def __init__(self, namespace: str = "session-teleporter", client: object | None = None) -> None:
        self._namespace = namespace
        if client is None:
            try:
                from bedrock_agentcore.memory import MemoryClient  # type: ignore
            except Exception as exc:  # pragma: no cover - exercised in runtime env
                raise RuntimeError("bedrock-agentcore MemoryClient is required.") from exc
            self._client = MemoryClient()
        else:
            self._client = client
        self._memory_id: str | None = None

    @staticmethod
    def _extract_id(result: object) -> str:
        if isinstance(result, dict):
            for key in ("id", "memory_id", "memoryId", "name"):
                if key in result:
                    return str(result[key])
        return str(result)

    def _ensure_memory_id(self) -> str:
        if self._memory_id:
            return self._memory_id

        if hasattr(self._client, "create_or_get_memory"):
            raw = self._client.create_or_get_memory(name=self._namespace)
        elif hasattr(self._client, "create_memory"):
            raw = self._client.create_memory(name=self._namespace)
        else:
            raise RuntimeError("AgentCore Memory client does not support memory creation APIs.")

        self._memory_id = self._extract_id(raw)
        return self._memory_id

    def write_session_metadata(self, session_key: str, metadata: dict) -> None:
        # Compatibility path for older/adapter clients used in tests.
        if hasattr(self._client, "create_memory"):
            try:
                self._client.create_memory(
                    namespace=self._namespace,
                    content={"session_key": session_key, "metadata": metadata},
                )
                return
            except TypeError:
                pass

        if not hasattr(self._client, "create_event"):
            raise RuntimeError("AgentCore Memory client does not support event creation APIs.")

        payload = json.dumps({"session_key": session_key, "metadata": metadata}, default=str)
        self._client.create_event(
            memory_id=self._ensure_memory_id(),
            actor_id="brainmass",
            session_id=session_key,
            messages=[("user", payload)],
        )

    def read_session_metadata(self, session_key: str) -> dict:
        # Compatibility path for older/adapter clients used in tests.
        if hasattr(self._client, "query"):
            try:
                rows = self._client.query(
                    namespace=self._namespace,
                    query=session_key,
                    top_k=1,
                )
                if not rows:
                    return {}
                content = rows[0].get("content", {})
                return dict(content.get("metadata", {}))
            except TypeError:
                pass

        if not hasattr(self._client, "retrieve_memories"):
            raise RuntimeError("AgentCore Memory client does not support retrieval APIs.")

        rows = self._client.retrieve_memories(
            memory_id=self._ensure_memory_id(),
            namespace=self._namespace,
            query=session_key,
            top_k=3,
        )
        if not isinstance(rows, list):
            return {}

        for row in rows:
            metadata = self._extract_metadata(row, session_key)
            if metadata is not None:
                return metadata
        return {}

    @staticmethod
    def _extract_metadata(row: Any, session_key: str) -> dict[str, Any] | None:
        if not isinstance(row, dict):
            return None

        content = row.get("content")
        if isinstance(content, dict):
            if (
                content.get("session_key") == session_key
                and isinstance(content.get("metadata"), dict)
            ):
                return dict(content["metadata"])

        for key in ("text", "value", "memory"):
            raw = row.get(key)
            if not isinstance(raw, str):
                continue
            try:
                parsed = json.loads(raw)
            except ValueError:
                continue
            if (
                isinstance(parsed, dict)
                and parsed.get("session_key") == session_key
                and isinstance(parsed.get("metadata"), dict)
            ):
                return dict(parsed["metadata"])

        return None


@dataclass
class ConflictRecord:
    """Records a last-write-wins conflict notification."""

    surface: str
    timestamp: datetime
    field: str

    def to_dict(self) -> dict:
        return {
            "surface": self.surface,
            "timestamp": self.timestamp.isoformat(),
            "field": self.field,
        }


class SessionTeleporter:
    """Serializes and restores full session state across surfaces.

    Backwards-compatible constructor:
    - storage_backend/load_backend callbacks are still accepted.
    New adapter-based constructor:
    - storage_adapter and memory_adapter can be supplied.

    When neither adapter is supplied, auto-selection applies
    (Requirements: 9.1, 9.2, 9.3, 9.4):
    - storage_adapter: uses S3SessionStorageAdapter when the
      ``S3_SESSION_BUCKET`` environment variable is set; falls back to
      InMemorySessionStorageAdapter with a warning.
    - memory_adapter: uses AgentCoreSessionMemoryAdapter when
      ``bedrock_agentcore`` is importable; falls back to
      InMemorySessionMemoryAdapter.
    """

    def __init__(
        self,
        storage_backend: Any | None = None,
        load_backend: Any | None = None,
        storage_adapter: SessionStorageAdapter | None = None,
        memory_adapter: SessionMemoryAdapter | None = None,
    ) -> None:

        # --- Production integration point ---
        # Storage adapter auto-selection: prefer S3 when bucket env var is set.
        # import os
        # bucket = os.environ.get("S3_SESSION_BUCKET")
        # if bucket:
        #     try:
        #         storage_adapter = S3SessionStorageAdapter(bucket=bucket)
        #     except Exception:
        #         logger.warning("S3SessionStorageAdapter init failed; falling back to in-memory")
        #         storage_adapter = InMemorySessionStorageAdapter()
        # else:
        #     storage_adapter = InMemorySessionStorageAdapter()
        if storage_adapter is None:
            bucket = os.environ.get("S3_SESSION_BUCKET")
            if bucket:
                try:
                    storage_adapter = S3SessionStorageAdapter(bucket=bucket)
                    logger.debug("SessionTeleporter: using S3 storage adapter (bucket=%s)", bucket)
                except Exception as exc:
                    logger.warning(
                        "S3SessionStorageAdapter init failed (%s); falling back to in-memory.",
                        exc,
                    )
                    storage_adapter = InMemorySessionStorageAdapter()
            else:
                storage_adapter = InMemorySessionStorageAdapter()

        # --- Production integration point ---
        # Memory adapter auto-selection: prefer AgentCore when SDK is available
        # AND the BRAINMASS_AGENTCORE_MEMORY env var is set.
        # This matches the pattern used in context_manager.py so that tests and
        # local development work without AWS credentials.
        # try:
        #     from bedrock_agentcore.memory import MemoryClient
        #     memory_adapter = AgentCoreSessionMemoryAdapter()
        # except ImportError:
        #     memory_adapter = InMemorySessionMemoryAdapter()
        if memory_adapter is None:
            if os.environ.get("BRAINMASS_AGENTCORE_MEMORY") == "1":
                try:
                    from bedrock_agentcore.memory import MemoryClient  # type: ignore  # noqa: F401

                    memory_adapter = AgentCoreSessionMemoryAdapter()
                    logger.debug("SessionTeleporter: using AgentCore memory adapter.")
                except ImportError:
                    memory_adapter = InMemorySessionMemoryAdapter()
            else:
                memory_adapter = InMemorySessionMemoryAdapter()

        self._storage_adapter = storage_adapter
        self._memory_adapter = memory_adapter

        if storage_backend is None:
            self._storage_backend = self._storage_adapter.put
        else:
            self._storage_backend = storage_backend

        if load_backend is None:
            self._load_backend = self._storage_adapter.get
        else:
            self._load_backend = load_backend

        self._conflicts: list[ConflictRecord] = []
        self._resume_timestamps: dict[str, dict[str, datetime]] = {}

    def serialize(self, state: SessionState) -> bytes:
        """Serialize all session state into a portable JSON blob."""
        payload = {
            "schema_version": "1.0",
            "serialized_at": datetime.now(UTC).isoformat(),
            "state": {
                "conversation_history": state.conversation_history,
                "tool_permissions": state.tool_permissions,
                "active_workers": state.active_workers,
                "pending_approvals": state.pending_approvals,
                "compaction_state": state.compaction_state,
                "context_manager_state": state.context_manager_state,
                "cost_tracking": state.cost_tracking,
                "hook_registrations": state.hook_registrations,
                "trace_id": state.trace_id,
            },
        }
        return json.dumps(payload, default=str).encode("utf-8")

    def deserialize(self, blob: bytes) -> SessionState:
        """Deserialize a portable blob back into a SessionState.

        Supports:
        - schema v1 payload {"schema_version", "state": {...}}
        - legacy payload that is already the state dict
        """
        payload = json.loads(blob.decode("utf-8"))
        raw = payload.get("state", payload)
        return SessionState(
            conversation_history=raw["conversation_history"],
            tool_permissions=raw["tool_permissions"],
            active_workers=raw["active_workers"],
            pending_approvals=raw["pending_approvals"],
            compaction_state=raw["compaction_state"],
            context_manager_state=raw["context_manager_state"],
            cost_tracking=raw["cost_tracking"],
            hook_registrations=raw["hook_registrations"],
            trace_id=raw["trace_id"],
        )

    def teleport(self, state: SessionState, from_surface: str, to_surface: str = "cli") -> str:
        """Transfer session from one surface to another and return session key."""
        self._validate_surface(from_surface)
        self._validate_surface(to_surface)

        session_key = f"session:{state.trace_id}"
        blob = self.serialize(state)
        self._storage_backend(session_key, blob)

        try:
            self._memory_adapter.write_session_metadata(
                session_key=session_key,
                metadata={
                    "from_surface": from_surface,
                    "to_surface": to_surface,
                    "teleported_at": datetime.now(UTC).isoformat(),
                    "trace_id": state.trace_id,
                },
            )
        except Exception:
            logger.debug("Session memory adapter write failed", exc_info=True)

        logger.info(
            "Session teleported",
            extra={
                "trace_id": state.trace_id,
                "from_surface": from_surface,
                "to_surface": to_surface,
                "active_workers": len(state.active_workers),
                "session_key": session_key,
            },
        )
        return session_key

    def desktop_handoff(self, state: SessionState, from_surface: str = "cli") -> str:
        """Hand off session to desktop app."""
        return self.teleport(state, from_surface=from_surface, to_surface="desktop")

    def resume(self, session_key: str, surface: str) -> SessionState:
        """Load and resume a session on the given surface."""
        self._validate_surface(surface)
        blob = self._load_backend(session_key)
        state = self.deserialize(blob)

        conflict = self._detect_conflict(session_key, surface)
        if conflict:
            self._conflicts.append(conflict)
            logger.warning(
                "Session conflict detected (last-write-wins applied)",
                extra={"session_key": session_key, "conflict": conflict.to_dict()},
            )

        self._record_resume(session_key, surface)
        return state

    def merge_permissions(self, base: SessionState, incoming: SessionState) -> dict[str, bool]:
        """Merge tool permissions preserving approvals and explicit denials."""
        merged: dict[str, bool] = dict(base.tool_permissions)
        for tool, allowed in incoming.tool_permissions.items():
            if tool in merged:
                merged[tool] = merged[tool] and allowed
            else:
                merged[tool] = allowed
        return merged

    def transfer_workers(self, state: SessionState, target_surface: str) -> SessionState:
        """Ensure active workers survive teleport to target surface."""
        logger.info(
            "Transferring workers to surface",
            extra={
                "target_surface": target_surface,
                "worker_count": len(state.active_workers),
            },
        )
        return state

    def ensure_trace_id(self, state: SessionState) -> SessionState:
        """Ensure the session has a stable trace ID for audit continuity."""
        if not state.trace_id:
            object.__setattr__(state, "trace_id", str(uuid.uuid4()))
        return state

    def merge_cost_tracking(self, base: SessionState, incoming: SessionState) -> dict:
        """Merge cost tracking from two states for unified attribution."""
        merged = dict(base.cost_tracking)
        for key, value in incoming.cost_tracking.items():
            if key in merged and isinstance(value, (int, float)):
                merged[key] = merged.get(key, 0) + value
            else:
                merged[key] = value
        return merged

    def get_conflicts(self) -> list[ConflictRecord]:
        """Return all recorded conflict notifications."""
        return list(self._conflicts)

    def clear_conflicts(self) -> None:
        """Clear conflict log."""
        self._conflicts.clear()

    def _validate_surface(self, surface: str) -> None:
        if surface not in VALID_SURFACES:
            raise ValueError(f"Unknown surface '{surface}'. Valid: {sorted(VALID_SURFACES)}")

    def _record_resume(self, session_key: str, surface: str) -> None:
        if session_key not in self._resume_timestamps:
            self._resume_timestamps[session_key] = {}
        self._resume_timestamps[session_key][surface] = datetime.now(UTC)

    def _detect_conflict(self, session_key: str, surface: str) -> ConflictRecord | None:
        resumes = self._resume_timestamps.get(session_key, {})
        for other_surface, ts in resumes.items():
            if other_surface == surface:
                continue
            return ConflictRecord(
                surface=other_surface,
                timestamp=ts,
                field="session_state",
            )
        return None
