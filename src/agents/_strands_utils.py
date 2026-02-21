"""Shared Strands SDK utilities for all agent modules.

Provides the canonical Bedrock model ID mapping and a normalisation helper
that converts any Strands Agent response shape into a consistent result dict.
All agent modules import from here to avoid duplication.

Requirements: 1.4, 2.3, 13.6
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Canonical Bedrock cross-region inference profile IDs
# ---------------------------------------------------------------------------

# --- Production integration point ---
# All model aliases and canonical IDs map to us.* cross-region inference
# profile IDs, consistent with orchestrator.py _MODEL_IDS.
_BEDROCK_MODEL_IDS: dict[str, str] = {
    # Short aliases used in AgentDefinition.model
    "sonnet": "us.anthropic.claude-sonnet-4-5-v1:0",
    "opus": "us.anthropic.claude-opus-4-6-v1:0",
    "haiku": "us.anthropic.claude-haiku-4-5-v1:0",
    "inherit": "us.anthropic.claude-sonnet-4-5-v1:0",
    # Canonical IDs from AgentLoader.MODEL_MAP (also accepted as-is)
    "claude-sonnet-4-5-20250929": "us.anthropic.claude-sonnet-4-5-v1:0",
    "claude-opus-4-6": "us.anthropic.claude-opus-4-6-v1:0",
    "claude-haiku-4-5-20251001": "us.anthropic.claude-haiku-4-5-v1:0",
    # Cross-region profile IDs (pass-through — already in correct form)
    "us.anthropic.claude-sonnet-4-5-v1:0": "us.anthropic.claude-sonnet-4-5-v1:0",
    "us.anthropic.claude-opus-4-6-v1:0": "us.anthropic.claude-opus-4-6-v1:0",
    "us.anthropic.claude-haiku-4-5-v1:0": "us.anthropic.claude-haiku-4-5-v1:0",
}


# ---------------------------------------------------------------------------
# Result normalisation
# ---------------------------------------------------------------------------

def _normalize_strands_result(raw: Any) -> dict:  # noqa: ANN401
    """Extract a standard result dict from any Strands Agent response shape.

    Handles all known response shapes:
    - ``str`` — plain text response
    - ``dict`` — mapping with ``output``/``text``/``usage`` keys
    - object with ``.output`` / ``.text`` / ``.usage`` / ``.content`` attrs
    - ``None`` — treated as empty string

    Returns a dict with exactly these six keys:

    .. code-block:: python

        {
            "summary": str,
            "tokens_consumed": {"input": int, "output": int, "cache_read": int},
            "tools_used": list[str],
            "files_modified": list[str],
            "exit_reason": str,
            "turns_used": int,
        }

    Requirements: 1.4, 13.6
    """
    # ------------------------------------------------------------------
    # Text extraction
    # ------------------------------------------------------------------
    if raw is None:
        response_text = ""
    elif isinstance(raw, str):
        response_text = raw
    elif isinstance(raw, dict):
        response_text = str(raw.get("output") or raw.get("text") or "")
    else:
        response_text = str(
            getattr(raw, "output", None)
            or getattr(raw, "text", None)
            or raw
        )

    # ------------------------------------------------------------------
    # Usage extraction
    # ------------------------------------------------------------------
    tokens: dict[str, int] = {"input": 0, "output": 0, "cache_read": 0}

    if isinstance(raw, dict):
        tokens = {
            "input": int(raw.get("input_tokens", 0) or 0),
            "output": int(raw.get("output_tokens", 0) or 0),
            "cache_read": int(raw.get("cache_read_input_tokens", 0) or 0),
        }
        usage_dict = raw.get("usage")
        if isinstance(usage_dict, dict):
            tokens = {
                "input": int(usage_dict.get("input_tokens", 0) or 0),
                "output": int(usage_dict.get("output_tokens", 0) or 0),
                "cache_read": int(
                    usage_dict.get("cache_read_input_tokens", 0)
                    or usage_dict.get("cache_read_tokens", 0)
                    or 0
                ),
            }
    else:
        usage = getattr(raw, "usage", None)
        if usage is not None:
            tokens = {
                "input": int(getattr(usage, "input_tokens", 0) or 0),
                "output": int(getattr(usage, "output_tokens", 0) or 0),
                "cache_read": int(
                    getattr(usage, "cache_read_input_tokens", 0)
                    or getattr(usage, "cache_read_tokens", 0)
                    or 0
                ),
            }

    # ------------------------------------------------------------------
    # Tool use and file extraction from content blocks
    # ------------------------------------------------------------------
    tools_used: list[str] = []
    files_modified: list[str] = []

    content_blocks: Any = None
    if isinstance(raw, dict):
        content_blocks = raw.get("content")
    else:
        content_blocks = getattr(raw, "content", None)

    if isinstance(content_blocks, list):
        for block in content_blocks:
            if isinstance(block, dict):
                if block.get("type") == "tool_use":
                    tool_name = str(block.get("name", "unknown"))
                    tools_used.append(tool_name)
                    tool_input = block.get("input") or {}
                    if isinstance(tool_input, dict) and "file_path" in tool_input:
                        files_modified.append(str(tool_input["file_path"]))
            else:
                if getattr(block, "type", None) == "tool_use":
                    tool_name = str(getattr(block, "name", "unknown"))
                    tools_used.append(tool_name)
                    tool_input = getattr(block, "input", {}) or {}
                    if isinstance(tool_input, dict) and "file_path" in tool_input:
                        files_modified.append(str(tool_input["file_path"]))

    # ------------------------------------------------------------------
    # Exit reason
    # ------------------------------------------------------------------
    if isinstance(raw, dict):
        stop_reason = raw.get("stop_reason", "end_turn") or "end_turn"
    else:
        stop_reason = getattr(raw, "stop_reason", "end_turn") or "end_turn"
    exit_reason = "complete" if stop_reason == "end_turn" else str(stop_reason)

    # ------------------------------------------------------------------
    # Turns used
    # ------------------------------------------------------------------
    if isinstance(raw, dict):
        turns_used = int(raw.get("turns", 0) or 0)
    else:
        turns_used = int(getattr(raw, "turns", 0) or 0)

    return {
        "summary": response_text,
        "tokens_consumed": tokens,
        "tools_used": tools_used,
        "files_modified": files_modified,
        "exit_reason": exit_reason,
        "turns_used": turns_used,
    }
