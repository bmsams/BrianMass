"""Unit tests for runtime config loader."""

from __future__ import annotations

import pytest

from src.runtime.config import RuntimeConfig, load_runtime_config


def test_load_runtime_config_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AWS_REGION", raising=False)
    cfg = load_runtime_config(strict=False)
    assert isinstance(cfg, RuntimeConfig)
    assert cfg.aws_region
    assert cfg.model_id_sonnet


def test_load_runtime_config_strict_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "AWS_REGION",
        "AGENTCORE_APP_NAME",
        "BRAINMASS_SESSION_BUCKET",
        "MODEL_ID_HAIKU",
        "MODEL_ID_SONNET",
        "MODEL_ID_OPUS",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="Missing required runtime env vars"):
        _ = load_runtime_config(strict=True)


def test_load_runtime_config_strict_local_dev_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    monkeypatch.setenv("AGENTCORE_APP_NAME", "brainmass-v3")
    monkeypatch.setenv("BRAINMASS_SESSION_BUCKET", "brainmass-sessions")
    monkeypatch.setenv("MODEL_ID_HAIKU", "h")
    monkeypatch.setenv("MODEL_ID_SONNET", "s")
    monkeypatch.setenv("MODEL_ID_OPUS", "o")
    monkeypatch.setenv("BRAINMASS_LOCAL_DEV", "1")
    with pytest.raises(RuntimeError, match="not allowed in strict runtime mode"):
        _ = load_runtime_config(strict=True)

