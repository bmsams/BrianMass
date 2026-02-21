"""Runtime configuration for AgentCore deployment.

Provides strict environment validation for production runtime mode and
canonical model ID resolution.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from src.types.core import ModelTier


@dataclass(frozen=True)
class RuntimeConfig:
    """Validated runtime configuration."""

    aws_region: str
    agentcore_app_name: str
    session_bucket: str
    model_id_haiku: str
    model_id_sonnet: str
    model_id_opus: str
    enable_tool_search: str
    autocompact_pct_override: str
    local_dev: bool

    def model_id_for_tier(self, tier: ModelTier) -> str:
        """Return canonical model ID for a model tier."""
        if tier == ModelTier.HAIKU:
            return self.model_id_haiku
        if tier == ModelTier.OPUS:
            return self.model_id_opus
        return self.model_id_sonnet


_REQUIRED_PROD_ENV = (
    "AWS_REGION",
    "AGENTCORE_APP_NAME",
    "BRAINMASS_SESSION_BUCKET",
    "MODEL_ID_HAIKU",
    "MODEL_ID_SONNET",
    "MODEL_ID_OPUS",
)


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def load_runtime_config(strict: bool = False) -> RuntimeConfig:
    """Load runtime config from environment.

    Args:
        strict: When True, require production-critical variables.
    """
    local_dev = _env("BRAINMASS_LOCAL_DEV", "0") == "1"
    if strict:
        missing = [name for name in _REQUIRED_PROD_ENV if not _env(name)]
        if missing:
            raise RuntimeError(
                "Missing required runtime env vars: " + ", ".join(missing)
            )
        if local_dev:
            raise RuntimeError(
                "BRAINMASS_LOCAL_DEV=1 is not allowed in strict runtime mode."
            )

    return RuntimeConfig(
        aws_region=_env("AWS_REGION", "us-east-1"),
        agentcore_app_name=_env("AGENTCORE_APP_NAME", "brainmass"),
        session_bucket=_env("BRAINMASS_SESSION_BUCKET", "brainmass-sessions"),
        model_id_haiku=_env("MODEL_ID_HAIKU", "us.anthropic.claude-haiku-4-5-v1:0"),
        model_id_sonnet=_env("MODEL_ID_SONNET", "us.anthropic.claude-sonnet-4-5-v1:0"),
        model_id_opus=_env("MODEL_ID_OPUS", "us.anthropic.claude-opus-4-6-v1:0"),
        enable_tool_search=_env("ENABLE_TOOL_SEARCH", "auto"),
        autocompact_pct_override=_env("BRAINMASS_AUTOCOMPACT_PCT_OVERRIDE", ""),
        local_dev=local_dev,
    )

