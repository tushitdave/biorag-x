"""Configuration and experiment-recipe loading for BioRAG-X.

Two layers:
  * Settings  -> environment-driven runtime knobs (LLM budget, model names).
  * Recipe    -> a YAML experiment configuration (chunking/retrieval/generation).

Recipes mirror the research doc's "Experiment Recipe" so the same config drives
backend runs and the frontend's Experiment DNA view.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from .paths import ENV_PATH, CONFIGS_DIR

load_dotenv(ENV_PATH if ENV_PATH.exists() else None)


def _get_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    """Runtime settings. LLM usage is capped hard for the POC."""
    # LLM master switch + strict budget (POC: <= 12 calls total per process).
    llm_enabled: bool = field(default_factory=lambda: _get_bool("LLM_ENABLED", False))
    llm_call_budget: int = field(default_factory=lambda: int(os.getenv("LLM_CALL_BUDGET", "12")))

    # Azure chat (generation) — confirmed working deployment.
    chat_deployment: str = field(default_factory=lambda: os.getenv("AZURE_DEPLOYMENT_GPT_4o", "gpt-4o"))
    chat_endpoint: str | None = field(default_factory=lambda: (
        os.getenv("AZURE_GPT_4o_ENDPOINT") or os.getenv("AZURE_AI_ENDPOINT")
    ))
    chat_api_key: str | None = field(default_factory=lambda: (
        os.getenv("AZURE_GPT_4o_API_KEY") or os.getenv("AZURE_AI_KEY")
    ))
    chat_api_version: str = field(default_factory=lambda: (
        os.getenv("AZURE_GPT_4o_API_VERSION")
        or os.getenv("OPENAI_API_VERSION")
        or os.getenv("AZURE_API_VERSION")
        or "2024-12-01-preview"
    ))

    # Azure embeddings (query embedding on demand) — ada-002.
    embed_deployment: str = field(default_factory=lambda: os.getenv("embedding_model_name", "text-embedding-ada-002"))
    embed_endpoint: str | None = field(default_factory=lambda: os.getenv("embedding_endpoint_url"))
    embed_api_key: str | None = field(default_factory=lambda: os.getenv("embedding_api_key"))
    embed_api_version: str = field(default_factory=lambda: os.getenv("embedding_api_version", "2023-05-15"))

    # Default benchmark size for LLM-touching evaluation (1 call/question).
    benchmark_size: int = field(default_factory=lambda: int(os.getenv("BENCHMARK_SIZE", "10")))

    def to_public_dict(self) -> dict:
        """Non-secret view for /health and the UI."""
        return {
            "llm_enabled": self.llm_enabled,
            "llm_call_budget": self.llm_call_budget,
            "chat_deployment": self.chat_deployment,
            "embed_deployment": self.embed_deployment,
            "benchmark_size": self.benchmark_size,
            "chat_configured": bool(self.chat_api_key and self.chat_endpoint),
            "embed_configured": bool(self.embed_api_key and self.embed_endpoint),
        }


SETTINGS = Settings()


def load_recipe(name_or_path: str | Path) -> dict[str, Any]:
    """Load an experiment recipe by config name (configs/<name>.yaml) or path."""
    p = Path(name_or_path)
    if not p.exists():
        p = CONFIGS_DIR / f"{name_or_path}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"Recipe not found: {name_or_path}")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_recipes() -> list[str]:
    return sorted(p.stem for p in CONFIGS_DIR.glob("*.yaml"))
