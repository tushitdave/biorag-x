"""LLM guard for BioRAG-X.

Design goals (POC constraints):
  * LLM_ENABLED master switch — when off, no network calls are ever made.
  * Hard call budget (default 12) enforced process-wide.
  * On-disk cache keyed by a hash of (model, prompt) — repeated/identical calls
    are free and deterministic, so re-runs do not burn the budget.
  * Query embeddings are cached the same way.

Every consumer (generation, optional query expansion) goes through here, so the
budget is centrally auditable via ``llm_usage()``.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Optional

from .config import SETTINGS
from .paths import CACHE_DIR

_LLM_CACHE = CACHE_DIR / "llm"
_EMB_CACHE = CACHE_DIR / "embeddings"
_LLM_CACHE.mkdir(parents=True, exist_ok=True)
_EMB_CACHE.mkdir(parents=True, exist_ok=True)

_USAGE = {"live_calls": 0, "cache_hits": 0, "total_tokens": 0, "seconds": 0.0, "budget_blocked": 0}

_chat_client = None
_embed_client = None


def _key(*parts: str) -> str:
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()[:24]


def llm_usage() -> dict:
    return dict(_USAGE, budget=SETTINGS.llm_call_budget, enabled=SETTINGS.llm_enabled)


def budget_remaining() -> int:
    return max(0, SETTINGS.llm_call_budget - _USAGE["live_calls"])


def _get_chat_client():
    global _chat_client
    if _chat_client is None:
        from openai import AzureOpenAI
        _chat_client = AzureOpenAI(
            api_key=SETTINGS.chat_api_key,
            api_version=SETTINGS.chat_api_version,
            azure_endpoint=SETTINGS.chat_endpoint,
        )
    return _chat_client


def _get_embed_client():
    global _embed_client
    if _embed_client is None:
        from openai import AzureOpenAI
        _embed_client = AzureOpenAI(
            api_key=SETTINGS.embed_api_key,
            api_version=SETTINGS.embed_api_version,
            azure_endpoint=SETTINGS.embed_endpoint,
        )
    return _embed_client


def chat_json(prompt: str, *, max_tokens: int = 700, temperature: float = 0.0,
              system: str = "You are a precise JSON-only API. Output only valid JSON.",
              force_json: bool = True) -> tuple[Optional[dict], str, str]:
    """Return (parsed_json_or_None, raw_text, source).

    source in {"cache", "live", "disabled", "budget"}. Cached results never count
    against the budget; a disabled switch or exhausted budget returns (None, "", ...)
    so callers can fall back to a deterministic path.
    """
    cache_file = _LLM_CACHE / f"{_key(SETTINGS.chat_deployment, prompt)}.json"
    if cache_file.exists():
        _USAGE["cache_hits"] += 1
        raw = cache_file.read_text(encoding="utf-8")
        return _safe_json(raw), raw, "cache"

    if not SETTINGS.llm_enabled:
        return None, "", "disabled"
    if budget_remaining() <= 0:
        _USAGE["budget_blocked"] += 1
        return None, "", "budget"

    client = _get_chat_client()
    kwargs = dict(
        model=SETTINGS.chat_deployment,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}],
        max_tokens=max_tokens, temperature=temperature,
    )
    if force_json:
        kwargs["response_format"] = {"type": "json_object"}

    t0 = time.perf_counter()
    resp = client.chat.completions.create(**kwargs)
    _USAGE["seconds"] += time.perf_counter() - t0
    _USAGE["live_calls"] += 1
    if resp.usage:
        _USAGE["total_tokens"] += resp.usage.total_tokens
    raw = resp.choices[0].message.content or ""
    cache_file.write_text(raw, encoding="utf-8")
    return _safe_json(raw), raw, "live"


def embed_query(text: str) -> Optional[list[float]]:
    """Embed a single query with ada-002 (cached). Returns None if unavailable.

    Query embedding is cheap and not counted against the chat budget, but it is
    still cached so repeated queries are free and offline-friendly.
    """
    cache_file = _EMB_CACHE / f"{_key(SETTINGS.embed_deployment, text)}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))
    if not (SETTINGS.embed_api_key and SETTINGS.embed_endpoint):
        return None
    client = _get_embed_client()
    resp = client.embeddings.create(model=SETTINGS.embed_deployment, input=[text])
    vec = list(resp.data[0].embedding)
    cache_file.write_text(json.dumps(vec), encoding="utf-8")
    return vec


def _safe_json(raw: str) -> Optional[dict]:
    try:
        return json.loads(raw)
    except Exception:
        import re
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
        return None
