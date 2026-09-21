"""Repair-tolerant parsing of LLM JSON into contract Claim/Citation objects.

The LLM is asked for a strict JSON shape, but models drift: they may return a
bare string, misname keys, nest citations, or emit a single claim as an object.
This module normalizes whatever comes back into a clean list of ``Claim`` objects
plus the raw answer string, tolerating common malformations without throwing.
"""
from __future__ import annotations

from typing import Any

from api.contract import Claim


_VALID_STATUS = {"sufficient", "weak", "insufficient"}


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _coerce_citations(raw: Any) -> list[str]:
    """Accept a string, list of strings, or list of dicts with an id-like key."""
    out: list[str] = []
    for c in _as_list(raw):
        if isinstance(c, str):
            cid = c.strip()
            if cid:
                out.append(cid)
        elif isinstance(c, dict):
            for k in ("passage_id", "id", "ref", "ref_id", "citation"):
                if c.get(k):
                    out.append(str(c[k]).strip())
                    break
        elif c is not None:
            out.append(str(c).strip())
    # De-dup preserving order.
    seen: set[str] = set()
    result = []
    for cid in out:
        if cid and cid not in seen:
            seen.add(cid)
            result.append(cid)
    return result


def parse_structured_answer(data: dict | None) -> tuple[str, list[Claim], str]:
    """Return (answer_text, claims, evidence_status) from possibly-messy JSON.

    Never raises; falls back to empty/neutral values so callers can decide whether
    to use an extractive fallback instead.
    """
    if not isinstance(data, dict):
        return "", [], "insufficient"

    answer = data.get("answer") or data.get("text") or data.get("response") or ""
    if not isinstance(answer, str):
        answer = str(answer)
    answer = answer.strip()

    raw_claims = data.get("claims") or data.get("statements") or []
    claims: list[Claim] = []
    for rc in _as_list(raw_claims):
        if isinstance(rc, dict):
            text = rc.get("text") or rc.get("claim") or rc.get("statement") or ""
            citations = _coerce_citations(
                rc.get("citations") or rc.get("citation") or rc.get("passage_ids")
            )
        elif isinstance(rc, str):
            text, citations = rc, []
        else:
            continue
        text = str(text).strip()
        if text:
            claims.append(Claim(text=text, citations=citations))

    status = str(data.get("evidence_status") or data.get("status") or "").strip().lower()
    if status not in _VALID_STATUS:
        status = "sufficient" if claims else "insufficient"

    return answer, claims, status
