"""Shared text helpers for the chunkers (ported from NB03-05 so app and notebooks agree).

Tokens are whitespace tokens (the notebooks' approximate token unit); sentences
split on terminal punctuation followed by an upper-case letter or digit. A chunk
is a plain dict with the columns every chunk table uses.
"""
from __future__ import annotations

import hashlib
import re

TOKEN_RE = re.compile(r"\S+")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

# Biomedical signal patterns (NB04 PATTERNS): used by biomedical-aware chunking
# and by the adaptive router's document profile.
PATTERNS = {
    "gene_like": re.compile(r"\b(?:BRCA\d*|TP\d+|EGFR|KRAS|BRAF|MYC|PTEN|APC)\b", re.I),
    "protein_like": re.compile(r"\b(?:p53|COX[- ]?2|TNF[- ]?alpha|IL[- ]?6|VEGF|HER2)\b", re.I),
    "drug_like": re.compile(r"\b(?:aspirin|ibuprofen|metformin|olaparib|pembrolizumab|trastuzumab|"
                            r"tamoxifen|bevacizumab)\b", re.I),
    "disease_like": re.compile(r"\b(?:breast cancer|ovarian cancer|lung cancer|diabetes|melanoma|"
                               r"alzheimer(?:'s)? disease)\b", re.I),
    "negation": re.compile(r"\b(?:no|not|never|without|lack(?:s|ing)?|absence of|did not|does not)\b",
                           re.I),
    "relation_trigger": re.compile(r"\b(?:inhibits?|activates?|associated with|increases?|decreases?|"
                                   r"causes?|prevents?|reduces?|induces?|expressed in|interacts with|"
                                   r"sensitizes?|resistant to|responsive to)\b", re.I),
}


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text or "")


def n_tokens(text: str) -> int:
    return len(tokens(text))


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def sentences(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def chunk_id(parent_id: str, strategy: str, idx: int, text: str) -> str:
    raw = f"{parent_id}|{strategy}|{idx}|{text}".encode()
    return "CHK-" + hashlib.sha256(raw).hexdigest()[:20]


def make_chunk(parent_id: str, strategy: str, idx: int, text: str,
               start_token: int = -1, end_token: int = -1) -> dict:
    text = normalize_spaces(text)
    return {
        "chunk_id": chunk_id(parent_id, strategy, idx, text),
        "parent_passage_id": parent_id,
        "strategy": strategy,
        "chunk_index": idx,
        "text": text,
        "n_tokens": n_tokens(text),
        "start_token": start_token,
        "end_token": end_token,
    }


def windows(n: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """[start, end) token windows of ``size`` with ``overlap``, covering 0..n."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    out, start, step = [], 0, size - overlap
    while start < n:
        end = min(start + size, n)
        out.append((start, end))
        if end == n:
            break
        start += step
    return out
