"""Synonym expansion from abbreviations defined in the corpus itself (no LLM, no UMLS).

Build: every "long form (SHORT)" definition in the 28k passages is found with the
Schwartz & Hearst (2003) algorithm - the short form's characters must appear, in
order, right to left, in the words just before the parenthesis, and its first
character must start a word:

    "sex determining region Y (SRY)"   ->  SRY <-> sex determining region y
    "extracorporeal membrane oxygenation (ECMO)"

A short form keeps at most MAX_LONG_FORMS long forms, each at least MIN_SHARE of its
definitions (ambiguous abbreviations keep their common senses only).

Query: a short form typed in the question (exact case) adds its long forms; a long
form typed in the question adds its short form. At most MAX_ADDED expansions. The
expanded text feeds the keyword-based channels (BM25, graph, PageIndex); dense search
and the reranker keep the question as typed, since they already match meaning.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

MAX_LONG_FORMS = 2
MIN_SHARE = 0.2
MAX_ADDED = 6
_PAREN = re.compile(r"\(([^()]{2,10})\)")
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-']*")


def _valid_short(sf: str) -> bool:
    sf = sf.strip()
    return (2 <= len(sf) <= 10 and sf[0].isalnum() and any(c.isupper() for c in sf)
            and len(sf.split()) <= 2 and not re.search(r"[=%;,<>]", sf)
            and not sf.replace(".", "").isdigit())


def best_long_form(sf: str, before: str) -> str | None:
    """Schwartz & Hearst: shortest span of ``before`` whose characters cover ``sf``."""
    words = before.split()
    cand = " ".join(words[-min(len(sf) + 5, len(sf) * 2):])
    s, l = len(sf) - 1, len(cand) - 1
    while s >= 0:
        c = sf[s].lower()
        if not c.isalnum():
            s -= 1
            continue
        while l >= 0 and (cand[l].lower() != c or (s == 0 and l > 0 and cand[l - 1].isalnum())):
            l -= 1
        if l < 0:
            return None
        l -= 1
        s -= 1
    lf = cand[cand.rfind(" ", 0, l + 1) + 1:].strip(" ,;:-").lower()
    if len(lf) <= len(sf) or sf.lower() in lf.split():
        return None
    return lf


def definitions(text: str):
    for m in _PAREN.finditer(text or ""):
        sf = m.group(1).strip()
        if not _valid_short(sf):
            continue
        before = text[max(0, m.start() - 200):m.start()]
        before = re.split(r"[.;:!?(\[]\s", before)[-1]           # same clause
        lf = best_long_form(sf, before)
        if lf:
            yield sf, lf


def build(texts: list[str], out_dir: Path, log: Callable[[str], None] = print) -> dict:
    counts: Counter = Counter()
    for t in texts:
        counts.update(definitions(t))
    by_sf: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for (sf, lf), c in counts.items():
        by_sf[sf].append((lf, c))
    rows = []
    for sf, lfs in by_sf.items():
        total = sum(c for _, c in lfs)
        for lf, c in sorted(lfs, key=lambda x: -x[1])[:MAX_LONG_FORMS]:
            if c / total >= MIN_SHARE:
                rows.append((sf, lf, c))
    df = pd.DataFrame(rows, columns=["short", "long", "count"])
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "abbreviations.parquet", index=False)
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "definitions_found": int(sum(counts.values())), "short_forms": int(df["short"].nunique()),
                "pairs": int(len(df)),
                "params": {"max_long_forms": MAX_LONG_FORMS, "min_share": MIN_SHARE,
                           "max_added": MAX_ADDED}}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"  synonyms: {manifest['pairs']:,} abbreviation pairs")
    return manifest


class SynonymExpander:
    def __init__(self, syn_dir: Path):
        df = pd.read_parquet(syn_dir / "abbreviations.parquet")
        self.long_of: dict[str, list[str]] = df.groupby("short")["long"].agg(list).to_dict()
        best = df.sort_values("count", ascending=False).drop_duplicates("long")
        self.short_of: dict[str, str] = dict(zip(best["long"], best["short"]))

    def known(self, question: str) -> list[str]:
        """Terms of the question that the dictionary can expand."""
        return [t for t, _ in self._matches(question)]

    def _matches(self, question: str) -> list[tuple[str, list[str]]]:
        words = _WORD.findall(question)
        out = [(w, self.long_of[w]) for w in dict.fromkeys(words) if w in self.long_of]
        low = [w.lower() for w in words]
        for n in range(6, 0, -1):
            for i in range(len(low) - n + 1):
                phrase = " ".join(low[i:i + n])
                if n == 1 and (phrase in ENGLISH_STOP_WORDS or len(phrase) < 4):
                    continue
                sf = self.short_of.get(phrase)
                if sf and sf not in words:
                    out.append((phrase, [sf]))
        return out

    def expand(self, question: str) -> tuple[str, list[dict]]:
        added, seen, typed = [], {w.lower() for w in _WORD.findall(question)}, question.lower()
        for term, forms in self._matches(question):
            for f in forms:
                key = f.lower().rstrip("s")                     # microrna / micrornas: once
                if key in seen or f.lower() in typed or len(added) >= MAX_ADDED:
                    continue
                seen.add(key)
                added.append({"term": term, "added": f})
        text = question + (" " + " ".join(a["added"] for a in added) if added else "")
        return text, added
