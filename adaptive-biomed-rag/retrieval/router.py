"""Agentic retrieval router: picks the search route per question, checks the evidence,
retries once with a wider route if it is weak (CRAG-style). Local rules - no LLM.

Round 1 - route from question features (rules add to BM25 + dense, which always run):
  R1  2+ graph entities, or 1 entity + a relation word  -> + graph traversal
  R2  an abbreviation / long form the dictionary knows  -> synonym expansion
  R3  list / summary question with no graph entity      -> + PageIndex
  none fired                                             -> BM25 + dense as typed

Check - key-term coverage: share of the question's key terms (content words and
entities, matched on a 6-letter stem) found anywhere in the selected evidence.

Round 2 - only if coverage < WEAK_COVERAGE: every ready channel (BM25, dense, graph,
PageIndex) with synonym expansion. The round with higher coverage is kept (ties keep
round 1). MAX_ROUNDS = 2. With agentic chunking there is no second round (it would
repeat the LLM call).

Coverage deliberately does not use the MedCPT judge, so the Ask page's judge still
scores the router's evidence independently.
"""
from __future__ import annotations

import re

from chunking.base import PATTERNS
from retrieval import structured

WEAK_COVERAGE = 0.6
MAX_ROUNDS = 2
STEM = 6
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]+")
_STOP = set("the a an of and or to in for with on by is are was were be as at from that this these "
            "those what which who how why when where does do did can could may might should would "
            "will it its their there any into than then also not no such via per about has have "
            "had been being used use using role effect effects known main list name describe "
            "explain between".split())
_RELATION = re.compile(PATTERNS["relation_trigger"].pattern + r"|\b(?:target|bind|interact|"
                       r"regulat|mediat|phosphorylat|block|suppress|encode)", re.I)
_LIST = re.compile(r"^\s*(?:list|name|enumerate)\b|^\s*(?:which|what) (?:\w+ ){0,3}are\b", re.I)
_SUMMARY = re.compile(r"^\s*(?:describe|explain|summari[sz]e|discuss|what is known|"
                      r"what are the (?:main|known)|how (?:does|do|is|are))\b", re.I)
_YESNO = re.compile(r"^\s*(?:is|are|does|do|did|can|could|should|was|were|has|have|will)\b", re.I)


def question_type(question: str) -> str:
    if _LIST.search(question):
        return "list"
    if _SUMMARY.search(question):
        return "summary"
    if _YESNO.search(question):
        return "yes/no"
    return "factoid"


def key_terms(question: str) -> list[str]:
    words = [w.lower() for w in _WORD.findall(question)]
    return list(dict.fromkeys(w for w in words if w not in _STOP and len(w) > 2))


def coverage(question: str, evidence: list[dict]) -> tuple[float, list[str]]:
    terms = key_terms(question)
    if not terms:
        return 1.0, []
    text = " ".join(p.get("text", "") for p in evidence).lower()
    missing = [t for t in terms if t[:STEM] not in text]
    return round(1 - len(missing) / len(terms), 3), missing


def features(question: str) -> dict:
    seeds = structured.get_graph().seeds(question) if structured.is_ready("graph") else []
    abbrevs = structured.get_synonyms().known(question) if structured.is_ready("synonyms") else []
    return {"type": question_type(question), "entities": seeds,
            "relation_word": bool(_RELATION.search(question)), "abbreviations": abbrevs}


def plan(f: dict) -> tuple[list[str], str, list[str]]:
    """Round-1 channels, query mode and the rules that fired."""
    channels, mode, fired = ["lexical", "dense"], "as_is", []
    n = len(f["entities"])
    if structured.is_ready("graph") and (n >= 2 or (n == 1 and f["relation_word"])):
        channels.append("graph")
        fired.append(f"R1 {n} entit{'y' if n == 1 else 'ies'}"
                     f"{' + relation word' if f['relation_word'] else ''} → graph traversal")
    if structured.is_ready("synonyms") and f["abbreviations"]:
        mode = "synonyms"
        fired.append(f"R2 known abbreviation ({', '.join(f['abbreviations'][:3])}) → synonyms")
    if structured.is_ready("pageindex") and f["type"] in ("list", "summary") and n == 0:
        channels.append("pageindex")
        fired.append(f"R3 {f['type']} question, no entity → PageIndex")
    if not fired:
        fired.append("R0 no rule fired → BM25 + dense as typed")
    return channels, mode, fired


def wide_route() -> tuple[list[str], str]:
    channels = ["lexical", "dense"] + [c for c in ("graph", "pageindex") if structured.is_ready(c)]
    return channels, "synonyms" if structured.is_ready("synonyms") else "as_is"


def retrieve_routed(question: str, recipe: dict, reranker_method: str | None = None) -> dict:
    from retrieval.hybrid import retrieve                     # avoid an import cycle

    def run(channels: list[str], mode: str) -> dict:
        r = {**recipe, "query": {"mode": mode},
             "retrieval": {**recipe.get("retrieval", {}), "channels": channels}}
        return retrieve(question, r, reranker_method=reranker_method)

    f = features(question)
    channels, mode, fired = plan(f)
    res = run(channels, mode)
    cov, missing = coverage(question, res["passages"])
    rounds = [{"round": 1, "channels": channels, "query": mode, "coverage": cov, "missing": missing}]
    chosen = 1
    agentic = recipe.get("chunking", {}).get("method") == "agentic"
    if cov < WEAK_COVERAGE and not agentic and MAX_ROUNDS > 1:
        ch2, mode2 = wide_route()
        if (set(ch2), mode2) != (set(channels), mode):
            res2 = run(ch2, mode2)
            cov2, missing2 = coverage(question, res2["passages"])
            rounds.append({"round": 2, "channels": ch2, "query": mode2, "coverage": cov2,
                           "missing": missing2})
            if cov2 > cov:
                res, chosen = res2, 2
    res["trace"] = dict(res["trace"]) | {"router": {
        "features": {**f, "entities": [structured.get_graph().surface[e] for e in f["entities"]]}
        if f["entities"] else f,
        "rules": fired, "rounds": rounds, "chosen_round": chosen,
        "weak_coverage": WEAK_COVERAGE}}
    return res
