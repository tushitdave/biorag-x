"""RetrievalConfig (the frontend flow) -> pipeline recipe, with validation.

Only combinations the backend can really run are accepted: MedCPT vectors exist for
every built chunking strategy (and agentic chunks are encoded at question time), while
ada-002 vectors exist only for NB04's semantic chunks; ANN indexes are built for MedCPT
vectors; a strategy must have finished building before it can be used; and the graph /
PageIndex channels and synonym expansion need their index (scripts/build_graph_pageindex.py).
"""
from __future__ import annotations

from api.contract import RetrievalConfig
from chunking import registry

EMBED_MODELS = {"medcpt": "medcpt", "ada002_azure": "text-embedding-ada-002"}
CHANNELS = {"lexical", "dense", "graph", "pageindex"}
QUERY_MODES = {"as_is", "synonyms", "router"}
INDEXES = {"flat", "hnsw", "ivf"}
RERANKERS = {"none", "medcpt_ce", "lexical_overlap", "cross_encoder", "colbert", "llm_reranker"}
EVIDENCE = {"mmr", "topk"}
CORPORA = {"dev", "full"}


def embedding_for(chunking: str) -> str:
    """The only embedding model with vectors for this chunking strategy."""
    return "ada002_azure" if chunking == "semantic_nb04" else "medcpt"


def validate(cfg: RetrievalConfig, corpus: str = "full") -> None:
    """Raise ValueError with a plain-language reason if the flow cannot run."""
    if corpus not in CORPORA:
        raise ValueError(f"Unknown corpus: {corpus}")
    if cfg.chunking not in registry.STRATEGIES:
        raise ValueError(f"Unknown chunking strategy: {cfg.chunking}")
    st = registry.status(cfg.chunking)
    if st["status"] != "ready":
        pct = f" ({st['progress']:.0%})" if st["progress"] is not None else ""
        raise ValueError(f"Chunking '{registry.STRATEGIES[cfg.chunking][0]}' is "
                         f"{st['status']}{pct}; try again when it is ready")
    if cfg.embedding not in EMBED_MODELS:
        raise ValueError(f"Unknown embedding model: {cfg.embedding}")
    if not cfg.channels or set(cfg.channels) - CHANNELS:
        raise ValueError("channels must be a non-empty subset of lexical, dense, graph, pageindex")
    from retrieval.structured import is_ready as channel_ready
    if cfg.query not in QUERY_MODES:
        raise ValueError(f"Unknown query handling: {cfg.query}")
    if cfg.query == "synonyms" and not channel_ready("synonyms"):
        raise ValueError("The synonym dictionary is not built; run scripts/build_graph_pageindex.py")
    for ch in ("graph", "pageindex"):
        if ch in cfg.channels and not channel_ready(ch):
            raise ValueError(f"The {ch} index is not built; run scripts/build_graph_pageindex.py")
    if cfg.index not in INDEXES:
        raise ValueError(f"Unknown vector index: {cfg.index}")
    if cfg.reranker not in RERANKERS:
        raise ValueError(f"Unknown reranker: {cfg.reranker}")
    if cfg.evidence not in EVIDENCE:
        raise ValueError(f"Unknown evidence selector: {cfg.evidence}")
    if corpus == "dev" and cfg.chunking == "semantic_nb04":
        raise ValueError("NB04's ada-002 semantic chunks are not part of the dev corpus")
    if "dense" in cfg.channels:
        if embedding_for(cfg.chunking) != cfg.embedding:
            raise ValueError(f"'{cfg.chunking}' chunks have {embedding_for(cfg.chunking)} "
                             f"vectors, not {cfg.embedding}")
        if cfg.index != "flat" and cfg.embedding != "medcpt":
            raise ValueError("HNSW / IVF indexes are built for MedCPT vectors only")


def to_recipe(cfg: RetrievalConfig, corpus: str = "full") -> dict:
    """The recipe dict retrieval.hybrid.retrieve understands."""
    validate(cfg, corpus)
    return {
        "corpus": corpus,
        "chunking": {"method": cfg.chunking},
        "query": {"mode": cfg.query},
        "embedding": {"model": EMBED_MODELS[cfg.embedding]},
        "ann": {"index": cfg.index, "ef_search": cfg.ef_search, "nprobe": cfg.nprobe},
        "retrieval": {"channels": list(cfg.channels), "top_k": cfg.depth},
        "fusion": {"method": "rrf", "k": cfg.rrf_k},
        "reranker": {"enabled": cfg.reranker != "none", "method": cfg.reranker,
                     "candidates": cfg.rerank_candidates},
        "evidence": {"selector": "topk" if cfg.evidence == "topk" else "coverage_diversity",
                     "top_k": cfg.evidence_k},
    }


def _matters(field: str, cfg: RetrievalConfig) -> bool:
    """Whether a field affects this config's results (e.g. efSearch only for HNSW)."""
    if field == "ef_search":
        return cfg.index == "hnsw"
    if field == "nprobe":
        return cfg.index == "ivf"
    if field == "rerank_candidates":
        return cfg.reranker != "none"
    if field == "channels":
        return cfg.query != "router"          # the router picks channels per question
    if field in ("embedding", "index"):
        return cfg.query == "router" or "dense" in cfg.channels
    if field == "rrf_k":
        return cfg.query == "router" or len(cfg.channels) > 1
    return True


def canonical(cfg: RetrievalConfig) -> RetrievalConfig:
    """Same flow with settings that do not affect results reset to defaults (so two
    configs that behave identically compare equal)."""
    defaults = RetrievalConfig().model_dump()
    data = cfg.model_dump()
    return RetrievalConfig(**{k: (v if _matters(k, cfg) else defaults[k]) for k, v in data.items()})


def changes(a: RetrievalConfig, b: RetrievalConfig) -> list[tuple[str, str, str]]:
    """Fields that differ between two configs and matter to at least one, as strings."""
    da, db = a.model_dump(), b.model_dump()
    fmt = lambda v: " + ".join(v) if isinstance(v, list) else str(v)  # noqa: E731
    return [(k, fmt(da[k]), fmt(db[k])) for k in da
            if da[k] != db[k] and (_matters(k, a) or _matters(k, b))]
