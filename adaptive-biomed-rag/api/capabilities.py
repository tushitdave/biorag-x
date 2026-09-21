"""Selectable capabilities for the frontend, grounded in the research.

Produces the per-step option sets in pipeline order (chunking -> embedding ->
index -> query -> retrieval -> reranker -> evidence -> generation). Options from
the research design
that are not built yet are listed with ``planned=True`` and ``available=False`` so
the UI can show the roadmap without offering choices that do nothing. Azure
availability is detected from configured keys. Notes carry the evidence for each
choice.
"""
from __future__ import annotations

from chunking import registry as chunk_registry
from common import paths
from common.config import SETTINGS
from retrieval.structured import is_ready as channel_ready, manifest as channel_manifest
from .contract import Capabilities, LayerOptions, Option


def _planned(id: str, label: str, note: str, cost: str = "medium",
             quality: str = "medium", uses_llm: bool = False) -> Option:
    return Option(id=id, label=label, note=f"Planned: {note}", cost=cost, quality=quality,
                  available=False, planned=True, uses_llm=uses_llm)


def build_capabilities() -> Capabilities:
    azure_chat = bool(SETTINGS.chat_api_key and SETTINGS.chat_endpoint)
    azure_embed = bool(SETTINGS.embed_api_key and SETTINGS.embed_endpoint)
    medcpt_ready = paths.MEDCPT_FAISS.exists()

    chunk_opts = []
    for sid, (label, what, source) in chunk_registry.STRATEGIES.items():
        st = chunk_registry.status(sid)
        ready = st["status"] == "ready"
        if sid == "agentic":
            ready = ready and azure_chat
        chunk_opts.append(Option(
            id=sid, label=label, note=f"{what} ({source})",
            cost="high" if sid == "agentic" else "low",
            quality="medium" if sid == "semantic_nb04" else "high",
            default=sid == "passage", available=ready, uses_llm=sid == "agentic",
            status=st["status"], progress=st["progress"],
        ))
    chunking = LayerOptions(layer="chunking", label="Chunking", options=chunk_opts)

    embedding = LayerOptions(
        layer="embedding", label="Embedding model",
        options=[
            Option(id="medcpt", label="MedCPT (biomedical)", cost="low", quality="high",
                   note="Local; trained on 255M PubMed searches. Vectors for every chunking "
                        "strategy except NB04's.",
                   default=True, available=medcpt_ready),
            Option(id="ada002_azure", label="Azure ada-002", cost="medium", quality="medium",
                   note="Generic model; vectors only for NB04's semantic chunks; each query "
                        "calls Azure.",
                   available=azure_embed and paths.SEMANTIC_EMBEDDINGS.exists()),
            _planned("bge_m3", "BGE-M3", "strong general-purpose comparison model."),
        ],
    )

    index = LayerOptions(
        layer="index", label="Vector index",
        options=[
            Option(id="flat", label="Exact (FAISS flat)", cost="low", quality="high",
                   note="Compares the question with every vector; the reference for ANN.",
                   default=True),
            Option(id="hnsw", label="HNSW (approximate)", cost="low", quality="high",
                   note="Graph search (M=32). Higher efSearch: closer to exact, slower.",
                   available=medcpt_ready),
            Option(id="ivf", label="IVF (approximate)", cost="low", quality="medium",
                   note="Searches only the nprobe nearest clusters (nlist = 4 x sqrt(n)).",
                   available=medcpt_ready),
            _planned("ivf_pq", "IVF-PQ", "compressed vectors for very large corpora.", cost="low"),
        ],
    )

    s = channel_manifest("synonyms")
    query = LayerOptions(
        layer="query", label="Query handling",
        options=[
            Option(id="as_is", label="Use question as typed", cost="low", quality="medium",
                   note="No rewriting: the question goes straight to search.", default=True),
            Option(id="synonyms", label="Synonym expansion", cost="low", quality="medium",
                   note=f"Adds long forms for abbreviations and short forms for long forms, from "
                        f"{s.get('pairs', 0):,} definitions found in the corpus (e.g. ECMO → "
                        f"extracorporeal membrane oxygenation). Feeds BM25, graph and PageIndex.",
                   available=channel_ready("synonyms"),
                   status="ready" if channel_ready("synonyms") else "missing"),
            _planned("hyde", "HyDE", "search with an LLM-written hypothetical answer.",
                     uses_llm=True),
            _planned("query2doc", "Query2Doc", "append an LLM pseudo-document to the query.",
                     uses_llm=True),
            _planned("decompose", "Decomposition", "split multi-part questions into sub-queries.",
                     uses_llm=True),
            Option(id="router", label="Agentic router", cost="low", quality="high",
                   note="Picks search channels and synonyms per question by rules, checks that "
                        "the evidence covers the question's key terms, and retries once with every "
                        "channel if not. Overrides your search channels. No LLM.",
                   available=True, status="ready"),
        ],
    )

    g, t = channel_manifest("graph"), channel_manifest("pageindex")
    retrieval = LayerOptions(
        layer="retrieval", label="Search channels", multi_select=True,
        options=[
            Option(id="lexical", label="BM25 (keywords)", cost="low", quality="medium",
                   note="Matches exact words: gene names, drugs, abbreviations.", default=True),
            Option(id="dense", label="Dense (meaning)", cost="low", quality="high",
                   note="Matches meaning, so paraphrased questions still find evidence.",
                   default=True),
            Option(id="graph", label="Graph traversal", cost="low", quality="medium",
                   note=f"Finds the question's entities in a rule-built entity graph "
                        f"({g.get('entities', 0):,} entities, {g.get('edges', 0):,} links) and "
                        f"follows their strongest links one hop; reaches passages that never "
                        f"name the question's entity.",
                   available=channel_ready("graph"),
                   status="ready" if channel_ready("graph") else "missing"),
            Option(id="pageindex", label="PageIndex (tree)", cost="low", quality="medium",
                   note=f"No vectors at question time: opens the best topics of a "
                        f"{t.get('topics', 0)}-topic / {t.get('leaves', 0)}-section table of "
                        f"contents by their words, then reads those sections' passages with BM25.",
                   available=channel_ready("pageindex"),
                   status="ready" if channel_ready("pageindex") else "missing"),
        ],
    )

    reranker = LayerOptions(
        layer="reranker", label="Reranker",
        options=[
            Option(id="none", label="None (keep fused order)", cost="low", quality="high",
                   note="Keeps the fused order; fastest.", default=True),
            Option(id="medcpt_ce", label="MedCPT cross-encoder", cost="high", quality="high",
                   note="Reads question and passage together. ~1.3 s per 20 candidates on M1.",
                   available=medcpt_ready),
            Option(id="lexical_overlap", label="Word overlap", cost="low", quality="low",
                   note="Re-sorts by shared words; dropped to 2/7 on conversational questions."),
            Option(id="cross_encoder", label="Word-pair overlap (proxy)", cost="low", quality="low",
                   note="Stand-in for a cross-encoder using word and word-pair overlap."),
            Option(id="colbert", label="Fuzzy token match (proxy)", cost="low", quality="low",
                   note="Stand-in for ColBERT using fuzzy token matching."),
            Option(id="llm_reranker", label="GPT-4o ranking", cost="high", quality="high",
                   note="Chat only: one LLM call per question, so not allowed in lab runs.",
                   available=azure_chat, uses_llm=True),
        ],
    )

    evidence = LayerOptions(
        layer="evidence", label="Evidence selection",
        options=[
            Option(id="mmr", label="Diverse (MMR)", cost="low", quality="high",
                   note="Best-scored passages, skipping near-duplicates.", default=True),
            Option(id="topk", label="Top-k by score", cost="low", quality="medium",
                   note="Simply the k best-scored passages."),
        ],
    )

    generation = LayerOptions(
        layer="generation", label="Answer generator",
        options=[
            Option(id="gpt4o_azure", label="Azure GPT-4o", cost="high", quality="high",
                   note="Answers only from the evidence, citing each claim; budget-capped.",
                   default=True, available=azure_chat, uses_llm=True),
            _planned("ollama_llama32", "Llama 3.2 (local)", "free local generator.",
                     cost="low", uses_llm=True),
        ],
    )

    return Capabilities(layers=[chunking, embedding, index, query, retrieval, reranker,
                                evidence, generation])
