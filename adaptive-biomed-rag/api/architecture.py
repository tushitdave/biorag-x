"""Architecture diagrams (process flow + system design) generated from the live system.

Every threshold, size and count shown is read from the constant the code uses, the
build status, or a manifest - so the diagrams cannot drift from behaviour. Each
diagram is Mermaid source plus a details entry per node (what it does, its rule or
thresholds, the source file, and live / building / planned status).
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np

import chunking.agentic as agentic_mod
import chunking.strategies as S
import generation.citation_validator as cv
import evaluation.rag_metrics as rm
import indexes.bm25 as bm25_mod
import indexes.dense as dense_mod
import retrieval.agentic as ragentic
import retrieval.compare as cmp_mod
import retrieval.evidence_selector as es
import graph.evidence_graph as graph_mod
import pageindex.tree as tree_mod
import retrieval.router as router_mod
import retrieval.synonyms as syn_mod
from api.contract import RetrievalConfig
from chunking import registry
from common import paths
from common.config import SETTINGS

CLASSES = """
  classDef data fill:#e3f2fd,stroke:#1976d2,color:#0b2a44;
  classDef step fill:#ffffff,stroke:#5b6b7b,color:#1c2733;
  classDef decision fill:#fdf1d6,stroke:#b7791f,color:#3d2a00;
  classDef llm fill:#f3e5f5,stroke:#7b1fa2,color:#2e0f3a;
  classDef model fill:#e0f2f1,stroke:#00796b,color:#003b33;
  classDef ready fill:#e8f5e9,stroke:#2e7d5b,color:#0f3322;
  classDef building fill:#fff8e1,stroke:#b7791f,color:#3d2a00,stroke-dasharray: 4 3;
  classDef planned fill:#f5f5f5,stroke:#9e9e9e,color:#6b6b6b,stroke-dasharray: 6 4;
  classDef output fill:#e3eef3,stroke:#2c6e8f,color:#0b2a44,font-weight:bold;
"""

LEGEND = [
    {"label": "Data / store", "cls": "data"},
    {"label": "Processing step", "cls": "step"},
    {"label": "Decision (rule / threshold)", "cls": "decision"},
    {"label": "Uses the LLM (GPT-4o)", "cls": "llm"},
    {"label": "Local model (MedCPT)", "cls": "model"},
    {"label": "Index ready", "cls": "ready"},
    {"label": "Index building", "cls": "building"},
    {"label": "Planned (research design)", "cls": "planned"},
    {"label": "Output", "cls": "output"},
]


def _node(title: str, what: str, rules: list[str] | None = None, source: str = "",
          status: str = "live") -> dict:
    return {"title": title, "what": what, "rules": rules or [], "source": source,
            "status": status}


@lru_cache(maxsize=1)
def _semantic_threshold_from_cache() -> float | None:
    f = paths.CHUNK_INDEX_DIR / "_sentence_similarities.npz"
    if not f.exists():
        return None
    sims = np.load(f, allow_pickle=True)["sims"]
    pooled = np.concatenate([s for s in sims if len(s)])
    return float(np.percentile(pooled, S.SEMANTIC_PERCENTILE))


def _semantic_threshold() -> float | None:
    m = paths.CHUNK_INDEX_DIR / "semantic" / "manifest.json"
    if m.exists():
        return json.loads(m.read_text()).get("semantic", {}).get("threshold")
    return _semantic_threshold_from_cache()


def _status_suffix(strategy: str) -> tuple[str, str]:
    """(label suffix, css class) from the live build status."""
    st = registry.status(strategy)
    if st["status"] == "ready":
        return "ready", "ready"
    if st["status"] in ("building", "paused"):
        pct = f" {st['progress']:.0%}" if st["progress"] is not None else ""
        return f"{st['status']}{pct}", "building"
    return st["status"], "building" if st["status"] == "queued" else "planned"


# --------------------------------------------------------------------------- #
# Process flow 1: build the index (offline)
# --------------------------------------------------------------------------- #
def _build_flow() -> dict:
    thr = _semantic_threshold()
    thr_txt = f"{thr:.2f}" if thr is not None else "calibrated at build"
    status = {s: _status_suffix(s) for s in registry.BUILT}

    def strat(sid: str, lines: str) -> str:
        suffix, cls = status[sid]
        return f'    {sid}["<b>{registry.STRATEGIES[sid][0]}</b> · {suffix}<br/>{lines}"]:::{cls}\n'

    mermaid = (
        "flowchart LR\n"
        '  ds[("<b>RAG-Mini-BioASQ</b><br/>40,221 passages<br/>4,719 questions")]:::data\n'
        '  nb01["<b>NB01 Dataset forensics</b><br/>question type · word-overlap<br/>gold passages per question"]:::step\n'
        '  bench[("<b>Benchmark freeze</b><br/>dev 300 · locked 100<br/>5 types × 4 overlap quartiles<br/>dev corpus 5,000 passages")]:::data\n'
        '  nb02["<b>NB02 Ingestion</b><br/>normalize · deduplicate<br/>drop 12,220 empty<br/>= 28,001 passages"]:::step\n'
        '  chunk{{"<b>Chunking</b><br/>one index per strategy"}}:::decision\n'
        '  c_whole["<b>Whole units</b>"]:::step\n'
        '  c_struct["<b>Structure-aware</b>"]:::step\n'
        '  c_window["<b>Fixed windows</b>"]:::step\n'
        '  passage["<b>Whole passage</b> · ready<br/>1 unit per passage"]:::ready\n'
        + strat("adaptive", "rules pick a chunker per passage<br/>see tab: Adaptive router")
        + '  agentic_note["<b>Agentic (LLM)</b> · at question time<br/>see tab: Agentic chunking"]:::llm\n'
        + '  sentvec["Sentence vectors · MedCPT<br/>254,620 sentences"]:::model\n'
        + strat("semantic", f"break where adjacent-sentence cosine<br/>is below {thr_txt} (p{S.SEMANTIC_PERCENTILE})"
                            f"<br/>or chunk over {S.SEMANTIC_MAX} words")
        + strat("recursive", f"paragraph → sentence → word<br/>≤ {S.RECURSIVE_MAX} words, merge under {S.RECURSIVE_MIN}")
        + strat("biomedical", f"pack sentences ≤ {S.BIOMEDICAL_MAX} words<br/>relation / negation sentence "
                              f"may stretch to {S.BIOMEDICAL_STRETCH}×")
        + strat("proposition", f"split clauses on and / but / while …<br/>≤ {S.PROPOSITION_MAX} words each")
        + strat("fixed", f"{S.FIXED_SIZE} words, {S.FIXED_OVERLAP} overlap")
        + strat("parent_child", f"search {S.CHILD_SIZE}-word children ({S.CHILD_OVERLAP} overlap)<br/>"
                                "LLM receives the parent passage")
        + strat("late", f"{S.LATE_SIZE}/{S.LATE_OVERLAP} spans, vectors pooled from<br/>"
                        "one contextual pass over the passage")
        + '  embed["<b>Embed</b><br/>MedCPT Article Encoder<br/>[CLS] · max 512 tokens<br/>768-d"]:::model\n'
        + f'  faiss[("<b>FAISS vector index</b><br/>exact inner product<br/>HNSW M={dense_mod.HNSW_M}, '
          f'efC={dense_mod.HNSW_EF_CONSTRUCTION}<br/>IVF nlist = {dense_mod.IVF_NLIST_FACTOR}·√n")]:::data\n'
        + f'  bm25[("<b>BM25 index</b><br/>k1={bm25_mod.K1}, b={bm25_mod.B}")]:::data\n'
        "  ds --> nb01 --> bench\n"
        "  nb01 --> nb02 --> chunk\n"
        "  chunk --> c_whole\n"
        "  chunk --> c_struct\n"
        "  chunk --> c_window\n"
        "  c_whole --> passage\n"
        "  c_whole --> adaptive\n"
        "  c_whole --> agentic_note\n"
        "  c_struct --> sentvec --> semantic\n"
        "  c_struct --> recursive\n"
        "  c_struct --> biomedical\n"
        "  c_struct --> proposition\n"
        "  c_window --> fixed\n"
        "  c_window --> parent_child\n"
        "  c_window --> late\n"
        "  passage & adaptive & semantic & recursive & biomedical & proposition --> embed\n"
        "  fixed & parent_child & late --> embed\n"
        "  embed --> faiss\n"
        "  embed --> bm25\n"
        '  kg[("<b>Evidence graph</b> · rules, CPU<br/>entities + same-sentence links<br/>'
        f'each link keeps its source passage")]:::data\n'
        f'  toc[("<b>PageIndex tree</b><br/>{tree_mod.TOP_NODES} topics → sections of ~{tree_mod.LEAF_SIZE}<br/>'
        'k-means on MedCPT passage vectors")]:::data\n'
        f'  syndict[("<b>Synonym dictionary</b><br/>abbreviation ↔ long form<br/>Schwartz–Hearst, from the corpus")]:::data\n'
        "  passage --> kg\n"
        "  passage --> syndict\n"
        "  faiss --> toc\n"
        + CLASSES
    )
    nodes = {
        "ds": _node("RAG-Mini-BioASQ", "BioASQ-derived RAG dataset: passages + questions with gold passage ids.",
                    ["40,221 passage rows", "4,719 questions (4,387 with usable gold)"], "Hugging Face"),
        "nb01": _node("Dataset forensics (NB01)", "Profiles every question for stratified sampling.",
                      ["inferred type: yes/no · factoid · list · summary · other",
                       "word-overlap quartiles", "gold passages per question: median 6"],
                      "Notebooks/BioRAG-X_01_dataset_forensics.ipynb"),
        "nb02": _node("Ingestion (NB02)", "Cleans text and assigns canonical ids with provenance.",
                      ["12,220 empty passages removed", "28,001 usable passages"],
                      "Notebooks/BioRAG-X_02_ingestion_and_provenance.ipynb"),
        "kg": _node("Evidence graph", "Built by rules from all passages in ~10 s (no LLM).",
                    ["entities: gene / protein symbols, miRNAs, drug and disease suffixes, NB04 patterns",
                     "link = two entities in one sentence, labelled by the sentence's relation word",
                     f"entities in > {graph_mod.GENERIC_SHARE:.0%} of passages marked generic"],
                    "scripts/build_graph_pageindex.py"),
        "syndict": _node("Synonym dictionary", "Abbreviations the corpus itself defines, built in ~1 s.",
                         ['"long form (SHORT)" found with the Schwartz–Hearst algorithm',
                          f"≤ {syn_mod.MAX_LONG_FORMS} long forms per short form, each ≥ "
                          f"{syn_mod.MIN_SHARE:.0%} of its definitions"],
                         "scripts/build_graph_pageindex.py"),
        "toc": _node("PageIndex tree", "A table of contents over the corpus, built in ~5 s.",
                     [f"{tree_mod.TOP_NODES} topics, then sections of ~{tree_mod.LEAF_SIZE} passages",
                      "grouped by k-means on the existing MedCPT passage vectors",
                      "each node summarised by the share of its passages containing each word"],
                     "scripts/build_graph_pageindex.py"),
        "bench": _node("Benchmark freeze", "Frozen question sets so experiments are comparable.",
                       ["dev 300 for tuning, locked 100 for the final check only",
                        "20 locked / 60 dev per inferred type, spread over overlap quartiles",
                        "duplicate questions kept on one side",
                        "dev corpus = 1,798 gold passages of dev 300 + 3,202 random distractors"],
                       "scripts/build_benchmark.py"),
        "passage": _node("Whole passage", "Each passage is one retrieval unit.",
                         ["passages are short: median ~160 words"], "ingestion/loader.py"),
        "sentvec": _node("Sentence vectors", "MedCPT [CLS] vectors for every sentence, used only to "
                         "find semantic boundaries.", [f"threshold = p{S.SEMANTIC_PERCENTILE} of all "
                         "adjacent-sentence cosines"], "scripts/build_chunk_indexes.py", ),
        "chunk": _node("Chunking", "Every strategy gets its own chunk table, vectors and indexes.",
                       ["pick one in the flow bar; Ask compares all ready ones"], "chunking/"),
        "embed": _node("Embedding", "Encodes every chunk with the MedCPT Article Encoder (local, GPU).",
                       ["[CLS] pooling, max 512 tokens, 768-d, inner product",
                        "late chunking: mean of contextual token vectors per span"],
                       "indexes/encoder.py"),
        "faiss": _node("FAISS vector index", "Dense search over one strategy's vectors.",
                       ["exact: every vector compared (reference)",
                        f"HNSW: M={dense_mod.HNSW_M}, efConstruction={dense_mod.HNSW_EF_CONSTRUCTION}, "
                        f"efSearch default {dense_mod.DEFAULT_EF_SEARCH}",
                        f"IVF: nlist = {dense_mod.IVF_NLIST_FACTOR}·√n, nprobe default {dense_mod.DEFAULT_NPROBE}"],
                       "indexes/dense.py"),
        "bm25": _node("BM25 sparse index", "Keyword search; exact Okapi BM25 as a sparse matrix product.",
                      [f"k1={bm25_mod.K1}, b={bm25_mod.B}, negative idf floored at "
                       f"{bm25_mod.EPSILON}× mean idf"], "indexes/bm25.py"),
        "agentic_note": _node("Agentic chunking", "Not built offline: the LLM segments the retrieved "
                              "passages at question time.", [], "chunking/agentic.py"),
    }
    what = {
        "recursive": [f"≤ {S.RECURSIVE_MAX} words; units under {S.RECURSIVE_MIN} merge with the previous"],
        "semantic": [f"break when adjacent-sentence cosine is below {thr_txt} "
                     f"(the weakest {S.SEMANTIC_PERCENTILE}% of transitions)",
                     f"or when a chunk would exceed {S.SEMANTIC_MAX} words; merge pieces under {S.SEMANTIC_MIN}"],
        "biomedical": [f"pack sentences up to {S.BIOMEDICAL_MAX} words",
                       f"a relation or negation sentence may stretch the chunk to {S.BIOMEDICAL_STRETCH}×"],
        "proposition": ["split on and, but, while, whereas, although, because, therefore, however",
                        f"pieces over {S.PROPOSITION_MAX} words are cut"],
        "fixed": [f"{S.FIXED_SIZE}-word windows with {S.FIXED_OVERLAP}-word overlap; ignores sentences"],
        "parent_child": [f"children: {S.CHILD_SIZE} words, {S.CHILD_OVERLAP} overlap",
                         "best child per parent kept; its parent passage goes to the LLM"],
        "late": [f"spans {S.LATE_SIZE}/{S.LATE_OVERLAP}", "one forward pass per passage; each span vector "
                 "is the mean of its contextual token vectors"],
        "adaptive": ["see the Adaptive router tab for the rules"],
    }
    for sid in registry.BUILT:
        label, desc, source = registry.STRATEGIES[sid]
        nodes[sid] = _node(label, desc, what.get(sid, []) + [f"index: {status[sid][0]}"],
                           "chunking/strategies.py", "live" if status[sid][1] == "ready" else "building")
    return {"id": "build", "title": "Build the index (offline)",
            "description": "Runs once. Turns the dataset into one BM25 + FAISS index per chunking strategy.",
            "mermaid": mermaid, "nodes": nodes}


# --------------------------------------------------------------------------- #
# Process flow 2: answer a question (online)
# --------------------------------------------------------------------------- #
def _answer_flow() -> dict:
    d = RetrievalConfig()
    n_ready = sum(registry.is_ready(s) for s in registry.STRATEGIES if s != "agentic")
    sup, par = cv._SUPPORTED, cv._PARTIAL
    mermaid = (
        "flowchart TB\n"
        '  q(["<b>Your question</b>"]):::output\n'
        '  flow["<b>Your flow</b> (top bar)<br/>chunking · search · RRF · rerank · evidence"]:::step\n'
        f'  cand["<b>Candidate combinations</b><br/>your flow<br/>+ {n_ready} ready chunkings × standard search<br/>'
        '+ BM25 / dense / hybrid / hybrid + cross-encoder<br/>+ graph / PageIndex, alone and with hybrid'
        '<br/>+ synonyms · agentic router'
        '<br/>duplicates removed"]:::step\n'
        "  q --> cand\n"
        "  flow --> cand\n"
        "  cand --> isag\n"
        '  subgraph each["For each combination: retrieval only, no LLM"]\n'
        '    isag{"Chunking is<br/>agentic?"}:::decision\n'
        '    agp["<b>Agentic chunking</b><br/>see tab: Agentic chunking"]:::llm\n'
        f'    search["<b>Search</b><br/>BM25 top {d.depth} + dense top {d.depth}<br/>'
        f'MedCPT query encoder, max {dense_mod.QUERY_MAX_LENGTH} tokens"]:::model\n'
        f'    fuse["<b>RRF fusion</b><br/>score = Σ 1 / ({d.rrf_k} + rank)"]:::step\n'
        '    isrr{"Reranker?"}:::decision\n'
        f'    ce["<b>MedCPT cross-encoder</b><br/>re-scores top {cmp_mod.COMPARE_RERANK_CANDIDATES}–{d.rerank_candidates}"]:::model\n'
        '    ispc{"Parent-child?"}:::decision\n'
        '    pcx["Swap each child for its parent passage"]:::step\n'
        f'    ev["<b>Evidence</b><br/>MMR λ={es.MMR_LAMBDA} → top {d.evidence_k}"]:::step\n'
        "    isag -->|yes| agp\n"
        '    qmode{"Query<br/>handling?"}:::decision\n'
        f'    syn["<b>Synonym expansion</b><br/>corpus abbreviations ↔ long forms<br/>'
        f'≤ {syn_mod.MAX_ADDED} added · keyword channels only"]:::step\n'
        '    rt["<b>Agentic router</b><br/>rules pick channels + synonyms<br/>'
        f'retry once if coverage < {router_mod.WEAK_COVERAGE}<br/>see tab: Retrieval router"]:::step\n'
        '    chans["Selected search channels"]:::step\n'
        "    isag -->|no| qmode\n"
        "    qmode -->|as typed| chans\n"
        "    qmode -->|synonyms| syn --> chans\n"
        "    qmode -->|router| rt --> chans\n"
        "    chans --> search\n"
        f'    graphrag["<b>Graph traversal</b> (optional channel)<br/>question entities → 1 hop, '
        f'top {graph_mod.MAX_NEIGHBORS} neighbours<br/>hop decay {graph_mod.HOP_DECAY} · bridge ×{graph_mod.BRIDGE_BOOST}"]:::step\n'
        f'    pidx["<b>PageIndex</b> (optional channel)<br/>open {tree_mod.TOP_BEAM} best topics → '
        f'read {tree_mod.MAX_LEAVES} best sections<br/>no vectors · BM25 inside sections"]:::step\n'
        "    chans --> graphrag\n"
        "    chans --> pidx\n"
        "    agp --> ev\n"
        "    search --> fuse --> isrr\n"
        "    graphrag --> fuse\n"
        "    pidx --> fuse\n"
        "    isrr -->|none| ispc\n"
        "    isrr -->|cross-encoder| ce --> ispc\n"
        "    ispc -->|yes| pcx --> ev\n"
        "    ispc -->|no| ev\n"
        "  end\n"
        '  judge["<b>Judge</b>: MedCPT cross-encoder<br/>relevance = σ(logit) per passage<br/>'
        'evidence score = mean over the evidence"]:::model\n'
        "  ev --> judge\n"
        '  pick{"<b>Pick best</b><br/>highest evidence score<br/>not self-judged · ties → faster"}:::decision\n'
        '  gold{"Question in<br/>the dataset?"}:::decision\n'
        '  truth["<b>True scores</b> vs gold passages<br/>nDCG@10 · Hit@5 · evidence recall"]:::data\n'
        "  judge --> pick\n"
        "  judge --> gold\n"
        "  gold -->|yes| truth\n"
        '  mode{"Answer with?"}:::decision\n'
        "  pick --> mode\n"
        f'  gen["<b>GPT-4o</b> · 1 call · temperature 0<br/>JSON: answer + claims + citations<br/>'
        f'budget {SETTINGS.llm_call_budget} calls/process · disk cache"]:::llm\n'
        "  mode -->|best combination| gen\n"
        "  mode -->|my flow| gen\n"
        '  parse{"Usable<br/>JSON?"}:::decision\n'
        '  ext["Extractive fallback<br/>best sentences, each cited"]:::step\n'
        '  ins["No claims → evidence insufficient<br/>the model\'s answer is shown"]:::step\n'
        f'  cite["<b>Citation check</b> per claim<br/>cited id not retrieved → dropped<br/>'
        f'word overlap ≥ {sup} SUPPORTED · ≥ {par} PARTIAL"]:::step\n'
        f'  status["<b>Evidence status</b><br/>model\'s label, else:<br/>grounding ≥ {rm.STATUS_SUFFICIENT_GROUNDING}'
        f' and coverage ≥ {rm.STATUS_SUFFICIENT_COVERAGE} → sufficient<br/>grounding ≥ {rm.STATUS_WEAK_GROUNDING} → weak'
        ' · else insufficient"]:::decision\n'
        '  report(["<b>Trust report</b><br/>answer · combinations · evidence · citations"]):::output\n'
        "  gen --> parse\n"
        "  parse -->|no answer| ext --> cite\n"
        "  parse -->|no claims| ins --> status\n"
        "  parse -->|yes| cite\n"
        "  cite --> status --> report\n"
        "  truth --> report\n"
        '  subgraph plannedzone["Planned additions (research design)"]\n'
        '    qe["LLM query rewriting<br/>HyDE · Query2Doc · decomposition"]:::planned\n'
        '    abstain["Abstain on unsupported claims"]:::planned\n'
        "  end\n"
        "  report ~~~ plannedzone\n"
        + CLASSES
    )
    nodes = {
        "q": _node("Your question", "Typed on the Ask page, or passed as a ?q= link."),
        "flow": _node("Your flow", "The choices in the top bar; always one of the candidates.",
                      ["Answer with my flow uses exactly these settings"], "frontend ControlBar"),
        "cand": _node("Candidate combinations", "Everything compared for this question.",
                      ["your flow", f"each ready chunking with standard search (BM25 + dense, RRF k={d.rrf_k}, "
                       f"no rerank, MMR top {d.evidence_k})",
                       "on your chunking: BM25 only, dense only, hybrid, hybrid + MedCPT cross-encoder "
                       f"({cmp_mod.COMPARE_RERANK_CANDIDATES} candidates)",
                       "graph only, PageIndex only, hybrid + graph, hybrid + PageIndex",
                       "synonym expansion, agentic router",
                       "agentic chunking only when it is your own choice (LLM cost)",
                       "duplicates: configs that behave identically are merged"],
                      "retrieval/compare.py"),
        "isag": _node("Agentic chunking?", "Agentic chunks are made at question time.",
                      [f"yes: retrieve top {ragentic.N_PASSAGES} passages, chunk them with the LLM"],
                      "retrieval/hybrid.py"),
        "agp": _node("Agentic chunking", "One batched GPT-4o call proposes sentence boundaries.",
                     ["see the Agentic chunking tab"], "chunking/agentic.py"),
        "search": _node("Search", "Keyword and meaning search run side by side.",
                        [f"BM25 top {d.depth}", f"dense top {d.depth} (MedCPT query encoder, "
                         f"max {dense_mod.QUERY_MAX_LENGTH} tokens, exact or ANN)"],
                        "retrieval/hybrid.py"),
        "fuse": _node("RRF fusion", "Merges ranked lists using ranks only (scores are on different scales).",
                      [f"score = Σ 1 / ({d.rrf_k} + rank)"], "retrieval/fusion.py"),
        "isrr": _node("Reranker?", "Optional second, slower pass over the top candidates.",
                      ["none (default) · MedCPT cross-encoder · word-overlap proxies · GPT-4o (Chat only)"]),
        "ce": _node("MedCPT cross-encoder", "Reads question and passage together.",
                    [f"{cmp_mod.COMPARE_RERANK_CANDIDATES} candidates in the comparison, "
                     f"{d.rerank_candidates} by default", "~1.3 s per 20 candidates on M1",
                     "same model as the judge → its rows are 'self-judged'"],
                    "retrieval/reranker.py"),
        "ispc": _node("Parent-child?", "Search children, answer from parents.", [], "retrieval/hybrid.py"),
        "pcx": _node("Parent expansion", "Each child is replaced by its whole parent passage "
                     "(deduplicated) before evidence selection.", [], "retrieval/hybrid.py"),
        "ev": _node("Evidence selection", "Picks the passages sent to the LLM.",
                    [f"MMR: {es.MMR_LAMBDA} × relevance − {round(1 - es.MMR_LAMBDA, 2)} × overlap with "
                     "already-picked evidence", f"top {d.evidence_k} from the best 10"],
                    "retrieval/evidence_selector.py"),
        "judge": _node("Judge", "Label-free score for questions without gold passages.",
                       ["relevance = sigmoid(cross-encoder logit), 0..1",
                        "evidence score = mean relevance of the evidence set",
                        "cached per (question, text)"], "retrieval/judge.py"),
        "pick": _node("Pick best", "Chooses the combination used for the answer.",
                      ["highest evidence score", "self-judged rows (reranked by the judge's own model) "
                       "cannot win", "ties → the faster combination"], "retrieval/compare.py"),
        "gold": _node("Dataset question?", "Exact (normalised) text match against the 4,387 "
                      "questions with gold passages.", [], "retrieval/compare.py"),
        "truth": _node("True scores", "Checks each combination against the gold passages.",
                       ["nDCG@10 and Hit@5 on the ranked passages", "evidence recall of the 5 sent"],
                       "evaluation/retrieval_metrics.py"),
        "mode": _node("Answer with?", "Best combination (default) or exactly your flow.", [],
                      "frontend ChatPage"),
        "gen": _node("GPT-4o", "Writes the answer only from the evidence, citing a passage per claim.",
                     ["1 LLM call per question", f"process budget: {SETTINGS.llm_call_budget} live calls",
                      "identical prompts served from the disk cache", "temperature 0, max 700 tokens"],
                     "generation/generator.py"),
        "parse": _node("Usable JSON?", "Repair-tolerant parsing of the model output.",
                       ["no answer → extractive fallback", "answer but no claims → marked insufficient"],
                       "generation/structured_output.py"),
        "ext": _node("Extractive fallback", "Best-matching sentences of the top passages, each cited.",
                     ["used when the LLM is off, over budget, or returns nothing usable"],
                     "generation/generator.py"),
        "ins": _node("No claims", "The model found nothing citable; its answer is shown, marked insufficient."),
        "cite": _node("Citation check", "Verifies every citation of every claim.",
                      ["cited id not among the retrieved evidence → dropped",
                       f"claim words found in the cited passage ≥ {sup} → SUPPORTED",
                       f"≥ {par} → PARTIALLY_SUPPORTED, else UNSUPPORTED"],
                      "generation/citation_validator.py"),
        "status": _node("Evidence status", "Sufficient / weak / insufficient.",
                        ["the model's own label when valid, otherwise:",
                         f"grounding ≥ {rm.STATUS_SUFFICIENT_GROUNDING} and coverage ≥ "
                         f"{rm.STATUS_SUFFICIENT_COVERAGE} → sufficient",
                         f"grounding ≥ {rm.STATUS_WEAK_GROUNDING} → weak, else insufficient"],
                        "evaluation/rag_metrics.py"),
        "report": _node("Trust report", "What the Ask page shows.",
                        ["answer and how it was built", "every combination compared, with scores",
                         "what was extracted", "claims and citation checks"], "frontend ChatPage"),
        "qe": _node("LLM query rewriting", "HyDE, Query2Doc and decomposition: each adds an LLM "
                    "call per question.", [], status="planned"),
        "qmode": _node("Query handling", "How the question is turned into searches.",
                       ["as typed", "synonym expansion", "agentic router"], "retrieval/hybrid.py"),
        "syn": _node("Synonym expansion", "Abbreviations defined in the corpus, matched both ways.",
                     ["short form typed (exact case) → add its long forms",
                      "long form typed → add its short form",
                      f"at most {syn_mod.MAX_ADDED} additions; dense search and reranker keep the "
                      "question as typed"], "retrieval/synonyms.py"),
        "rt": _node("Agentic router", "Chooses the route per question; see the Retrieval router tab.",
                    [f"retry once if key-term coverage < {router_mod.WEAK_COVERAGE}",
                     f"max {router_mod.MAX_ROUNDS} rounds; no LLM"], "retrieval/router.py"),
        "chans": _node("Search channels", "Your ticked channels, or the router's choice.", []),
        "graphrag": _node("Graph traversal", "Search channel: follows entity links in a rule-built "
                          "evidence graph; each passage keeps the path that found it.",
                          ["seeds = question entities found in the graph",
                           "passage score = Σ idf(seed) + pair bonus when two seeds share a sentence "
                           f"(× {graph_mod.TYPED_WEIGHT} with a relation word)",
                           f"hop 1: {graph_mod.MAX_NEIGHBORS} strongest neighbours add "
                           f"{graph_mod.HOP_DECAY} × idf; × {graph_mod.BRIDGE_BOOST} if linked to 2+ seeds",
                           f"entities in > {graph_mod.GENERIC_SHARE:.0%} of passages are not traversed",
                           "no LLM, no vectors"], "graph/evidence_graph.py"),
        "pidx": _node("PageIndex", "Search channel: navigates a topic table of contents, no vector "
                      "search at question time.",
                      ["node score = Σ idf(word) × coverage(node, word)^"
                       f"{tree_mod.NAV_POWER}", f"open the {tree_mod.TOP_BEAM} best topics, read the "
                       f"{tree_mod.MAX_LEAVES} best sections (more if too few passages)",
                       "rank the passages read by BM25", "LLM navigation off (no LLM calls)"],
                      "pageindex/tree.py"),
        "abstain": _node("Abstention", "Refuse instead of answering when claims stay unsupported.", [],
                         status="planned"),
    }
    return {"id": "answer", "title": "Answer a question (online)",
            "description": "Runs for every question: compare combinations, pick the best, answer once, "
                           "check every citation.",
            "mermaid": mermaid, "nodes": nodes}


# --------------------------------------------------------------------------- #
# Process flow 3 + 4: decision trees
# --------------------------------------------------------------------------- #
def _router_flow() -> dict:
    manifest = paths.CHUNK_INDEX_DIR / "adaptive" / "manifest.json"
    share = {}
    if manifest.exists():
        routed = json.loads(manifest.read_text()).get("routed_to", {})
        total = sum(routed.values()) or 1
        share = {k: f"<br/>{v / total:.0%} of passages" for k, v in routed.items()}
    mermaid = (
        "flowchart TB\n"
        '  p["<b>Passage profile</b><br/>words · entity / relation / negation density<br/>'
        'lexical shift between sentences"]:::data\n'
        f'  r1{{"words ≤ {S.ROUTE_SHORT_TOKENS}?"}}:::decision\n'
        f'  r2{{"relation density ≥ {S.ROUTE_RELATION_DENSITY}<br/>and any negation?"}}:::decision\n'
        f'  r3{{"entity density ≥ {S.ROUTE_ENTITY_DENSITY}?"}}:::decision\n'
        f'  r4{{"lexical shift ≥ {S.ROUTE_TOPIC_SHIFT}?"}}:::decision\n'
        f'  r5{{"words ≥ {S.ROUTE_LONG_TOKENS}?"}}:::decision\n'
        f'  o1["<b>Recursive</b><br/>short passage{share.get("recursive", "")}"]:::ready\n'
        f'  o2["<b>Biomedical-aware</b><br/>keep relations with their negations{share.get("biomedical", "")}"]:::ready\n'
        '  o3["<b>Biomedical-aware</b><br/>entity-dense text"]:::ready\n'
        f'  o4["<b>Semantic</b><br/>topic changes inside the passage{share.get("semantic", "")}"]:::ready\n'
        f'  o5["<b>Parent-child</b><br/>long passage{share.get("parent_child", "")}"]:::ready\n'
        '  o6["<b>Recursive</b><br/>default"]:::ready\n'
        "  p --> r1\n"
        "  r1 -->|yes| o1\n"
        "  r1 -->|no| r2\n"
        "  r2 -->|yes| o2\n"
        "  r2 -->|no| r3\n"
        "  r3 -->|yes| o3\n"
        "  r3 -->|no| r4\n"
        "  r4 -->|yes| o4\n"
        "  r4 -->|no| r5\n"
        "  r5 -->|yes| o5\n"
        "  r5 -->|no| o6\n"
        + CLASSES
    )
    rules = {"r1": f"words ≤ {S.ROUTE_SHORT_TOKENS} → Recursive",
             "r2": f"relation-trigger density ≥ {S.ROUTE_RELATION_DENSITY} and a negation cue → Biomedical",
             "r3": f"biomedical entity density ≥ {S.ROUTE_ENTITY_DENSITY} → Biomedical",
             "r4": f"mean lexical shift between adjacent sentences ≥ {S.ROUTE_TOPIC_SHIFT} → Semantic",
             "r5": f"words ≥ {S.ROUTE_LONG_TOKENS} → Parent-child"}
    nodes = {k: _node("Router rule", v, [v, "rules are checked top to bottom; first match wins"],
                      "chunking/strategies.py") for k, v in rules.items()}
    nodes["p"] = _node("Passage profile", "Cheap text statistics computed per passage (no model).",
                       ["densities = matches per word", "lexical shift = 1 − word overlap of adjacent sentences"],
                       "chunking/strategies.py")
    return {"id": "router", "title": "Adaptive chunking router",
            "description": "How the adaptive strategy picks a chunker for each passage (NB05 rules, no LLM).",
            "mermaid": mermaid, "nodes": nodes}


def _retrieval_router_flow() -> dict:
    w = router_mod.WEAK_COVERAGE
    mermaid = (
        "flowchart TB\n"
        '  q(["Question"]):::output\n'
        '  feat["<b>Question features</b> (no model)<br/>type: yes/no · factoid · list · summary<br/>'
        'graph entities · relation word · known abbreviations"]:::step\n'
        '  base["Always: BM25 + dense"]:::step\n'
        '  r1{"2+ entities, or<br/>1 entity + relation word?"}:::decision\n'
        '  r2{"Known abbreviation<br/>or long form?"}:::decision\n'
        '  r3{"List / summary question<br/>with no entity?"}:::decision\n'
        '  addg["+ Graph traversal"]:::step\n'
        '  adds["Synonym expansion on"]:::step\n'
        '  addp["+ PageIndex"]:::step\n'
        '  round1["<b>Round 1</b> retrieval<br/>fuse · rerank · evidence as your flow"]:::model\n'
        f'  cov{{"Key-term coverage<br/>≥ {w}?"}}:::decision\n'
        '  round2["<b>Round 2</b>: every ready channel<br/>BM25 + dense + graph + PageIndex<br/>+ synonyms"]:::model\n'
        '  better{"Round 2 covers<br/>more key terms?"}:::decision\n'
        '  keep1(["Use round 1"]):::output\n'
        '  keep2(["Use round 2"]):::output\n'
        "  q --> feat\n"
        "  feat --> base --> round1\n"
        "  feat --> r1 -->|yes| addg --> round1\n"
        "  feat --> r2 -->|yes| adds --> round1\n"
        "  feat --> r3 -->|yes| addp --> round1\n"
        "  round1 --> cov\n"
        "  cov -->|yes| keep1\n"
        "  cov -->|no| round2 --> better\n"
        "  better -->|yes| keep2\n"
        "  better -->|no| keep1\n"
        + CLASSES
    )
    nodes = {
        "feat": _node("Question features", "Computed by rules in milliseconds.",
                      ["type from the first words (list / describe / is …)",
                       "entities: question terms found in the evidence graph",
                       "relation word: inhibit, target, bind, regulate, …",
                       "abbreviations: terms in the corpus synonym dictionary"], "retrieval/router.py"),
        "r1": _node("Rule R1", "Entity questions follow entity links.",
                    ["2+ graph entities, or 1 entity + a relation word → add graph traversal"]),
        "r2": _node("Rule R2", "Abbreviations are expanded.",
                    ["a known short or long form → synonym expansion"]),
        "r3": _node("Rule R3", "Broad questions browse the topic tree.",
                    ["list / summary question with no graph entity → add PageIndex"]),
        "cov": _node("Evidence check (CRAG-style)", "Does the evidence mention what was asked?",
                     ["key terms = question content words", f"matched on the first {router_mod.STEM} letters",
                      f"coverage < {w} → weak → round 2",
                      "not the MedCPT judge, so the Ask page's judge stays independent"],
                     "retrieval/router.py"),
        "round2": _node("Round 2", "The widest route, tried once.",
                        [f"max {router_mod.MAX_ROUNDS} rounds", "skipped with agentic chunking "
                         "(would repeat the LLM call)"], "retrieval/router.py"),
        "better": _node("Keep the better round", "Higher key-term coverage wins; ties keep round 1.", []),
    }
    return {"id": "retrieval_router", "title": "Retrieval router",
            "description": "How the agentic router picks search channels per question and retries "
                           "once when the evidence is weak (rules, no LLM).",
            "mermaid": mermaid, "nodes": nodes}


def _agentic_flow() -> dict:
    mermaid = (
        "flowchart TB\n"
        '  q(["Question"]):::output\n'
        f'  base["<b>Retrieve passages</b><br/>your search settings on whole passages<br/>'
        f'top {ragentic.N_PASSAGES}"]:::step\n'
        '  one{"Passage has<br/>2+ sentences?"}:::decision\n'
        '  single["Keep as one chunk"]:::step\n'
        '  cached{"Plan cached<br/>for this passage?"}:::decision\n'
        '  reuse["Reuse the cached plan<br/>no LLM call"]:::ready\n'
        f'  llmcall["<b>1 batched GPT-4o call</b> for all uncached passages<br/>numbered sentences in<br/>'
        f'boundaries out · chunks ≤ {agentic_mod.MAX_TOKENS} words<br/>keep entities, relations, negation"]:::llm\n'
        '  llmok{"LLM available<br/>and within budget?"}:::decision\n'
        '  valid{"Plan valid?<br/>integers · sorted · unique<br/>within 1 … sentences−1"}:::decision\n'
        '  apply["Apply boundaries<br/>cache the plan per passage"]:::step\n'
        '  fb["<b>Fallback</b>: recursive chunking<br/>invalid plans cached too"]:::step\n'
        '  fbn["<b>Fallback</b>: recursive chunking<br/>not cached · retried next time"]:::step\n'
        '  rank["<b>Rank the chunks</b><br/>MedCPT dense (chunks encoded on the fly)<br/>+ BM25 → RRF"]:::model\n'
        '  out(["Rerank + evidence selection<br/>as in the normal pipeline"]):::output\n'
        "  q --> base --> one\n"
        "  one -->|no| single --> rank\n"
        "  one -->|yes| cached\n"
        "  cached -->|yes| reuse --> rank\n"
        "  cached -->|no| llmok\n"
        "  llmok -->|no| fbn --> rank\n"
        "  llmok -->|yes| llmcall --> valid\n"
        "  valid -->|yes| apply --> rank\n"
        "  valid -->|no| fb --> rank\n"
        "  rank --> out\n"
        + CLASSES
    )
    nodes = {
        "base": _node("Retrieve passages", "Finds which passages to segment for this question.",
                      [f"top {ragentic.N_PASSAGES} whole passages with your search settings"],
                      "retrieval/agentic.py"),
        "llmcall": _node("Batched LLM call", "NB05's LLM-guided chunking, for several passages at once.",
                      ["+1 LLM call per question at most", "numbered sentences in, boundary indexes out",
                       f"prefer chunks ≤ {agentic_mod.MAX_TOKENS} words",
                       "keep entities, relation statements and negations together"],
                      "chunking/agentic.py"),
        "valid": _node("Plan validation", "NB05 validate_llm_plan.",
                       ["integers only", "sorted and unique", "within 1 … number of sentences − 1"],
                       "chunking/agentic.py"),
        "cached": _node("Plan cache", "A passage is segmented by the LLM at most once, whatever the question.",
                        ["data/cache/agentic/<passage id>.json"], "chunking/agentic.py"),
        "rank": _node("Rank the chunks", "Scores the new chunks for this question.",
                      ["MedCPT Article Encoder vectors (cached) · inner product with the query",
                       "BM25 over the same chunks", "fused with RRF"], "retrieval/agentic.py"),
    }
    return {"id": "agentic", "title": "Agentic chunking",
            "description": "How the LLM chunks retrieved passages at question time, with validation, "
                           "fallbacks and caching.",
            "mermaid": mermaid, "nodes": nodes}


# --------------------------------------------------------------------------- #
# System design
# --------------------------------------------------------------------------- #
def _system_architecture() -> dict:
    ready = [s for s in registry.BUILT if registry.is_ready(s)]
    mermaid = (
        "flowchart TB\n"
        '  subgraph client["1 · Frontend · React + TypeScript"]\n'
        '    fe_flow["Pipeline flow bar<br/>every choice"]:::step\n'
        '    fe_ask["Ask<br/>trust report"]:::output\n'
        '    fe_info["Process flow<br/>System design"]:::step\n'
        '    fe_research["Research tools<br/>Lab · Runs"]:::step\n'
        "  end\n"
        '  subgraph api["2 · API · FastAPI"]\n'
        '    ep_flow["/capabilities<br/>/architecture"]:::step\n'
        '    ep_ask["/ask/compare<br/>/ask/answer"]:::step\n'
        '    ep_misc["/overview · /system<br/>/health"]:::step\n'
        '    ep_lab["/lab/info · /runs<br/>/runs/compare"]:::step\n'
        "  end\n"
        '  subgraph services["3 · Services · Python"]\n'
        '    sv_chunk["Chunking<br/>10 strategies"]:::step\n'
        '    sv_compare["Comparison<br/>+ judge"]:::step\n'
        '    sv_gen["Generation<br/>+ citation check"]:::step\n'
        '    sv_lab["Experiment lab<br/>1 worker"]:::step\n'
        '    sv_retr["Retrieval engine<br/>BM25 · FAISS · RRF · rerank · MMR"]:::step\n'
        '    sv_guard{{"LLM guard<br/>switch · budget · cache"}}:::decision\n'
        "  end\n"
        '  subgraph models["4 · Models"]\n'
        '    m_q["MedCPT<br/>query encoder"]:::model\n'
        '    m_a["MedCPT<br/>article encoder"]:::model\n'
        '    m_ce["MedCPT cross-encoder<br/>reranker + judge"]:::model\n'
        '    x_gpt["Azure GPT-4o<br/>answers · agentic chunking"]:::llm\n'
        '    x_ada["Azure ada-002<br/>NB04 chunks only"]:::llm\n'
        "  end\n"
        '  subgraph storage["5 · Storage · local disk"]\n'
        '    st_canon[("Canonical data<br/>28,001 passages")]:::data\n'
        f'    st_idx[("Indexes<br/>passage + {len(ready)}/{len(registry.BUILT)} chunkings")]:::data\n'
        '    st_bench[("Benchmark<br/>dev 300 · locked 100")]:::data\n'
        '    st_runs[("Run store<br/>SQLite")]:::data\n'
        '    st_cache[("Caches<br/>LLM · plans")]:::data\n'
        "  end\n"
        '  subgraph offline["6 · Offline jobs"]\n'
        '    j_nb["Notebooks<br/>NB01–NB05"]:::step\n'
        '    j_build["Index builds<br/>chunk + embed"]:::step\n'
        '    j_bench["Benchmark +<br/>judge calibration"]:::step\n'
        "  end\n"
        "  fe_flow --> ep_flow\n"
        "  fe_ask -->|HTTP JSON| ep_ask\n"
        "  fe_info --> ep_misc\n"
        "  fe_research --> ep_lab\n"
        "  ep_flow --> sv_chunk\n"
        "  ep_ask --> sv_compare\n"
        "  ep_ask --> sv_gen\n"
        "  ep_lab --> sv_lab\n"
        "  sv_compare --> sv_retr\n"
        "  sv_lab --> sv_retr\n"
        "  sv_gen --> sv_guard\n"
        "  sv_retr --> m_q\n"
        "  sv_compare --> m_ce\n"
        "  sv_chunk --> m_a\n"
        "  sv_guard -.->|if enabled, within budget| x_gpt\n"
        "  sv_retr -.-> x_ada\n"
        "  models ~~~ storage\n"
        "  sv_retr --> st_idx\n"
        "  sv_lab --> st_runs\n"
        "  st_canon <-.- j_nb\n"
        "  st_idx <-.- j_build\n"
        "  st_bench <-.- j_bench\n"
        + CLASSES
    )
    n = _node
    nodes = {
        "fe_flow": n("Pipeline flow bar", "Every choice of the pipeline, in order, with live build status.",
                     ["unbuilt options shown as planned / building, never as working"],
                     "frontend/src/components/ControlBar.tsx"),
        "fe_ask": n("Ask page", "Per-question trust report.", ["compare → pick best → answer once"],
                    "frontend/src/pages/ChatPage.tsx"),
        "fe_research": n("Research tools", "Batch experiments on frozen question sets.",
                         ["Retrieval Lab: run a flow on dev 300", "Runs & Compare: paired comparisons with CIs"],
                         "frontend/src/pages/LabPage.tsx, RunsPage.tsx"),
        "ep_ask": n("Ask endpoints", "/ask/compare runs the comparison (no LLM); /ask/answer answers once.",
                    [], "api/app.py"),
        "ep_flow": n("Flow endpoints", "/capabilities: options + build status; /architecture: these diagrams.",
                     [], "api/capabilities.py, api/architecture.py"),
        "ep_lab": n("Lab endpoints", "Queue runs, list them, compare two.", [], "api/app.py"),
        "sv_compare": n("Comparison + judge", "Builds candidates, runs each, judges evidence, picks best.",
                        ["self-judged rows cannot win", "one failing combination does not stop the report"],
                        "retrieval/compare.py, retrieval/judge.py"),
        "sv_retr": n("Retrieval engine", "BM25 + dense (FAISS) + RRF + rerank + MMR evidence.",
                     ["FAISS search parameters are per call (thread-safe)",
                      "indexes cached in memory after first use; warmed in the background"],
                     "retrieval/hybrid.py, indexes/"),
        "sv_chunk": n("Chunking", "Chunk tables per strategy; agentic chunks at question time.",
                      ["9 built strategies + agentic; registry reports build status"], "chunking/"),
        "sv_gen": n("Generation", "Strict grounded prompt, repair-tolerant JSON parsing, citation check.",
                    [], "generation/"),
        "sv_lab": n("Experiment lab", "Runs flows on frozen sets; paired bootstrap comparisons.",
                    ["one run at a time", "runs interrupted by a restart are marked failed"],
                    "experiments/"),
        "sv_guard": n("LLM guard", "The only path to Azure.",
                      [f"master switch LLM_ENABLED={SETTINGS.llm_enabled}",
                       f"budget {SETTINGS.llm_call_budget} live calls per process",
                       "sha256-keyed disk cache: repeats are free"], "common/llm.py"),
        "m_q": n("MedCPT Query Encoder", "Question → 768-d vector.",
                 [f"max {dense_mod.QUERY_MAX_LENGTH} tokens", "CPU", "cached per question"], "indexes/dense.py"),
        "m_a": n("MedCPT Article Encoder", "Text → 768-d vector (offline builds, agentic chunks).",
                 ["max 512 tokens", "GPU (MPS)"], "indexes/encoder.py"),
        "m_ce": n("MedCPT Cross-Encoder", "Relevance of (question, passage) pairs.",
                  ["reranker and judge", "GPU (MPS), one call at a time (lock)"], "retrieval/reranker.py"),
        "x_gpt": n("Azure GPT-4o", "Answers, agentic chunking, optional LLM reranking.",
                   ["called only through the LLM guard"]),
        "x_ada": n("Azure ada-002", "Query embeddings for NB04's semantic chunks only.", []),
        "st_idx": n("Indexes", "One folder per strategy: chunks.parquet, embeddings.npy, manifest.json.",
                    [f"ready: whole passage + {', '.join(ready) or 'none yet'}"],
                    "data/processed/indexes/"),
        "st_bench": n("Benchmark", "dev_300, locked_100, dev corpus ids; refuses to be overwritten.",
                      [], "data/benchmark/"),
        "st_runs": n("Run store", "One SQLite row per run + per-question parquet.", [], "data/runs/"),
        "st_cache": n("Caches", "LLM responses, query embeddings, agentic plans.", [], "data/cache/"),
        "j_build": n("Index builds", "Chunk + embed + save; resumable shards; live status file.",
                     [], "scripts/build_chunk_indexes.py"),
        "j_bench": n("Benchmark + calibration", "Freeze question sets; measure judge-vs-gold agreement.",
                     [], "scripts/build_benchmark.py, scripts/calibrate_judge.py"),
    }
    return {"id": "architecture", "title": "Architecture",
            "description": "Components, what calls what, and where data lives.",
            "mermaid": mermaid, "nodes": nodes}


def _system_reliability() -> dict:
    mermaid = (
        "flowchart LR\n"
        '  subgraph fails["If this fails"]\n'
        '    f1["LLM off / over budget / API error"]:::decision\n'
        '    f2["LLM returns no usable answer"]:::decision\n'
        '    f3["Agentic plan invalid"]:::decision\n'
        '    f4["Chunking index not built yet"]:::decision\n'
        '    f5["One combination errors"]:::decision\n'
        '    f6["Query embedding unavailable (ada-002)"]:::decision\n'
        '    f7["Server restarts during a run"]:::decision\n'
        '    f8["Index build interrupted"]:::decision\n'
        '    f9["Backend unreachable"]:::decision\n'
        "  end\n"
        '  subgraph then["The system does this"]\n'
        '    a1["Extractive answer from the evidence, labelled LLM off"]:::step\n'
        '    a2["Extractive fallback; label says unusable"]:::step\n'
        '    a3["Recursive chunking for that passage"]:::step\n'
        '    a4["Option disabled in the flow; API returns 400 with the reason"]:::step\n'
        '    a5["Row shows the error; the rest of the report continues"]:::step\n'
        '    a6["Dense search returns nothing; BM25 still answers"]:::step\n'
        '    a7["Run marked failed with the reason; rerun"]:::step\n'
        '    a8["Resumes from saved shards; finished strategies skipped"]:::step\n'
        '    a9["Pages show a backend-offline badge; lab pages show the error"]:::step\n'
        "  end\n"
        "  f1 --> a1\n  f2 --> a2\n  f3 --> a3\n  f4 --> a4\n  f5 --> a5\n"
        "  f6 --> a6\n  f7 --> a7\n  f8 --> a8\n  f9 --> a9\n"
        + CLASSES
    )
    nodes = {
        "f1": _node("LLM unavailable", "Covers LLM_ENABLED=false, exhausted budget, or network errors.",
                    [f"budget: {SETTINGS.llm_call_budget} live calls per process"], "common/llm.py"),
        "f4": _node("Strategy not ready", "Validation checks the build status before running a flow.",
                    [], "experiments/config.py"),
        "f8": _node("Build interrupted", "Embeddings are checkpointed every 4,096 chunks.",
                    [], "indexes/encoder.py"),
    }
    return {"id": "reliability", "title": "Reliability",
            "description": "Every failure path and the fallback that keeps the system answering.",
            "mermaid": mermaid, "nodes": nodes}


def process_diagrams() -> list[dict]:
    return [_build_flow(), _answer_flow(), _retrieval_router_flow(), _router_flow(), _agentic_flow()]


def system_diagrams() -> list[dict]:
    return [_system_architecture(), _system_reliability()]
