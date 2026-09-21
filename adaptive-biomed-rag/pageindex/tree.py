"""PageIndex-style retrieval: navigate a table-of-contents tree instead of a vector search
(research design, "PageIndex" route; NB08).

PageIndex reasons over a document's section tree. Our corpus is 28k short abstracts
with no sections, so the tree is a topic table of contents over the corpus:

  root -> TOP_NODES topics -> leaf sections of about LEAF_SIZE passages -> passages

Build (offline, CPU, seconds): passages are grouped by k-means on their MedCPT
vectors (already computed). Each node gets a summary: the share of its passages that
contain each word (``coverage``) and a label of its most distinctive words.

Query (no vectors, no LLM):
  1. score every topic by sum over question words of idf(word) x coverage(topic, word)^0.25
     (the damping lets a rare question word count even where few passages use it; on
     dev_300 it puts 79% of gold passages inside the leaves read, vs 45% undamped)
  2. open the TOP_BEAM best topics, score their leaves the same way
  3. read the MAX_LEAVES best leaves (more, widening the beam, if they hold fewer
     than ``top_k`` passages) and rank their passages by BM25
Each passage keeps its tree path (topic label > leaf label) as the reason it was read.
LLM navigation over the node labels (PageIndex's original) is left off: no LLM calls.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

from indexes.bm25 import _tokenize

TOP_NODES = 40            # first-level topics
LEAF_SIZE = 80            # target passages per leaf section
TOP_BEAM = 5              # topics opened per question
NAV_POWER = 0.25          # coverage damping in node scores (tuned on dev_300)
MAX_LEAVES = 6            # leaf sections read per question (more if too few passages)
LABEL_WORDS = 4
SEED = 7


def _words(text: str) -> list[str]:
    return [t for t in _tokenize(text) if t not in ENGLISH_STOP_WORDS and not t.isdigit()]


def _kmeans(x: np.ndarray, k: int) -> np.ndarray:
    import faiss
    if k <= 1 or len(x) <= k:
        return np.zeros(len(x), dtype=np.int64)
    km = faiss.Kmeans(x.shape[1], k, niter=20, seed=SEED, spherical=True, verbose=False,
                      min_points_per_centroid=1)
    km.train(x)
    return km.index.search(x, 1)[1].ravel()


def _coverage(member: np.ndarray, n_nodes: int, binary: sp.csr_matrix) -> sp.csr_matrix:
    """nodes x words: share of each node's passages that contain the word."""
    a = sp.csr_matrix((np.ones(len(member), np.float32), (member, np.arange(len(member)))),
                      shape=(n_nodes, len(member)))
    counts = a @ binary
    sizes = np.asarray(a.sum(axis=1)).ravel()
    return sp.diags(1.0 / np.maximum(sizes, 1)).astype(np.float32) @ counts


def _labels(cov: sp.csr_matrix, idf: np.ndarray, vocab: np.ndarray) -> list[str]:
    out = []
    for i in range(cov.shape[0]):
        row = cov.getrow(i)
        w = row.data * idf[row.indices] * (row.data >= 0.1)
        top = row.indices[np.argsort(-w)[:LABEL_WORDS]]
        out.append(" · ".join(vocab[top]))
    return out


def build(ids: list[str], texts: list[str], vectors: np.ndarray, out_dir: Path,
          log: Callable[[str], None] = print) -> dict:
    x = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)
    x = np.ascontiguousarray(x, dtype=np.float32)
    top = _kmeans(x, TOP_NODES)
    leaf = np.empty(len(ids), dtype=np.int64)
    parents, next_leaf = [], 0
    for t in range(TOP_NODES):
        idx = np.where(top == t)[0]
        sub = _kmeans(x[idx], max(1, round(len(idx) / LEAF_SIZE)))
        for s in np.unique(sub):
            leaf[idx[sub == s]] = next_leaf
            parents.append(t)
            next_leaf += 1
    log(f"  pageindex: {TOP_NODES} topics, {next_leaf} leaf sections")

    vec = CountVectorizer(tokenizer=_words, lowercase=False, token_pattern=None,
                          binary=True, min_df=2, dtype=np.float32)
    binary = vec.fit_transform(texts).tocsr()
    vocab = vec.get_feature_names_out()
    df = np.bincount(binary.indices, minlength=len(vocab))
    idf = np.maximum(np.log((len(ids) - df + 0.5) / (df + 0.5)), 0).astype(np.float32)
    cov_top = _coverage(top, TOP_NODES, binary)
    cov_leaf = _coverage(leaf, next_leaf, binary)

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"passage_id": ids, "topic": top, "leaf": leaf}).to_parquet(
        out_dir / "tree.parquet", index=False)
    sizes_top = np.bincount(top, minlength=TOP_NODES)
    sizes_leaf = np.bincount(leaf, minlength=next_leaf)
    nodes = pd.DataFrame({
        "node": [f"T{t}" for t in range(TOP_NODES)] + [f"L{i}" for i in range(next_leaf)],
        "level": ["topic"] * TOP_NODES + ["leaf"] * next_leaf,
        "index": list(range(TOP_NODES)) + list(range(next_leaf)),
        "parent": [-1] * TOP_NODES + parents,
        "passages": list(sizes_top) + list(sizes_leaf),
        "label": _labels(cov_top, idf, vocab) + _labels(cov_leaf, idf, vocab),
    })
    nodes.to_parquet(out_dir / "nodes.parquet", index=False)
    sp.save_npz(out_dir / "coverage_topic.npz", cov_top.tocsc())
    sp.save_npz(out_dir / "coverage_leaf.npz", cov_leaf.tocsc())
    np.save(out_dir / "idf.npy", idf)
    (out_dir / "vocab.json").write_text(json.dumps(vocab.tolist()))
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "passages": len(ids), "topics": TOP_NODES, "leaves": int(next_leaf),
        "median_leaf_size": int(np.median(sizes_leaf)), "vocabulary": int(len(vocab)),
        "params": {"top_nodes": TOP_NODES, "leaf_size": LEAF_SIZE, "top_beam": TOP_BEAM,
                   "max_leaves": MAX_LEAVES, "nav_power": NAV_POWER, "clustered_on": "MedCPT passage vectors"},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


class PageIndexTree:
    def __init__(self, tree_dir: Path):
        self.manifest = json.loads((tree_dir / "manifest.json").read_text())
        tree = pd.read_parquet(tree_dir / "tree.parquet")
        nodes = pd.read_parquet(tree_dir / "nodes.parquet")
        self.topic_label = nodes[nodes["level"] == "topic"].set_index("index")["label"].to_dict()
        leaves = nodes[nodes["level"] == "leaf"].set_index("index")
        self.leaf_label = leaves["label"].to_dict()
        self.leaf_parent = leaves["parent"].to_dict()
        self.children = leaves.groupby("parent").groups
        self.leaf_passages = tree.groupby("leaf")["passage_id"].agg(list).to_dict()
        self.cov_top = sp.load_npz(tree_dir / "coverage_topic.npz").tocsc()
        self.cov_leaf = sp.load_npz(tree_dir / "coverage_leaf.npz").tocsc()
        self.idf = np.load(tree_dir / "idf.npy")
        self.vocab = {w: i for i, w in enumerate(json.loads((tree_dir / "vocab.json").read_text()))}

    def _node_scores(self, cov: sp.csc_matrix, cols: list[int]) -> np.ndarray:
        return np.asarray(cov[:, cols].power(NAV_POWER) @ self.idf[cols]).ravel()

    def navigate(self, question: str, allowed=None, top_k: int = 40) -> tuple[list[tuple[int, list[str]]], dict]:
        """Leaves read, best first, with their passages (restricted to ``allowed``)."""
        cols = list(dict.fromkeys(self.vocab[w] for w in _words(question) if w in self.vocab))
        info = {"topics": [], "leaves": []}
        if not cols:
            info["note"] = "no question word is in the tree's vocabulary"
            return [], info
        s_top = self._node_scores(self.cov_top, cols)
        topics = [int(t) for t in np.argsort(-s_top)]
        s_leaf = self._node_scores(self.cov_leaf, cols)
        read, n = [], 0
        for beam in range(TOP_BEAM, len(topics) + TOP_BEAM, TOP_BEAM):   # widen if too few passages
            opened = topics[:beam]
            leaves = sorted((int(l) for t in opened for l in self.children.get(t, [])),
                            key=lambda l: -s_leaf[l])
            for l in leaves:
                if any(l == r for r, _ in read):
                    continue
                if len(read) >= MAX_LEAVES and n >= top_k:
                    break
                ps = [p for p in self.leaf_passages.get(l, []) if allowed is None or p in allowed]
                read.append((l, ps))
                n += len(ps)
            if n >= top_k:
                break
        info["topics"] = [{"topic": f"T{t}", "label": self.topic_label[t],
                           "score": round(float(s_top[t]), 3)} for t in topics[:TOP_BEAM]]
        info["leaves"] = [{"leaf": f"L{l}", "topic": f"T{self.leaf_parent[l]}",
                           "label": self.leaf_label[l], "passages": len(ps),
                           "score": round(float(s_leaf[l]), 3)} for l, ps in read]
        return read, info

    def path(self, leaf: int) -> str:
        t = self.leaf_parent[leaf]
        return f"{self.topic_label[t]} › {self.leaf_label[leaf]}"
