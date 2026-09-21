"""BM25 lexical index over chunk texts.

Robust for exact biomedical terminology, gene/drug names and abbreviations
(where dense embeddings can under-perform). Built lazily and cached per chunk
strategy and corpus. Zero external services, zero LLM.

Scores are identical to rank_bm25's BM25Okapi (k1=1.5, b=0.75, negative idf floored
at 0.25 x mean idf), but computed as one sparse matrix-vector product, so queries
stay fast on large chunk tables (e.g. ~356k proposition units).
"""
from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from ingestion.loader import load_chunks

_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]+")
K1, B, EPSILON = 1.5, 0.75, 0.25


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text or "")]


class BM25Index:
    def __init__(self, chunk_ids: list[str], parent_ids: list[str], texts: list[str]):
        self.chunk_ids = chunk_ids
        self.parent_ids = parent_ids
        self.texts = texts
        self._vec = CountVectorizer(tokenizer=_tokenize, lowercase=False, token_pattern=None)
        tf = self._vec.fit_transform(texts).tocsr().astype(np.float32)      # docs x terms

        n_docs = tf.shape[0]
        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log(n_docs - df + 0.5) - np.log(df + 0.5)
        idf[idf < 0] = EPSILON * idf.mean()
        self._idf = idf.astype(np.float32)

        # Per-entry term weight tf*(k1+1) / (tf + k1*(1-b+b*dl/avgdl)).
        norm = K1 * (1 - B + B * doc_len / max(doc_len.mean(), 1e-9))
        rows = np.repeat(np.arange(n_docs), np.diff(tf.indptr))
        w = tf.copy()
        w.data = tf.data * (K1 + 1) / (tf.data + norm[rows])
        self._w = w.tocsc()                                                   # fast column slicing

    def scores(self, query: str) -> np.ndarray:
        terms = self._vec.vocabulary_
        counts: dict[int, int] = {}
        for t in _tokenize(query):                  # repeated query terms count again
            j = terms.get(t)
            if j is not None:
                counts[j] = counts.get(j, 0) + 1
        if not counts:
            return np.zeros(len(self.chunk_ids), dtype=np.float32)
        cols = np.fromiter(counts, dtype=np.int64)
        q = self._idf[cols] * np.fromiter(counts.values(), dtype=np.float32)
        return np.asarray(self._w[:, cols].dot(q)).ravel()

    def search(self, query: str, top_k: int = 40) -> list[dict]:
        scores = self.scores(query)
        k = min(top_k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k] if k else np.array([], dtype=int)
        order = top[np.argsort(-scores[top], kind="stable")]
        return [
            {
                "chunk_id": self.chunk_ids[i],
                "parent_passage_id": self.parent_ids[i],
                "text": self.texts[i],
                "bm25_score": float(scores[i]),
            }
            for i in order if scores[i] > 0
        ]


@lru_cache(maxsize=16)
def get_bm25(strategy: str = "semantic", corpus: str = "full") -> BM25Index:
    df = load_chunks(strategy, corpus)
    return BM25Index(
        chunk_ids=df["chunk_id"].astype(str).tolist(),
        parent_ids=df["parent_passage_id"].astype(str).tolist(),
        texts=df["text"].tolist(),
    )
