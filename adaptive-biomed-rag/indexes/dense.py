"""Dense vector indexes.

Two embedding backends, chosen by the recipe's ``embedding.model``:

  * medcpt  : MedCPT vectors (768-d) for any built chunking strategy - whole
              passages (scripts/build_medcpt_index.py) or chunks
              (scripts/build_chunk_indexes.py) - searched with FAISS inner product
              (exact, HNSW or IVF) over the full or the dev corpus. Queries are
              encoded locally with the MedCPT Query Encoder, so dense search makes
              no API call.
  * ada-002 : the 3,610 NB04 semantic-chunk vectors (1536-d, strategy
              ``semantic_nb04``), with
              ``flat`` exact cosine or ``hnsw`` (if `hnswlib` is installed; else it
              falls back to flat and reports it). Query embedding goes through
              common.llm.embed_query (cached); if unavailable, search returns [] and
              the caller relies on BM25.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from common import paths
from ingestion.loader import load_chunks, load_dev_corpus_ids, load_semantic_embeddings
from common.llm import embed_query

MEDCPT_QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"
QUERY_MAX_LENGTH = 64
HNSW_M, HNSW_EF_CONSTRUCTION, DEFAULT_EF_SEARCH = 32, 200, 64
IVF_NLIST_FACTOR, DEFAULT_NPROBE = 4, 8          # nlist = factor * sqrt(n vectors)


class DenseIndex:
    def __init__(self, strategy: str = "semantic_nb04", index_type: str = "flat"):
        emb = load_semantic_embeddings()
        chunks = load_chunks(strategy)[["chunk_id", "parent_passage_id", "text"]]
        merged = emb.merge(chunks, on="chunk_id", how="inner", suffixes=("", "_c"))

        self.chunk_ids = merged["chunk_id"].astype(str).tolist()
        self.parent_ids = merged["parent_passage_id"].astype(str).tolist()
        self.texts = merged["text"].astype(str).tolist()

        mat = np.array(merged["embedding"].tolist(), dtype=np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
        self._mat = mat / norms                    # L2-normalized for cosine
        self.dim = self._mat.shape[1]
        self.requested_type = index_type
        self.index_type = "flat"
        self._hnsw = None
        if index_type == "hnsw":
            self._try_build_hnsw()

    def _try_build_hnsw(self):
        try:
            import hnswlib
            idx = hnswlib.Index(space="cosine", dim=self.dim)
            idx.init_index(max_elements=len(self.chunk_ids), ef_construction=200, M=16)
            idx.add_items(self._mat, np.arange(len(self.chunk_ids)))
            idx.set_ef(64)
            self._hnsw = idx
            self.index_type = "hnsw"
        except Exception:
            self.index_type = "flat"           # graceful fallback

    def search(self, query: str, top_k: int = 40, ef_search: int | None = None,
               nprobe: int | None = None) -> list[dict]:
        qv = embed_query(query)
        if qv is None:
            return []
        q = np.asarray(qv, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-12)

        if self._hnsw is not None:
            labels, dists = self._hnsw.knn_query(q, k=min(top_k, len(self.chunk_ids)))
            idx = labels[0]
            sims = 1.0 - dists[0]
        else:
            sims = self._mat @ q
            idx = sims.argsort()[::-1][:top_k]
            sims = sims[idx]

        return [
            {
                "chunk_id": self.chunk_ids[i],
                "parent_passage_id": self.parent_ids[i],
                "text": self.texts[i],
                "dense_score": float(s),
            }
            for i, s in zip(idx, sims)
        ]


@lru_cache(maxsize=1)
def _medcpt_query_encoder():
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MEDCPT_QUERY_MODEL)
    model = AutoModel.from_pretrained(MEDCPT_QUERY_MODEL).eval()
    return tok, model


@lru_cache(maxsize=8192)
def encode_medcpt_query(text: str) -> np.ndarray:
    """MedCPT query vector: [CLS] of the Query Encoder, max_length=64, unnormalized.

    Cached, so comparing index types on the same questions encodes each once.
    Callers must not modify the returned array.
    """
    import torch
    tok, model = _medcpt_query_encoder()
    enc = tok([text], truncation=True, padding=True, max_length=QUERY_MAX_LENGTH,
              return_tensors="pt")
    with torch.inference_mode():
        return model(**enc).last_hidden_state[0, 0, :].numpy().astype(np.float32)


class MedCPTIndex:
    """FAISS inner-product search over the MedCPT vectors of one chunking strategy.

    ``corpus`` selects the full corpus or the frozen dev corpus (a row subset of the
    same vectors, by parent passage). ``index_type`` is ``flat`` (exact), ``hnsw``
    (graph ANN, M=32) or ``ivf`` (cluster ANN, nlist = 4*sqrt(n)); efSearch / nprobe
    are per-query.
    """

    def __init__(self, corpus: str = "full", index_type: str = "flat",
                 strategy: str = "passage"):
        import faiss
        if strategy == "passage":
            vecs = np.load(paths.MEDCPT_EMBEDDINGS)
            ids = pd.read_parquet(paths.MEDCPT_IDS)["canonical_passage_id"].astype(str)
            table = pd.DataFrame({"chunk_id": ids, "parent_passage_id": ids})
            texts = load_chunks("passage").set_index("chunk_id")["text"].reindex(ids).fillna("")
            table["text"] = texts.to_numpy()
        else:
            vecs = np.load(paths.CHUNK_INDEX_DIR / strategy / "embeddings.npy")
            table = load_chunks(strategy)[["chunk_id", "parent_passage_id", "text"]]
        if corpus == "dev":
            keep = table["parent_passage_id"].isin(load_dev_corpus_ids()).to_numpy()
            vecs, table = vecs[keep], table[keep]
        self.chunk_ids = table["chunk_id"].astype(str).tolist()
        self.parent_ids = table["parent_passage_id"].astype(str).tolist()
        self.texts = table["text"].astype(str).tolist()
        self.requested_type = index_type
        self.index_type = index_type
        self.nlist = None

        d = vecs.shape[1]
        if index_type == "hnsw":
            index = faiss.IndexHNSWFlat(d, HNSW_M, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
        elif index_type == "ivf":
            self.nlist = int(IVF_NLIST_FACTOR * np.sqrt(len(vecs)))
            index = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, self.nlist,
                                       faiss.METRIC_INNER_PRODUCT)
            index.train(vecs)
        else:
            index = faiss.IndexFlatIP(d)
            self.index_type = "flat"
        index.add(np.ascontiguousarray(vecs, dtype=np.float32))
        self._index = index

    def search(self, query: str, top_k: int = 40, ef_search: int | None = None,
               nprobe: int | None = None) -> list[dict]:
        import faiss
        q = encode_medcpt_query(query)[None, :]
        params = None                      # per-call params keep concurrent searches safe
        if self.index_type == "hnsw":
            params = faiss.SearchParametersHNSW(efSearch=max(ef_search or DEFAULT_EF_SEARCH, top_k))
        elif self.index_type == "ivf":
            params = faiss.SearchParametersIVF(nprobe=nprobe or DEFAULT_NPROBE)
        scores, idx = self._index.search(q, top_k, params=params)
        return [
            {
                "chunk_id": self.chunk_ids[i],
                "parent_passage_id": self.parent_ids[i],
                "text": self.texts[i],
                "dense_score": float(s),
            }
            for i, s in zip(idx[0], scores[0]) if i >= 0
        ]


@lru_cache(maxsize=24)
def get_dense(strategy: str = "passage", index_type: str = "flat",
              model: str = "medcpt",
              corpus: str = "full") -> DenseIndex | MedCPTIndex:
    if model == "medcpt":
        if strategy in ("semantic_nb04", "agentic"):
            raise ValueError(f"No MedCPT index for chunking '{strategy}'")
        return MedCPTIndex(corpus, index_type, strategy)
    if strategy != "semantic_nb04" or corpus != "full":
        raise ValueError("ada-002 vectors exist only for NB04's semantic chunks (full corpus)")
    return DenseIndex(strategy, index_type)
