"""MedCPT Article Encoder for offline index builds.

  encode()       [CLS] vector per text (the official MedCPT recipe; max_length 512).
  encode_late()  late chunking: one contextual pass over a passage's words, then the
                 mean of the contextual token vectors inside each span. Passages longer
                 than one 512-token window are covered by consecutive windows. MedCPT was
                 trained with [CLS] pooling, so span mean-pooling is experimental.

Both sort inputs by length so batches pad little, and can checkpoint to shards so an
interrupted build resumes. Vectors are float32, unnormalized (inner product).

``cool_down`` makes the encoder rest ``cool_down`` seconds per second of work (1.0 =
about half duty), so long builds do not overheat a fanless laptop.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import numpy as np

ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"
MAX_LENGTH = 512
DIM = 768


class ArticleEncoder:
    def __init__(self, device: str | None = None, cool_down: float = 0.0):
        import torch
        from transformers import AutoModel, AutoTokenizer
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        self.torch = torch
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(ARTICLE_MODEL)
        self.model = AutoModel.from_pretrained(ARTICLE_MODEL).to(self.device).eval()
        self.revision = getattr(self.model.config, "_commit_hash", None)
        self.cool_down = cool_down

    def _rest(self, t0: float) -> None:
        if self.cool_down > 0:
            time.sleep((time.perf_counter() - t0) * self.cool_down)

    # ------------------------------------------------------------------ #
    def _cls(self, texts: list[str]) -> np.ndarray:
        t0 = time.perf_counter()
        enc = self.tok(texts, truncation=True, padding=True, max_length=MAX_LENGTH,
                       return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            out = self.model(**enc).last_hidden_state[:, 0, :].float().cpu().numpy()
        self._rest(t0)
        return out

    def encode(self, texts: list[str], batch_size: int = 32, shard_dir: Path | None = None,
               shard_size: int = 4096, log: Callable[[str], None] = print,
               progress: Callable[[int, int], None] | None = None) -> np.ndarray:
        """[CLS] vectors in input order; shards (if ``shard_dir``) make it resumable."""
        order = np.argsort([len(t) for t in texts], kind="stable")
        vecs = np.empty((len(texts), DIM), dtype=np.float32)
        n_shards = max(1, (len(order) + shard_size - 1) // shard_size)
        t0, done = time.perf_counter(), 0
        for s in range(n_shards):
            idx = order[s * shard_size:(s + 1) * shard_size]
            shard = shard_dir / f"shard_{s:04d}.npy" if shard_dir else None
            if progress:
                progress(min(len(order), s * shard_size), len(order))
            if shard is not None and shard.exists():
                vecs[idx] = np.load(shard)
                continue
            out = np.vstack([self._cls([texts[i] for i in idx[b:b + batch_size]])
                             for b in range(0, len(idx), batch_size)]) if len(idx) else \
                np.empty((0, DIM), np.float32)
            if shard is not None:
                np.save(shard, out)
            vecs[idx] = out
            done += len(idx)
            rate = done / max(1e-9, time.perf_counter() - t0)
            left = len(order) - (s + 1) * shard_size
            log(f"  {min(len(order), (s + 1) * shard_size):,}/{len(order):,} encoded "
                f"| {rate:.0f}/s | ETA {max(0, left) / max(rate, 1e-9) / 60:.1f} min")
        return vecs

    # ------------------------------------------------------------------ #
    def encode_late(self, words: list[list[str]], spans: list[list[tuple[int, int]]],
                    log: Callable[[str], None] = print,
                    progress: Callable[[int, int], None] | None = None) -> list[np.ndarray]:
        """Late-chunk vectors: for passage i, one row per span in ``spans[i]``."""
        out: list[np.ndarray] = []
        t0 = time.perf_counter()
        for i, (w, sp) in enumerate(zip(words, spans)):
            t_one = time.perf_counter()
            out.append(self._late_one(w, sp))
            self._rest(t_one)
            if i % 2000 == 0:
                if progress:
                    progress(i, len(words))
                rate = (i + 1) / max(1e-9, time.perf_counter() - t0)
                log(f"  late {i:,}/{len(words):,} passages | {rate:.0f}/s")
        return out

    def _late_one(self, words: list[str], spans: list[tuple[int, int]]) -> np.ndarray:
        """Spans are sorted by start. Each window starts at the first unresolved span,
        so every pass resolves at least that span (pooling whatever of it fits)."""
        vecs = np.zeros((len(spans), DIM), dtype=np.float32)
        pending = list(range(len(spans)))
        start = 0
        while pending:
            enc = self.tok(words[start:], is_split_into_words=True, truncation=True,
                           max_length=MAX_LENGTH, return_tensors="pt")
            word_ids = enc.word_ids(0)
            covered_end = start + 1 + max(w for w in word_ids if w is not None)   # exclusive
            with self.torch.inference_mode():
                hidden = self.model(**enc.to(self.device)).last_hidden_state[0].float().cpu().numpy()
            pos = np.array([-1 if w is None else w + start for w in word_ids])
            remaining = []
            for j in pending:
                a, b = spans[j]
                if b <= covered_end or a == start:
                    vecs[j] = hidden[(pos >= a) & (pos < b)].mean(axis=0)
                else:
                    remaining.append(j)
            pending = remaining
            if pending:
                start = spans[pending[0]][0]
        return vecs
