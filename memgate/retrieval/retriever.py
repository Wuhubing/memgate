"""Fixed retriever: BM25Plus + embedding cosine, fused with reciprocal rank fusion, top-3 (plan §3.3)."""
from __future__ import annotations

import numpy as np
from rank_bm25 import BM25Plus

from memgate.retrieval.embed import EmbeddingStore
from memgate.schema import Candidate, MemoryRecord
from memgate.text import sha, tokenize

RRF_K = 60


class Retriever:
    def __init__(self, emb: EmbeddingStore, k: int = 3):
        self.emb = emb
        self.k = k
        self._bm25_cache: dict[tuple[str, ...], tuple[BM25Plus, np.ndarray]] = {}

    def _index(self, store: list[MemoryRecord]) -> tuple[BM25Plus, np.ndarray]:
        key = tuple(r.id for r in store)
        hit = self._bm25_cache.get(key)
        if hit is None:
            bm = BM25Plus([tokenize(r.text) for r in store])
            mat = np.stack([self.emb.get(r.text) for r in store])
            if len(self._bm25_cache) > 50_000:
                self._bm25_cache.clear()
            hit = self._bm25_cache[key] = (bm, mat)
        return hit

    def topk(self, query: str, store: list[MemoryRecord], exclude: set[str] | frozenset[str] = frozenset()
             ) -> list[Candidate | None]:
        """Always returns exactly k slots; empty slots are None (D6). Old versions are not filtered (M2)."""
        pool_idx = [i for i, r in enumerate(store) if r.id not in exclude]
        out: list[Candidate | None] = [None] * self.k
        if not pool_idx:
            return out
        bm, mat = self._index(store)
        bm_scores = np.asarray(bm.get_scores(tokenize(query)), dtype=float)[pool_idx]
        cos = (mat[pool_idx] @ self.emb.get(query)).astype(float)
        # tie-break by sha256 of the record id: deterministic and unrelated to recency (D2)
        tb = np.array([int(sha(store[i].id, 8), 16) for i in pool_idx], dtype=float)
        r_bm = _ranks(bm_scores, tb)
        r_cos = _ranks(cos, tb)
        fused = 1.0 / (RRF_K + r_bm) + 1.0 / (RRF_K + r_cos)
        order = sorted(range(len(pool_idx)), key=lambda j: (-fused[j], tb[j]))[: self.k]
        top_bm = max((bm_scores[j] for j in order), default=0.0)
        for rank, j in enumerate(order, start=1):
            out[rank - 1] = Candidate(
                record=store[pool_idx[j]], cos=float(cos[j]), bm25=float(bm_scores[j]),
                bm25_norm=float(bm_scores[j] / top_bm) if top_bm > 0 else 0.0, rank=rank, fused=float(fused[j]))
        return out


def _ranks(scores: np.ndarray, tiebreak: np.ndarray) -> np.ndarray:
    """1-based ranks, higher score first, ties broken by `tiebreak`."""
    order = np.lexsort((tiebreak, -scores))
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks
