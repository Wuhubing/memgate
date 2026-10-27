"""SessionState -> 50-dim observation (plan §2.3). Only harness-observable information is used:
the function never reads superseded_by or relevant_ids (E8)."""
from __future__ import annotations

import numpy as np

from memgate.env.state import SessionState
from memgate.schema import CATEGORIES, Candidate, MemoryRecord
from memgate.world.generator import TODAY

K = 3
PER_CAND = 14
OBS_DIM = PER_CAND * K + 8   # 50
CAND_HISTORY = (12, 13)
GLOBAL_HISTORY = (44, 45, 46, 47, 48, 49)
HISTORY_IDX = tuple(PER_CAND * i + j for i in range(K) for j in CAND_HISTORY) + GLOBAL_HISTORY


def _has_newer(rec: MemoryRecord, others: list[MemoryRecord]) -> bool:
    return any(o.key == rec.key and o.id != rec.id and o.created_at > rec.created_at for o in others)


def featurize(st: SessionState, cands: list[Candidate | None], mask_history: bool = False) -> np.ndarray:
    x = np.zeros(OBS_DIM, dtype=np.float32)
    recs = st.by_id()
    in_ctx = [recs[i] for i in st.in_context if i in recs]
    cand_recs = [c.record for c in cands if c is not None]
    now = TODAY + st.t
    for i, c in enumerate(cands):
        if c is None:
            continue
        r, o = c.record, PER_CAND * i
        x[o + 0] = 1.0
        x[o + 1] = c.cos
        x[o + 2] = c.bm25_norm
        x[o + 3] = (c.rank - 1) / (K - 1)
        x[o + 4] = min(max(now - r.created_at, 0) / 365.0, 1.0)
        x[o + 5] = min(r.tokens / 60.0, 1.0)
        x[o + 6 + CATEGORIES.index(r.category)] = 1.0
        x[o + 11] = float(_has_newer(r, cand_recs))
        x[o + 12] = float(any(m.key == r.key and m.id != r.id for m in in_ctx))
        x[o + 13] = min(st.cand_seen.get(r.id, 0) / 5.0, 1.0)
    T = max(st.T, 1)
    x[42] = st.t / T
    x[43] = (T - st.t) / T
    x[44] = min(len(in_ctx) / 10.0, 1.0)
    x[45] = min(sum(r.tokens for r in in_ctx) / 300.0, 1.0)
    x[46] = min(len(st.last_mentioned & set(st.in_context)) / 3.0, 1.0)
    x[47] = min(max(st.mentions.values(), default=0) / 5.0, 1.0)
    x[48] = min(sum(_has_newer(r, st.store) for r in in_ctx) / 3.0, 1.0)
    x[49] = float(st.conflict_flag)
    if mask_history:
        x[list(HISTORY_IDX)] = 0.0
    x = np.clip(x, -1.0, 1.0)
    assert not np.isnan(x).any()
    return x
