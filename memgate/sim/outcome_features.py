"""Features of the behaviour model. One implementation for logs and for the simulator (F2).

Functions take any object with SessionState's attributes: a live SessionState, or a StateView
rebuilt from a logged snapshot.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from memgate.schema import CATEGORIES, MemoryRecord, Turn

ANSWER_CLASSES = ("correct", "stale", "neither")

ANSWER_FEATURES = (
    "cur_vis", "cur_inj_now", "n_stale_vis", "stale_inj_now", "user_said_new", "cur_ctx_age", "stale_ctx_age",
    "stale_mentioned", "cur_mentioned", "stale_ever_injected", "n_ctx", "persistent", "t_norm",
    "stale_and_cur", "stale_and_said", "stale_vis_x_persistent",
)

MENTION_FEATURES = (
    "in_eff", "inj_now", "ctx_age", "ever_injected", "n_injections", "n_intr", "n_ment", "last_ment", "sim",
    "topic_match", "kind_query", "kind_chat", "kind_update", *[f"cat_{c}" for c in CATEGORIES], "n_ctx",
    "persistent", "in_eff_x_topic", "ment_hist_not_vis",
)


@dataclass
class StateView:
    t: int
    T: int
    mode: str
    store: list[MemoryRecord]
    in_context: dict[str, int] = field(default_factory=dict)
    injected_at: dict[str, list[int]] = field(default_factory=dict)
    mentions: dict[str, int] = field(default_factory=dict)
    intrusions: dict[str, int] = field(default_factory=dict)
    last_mentioned: frozenset[str] = frozenset()
    updated_keys: set[str] = field(default_factory=set)

    def effective_context(self, chosen_ids) -> set[str]:
        if self.mode == "persistent":
            return set(self.in_context) | set(chosen_ids)
        return set(chosen_ids)

    @classmethod
    def from_snapshot(cls, snap: dict, records: dict[str, MemoryRecord]) -> StateView:
        return cls(
            t=snap["t"], T=snap["T"], mode=snap["mode"], store=[records[i] for i in snap["store_ids"]],
            in_context={k: int(v) for k, v in snap["in_context"].items()},
            injected_at={k: list(v) for k, v in snap["injected_at"].items()},
            mentions=dict(snap["mentions"]), intrusions=dict(snap["intrusions"]),
            last_mentioned=frozenset(snap["last_mentioned"]), updated_keys=set(snap["updated_keys"]))


def answer_features(st, turn: Turn, chosen_ids: list[str]) -> dict[str, float]:
    eff = st.effective_context(chosen_ids)
    by_val = {r.value: r for r in st.store if r.key == turn.target_key}
    cur = by_val.get(turn.answer_value)
    stale = [by_val[v] for v in turn.stale_values if v in by_val]
    stale_vis = [r for r in stale if r.id in eff]

    def age(rid: str) -> float:
        inj = st.injected_at.get(rid)
        return 0.0 if not inj else min((st.t - inj[0]) / 10.0, 2.0)

    cur_vis = float(cur is not None and cur.id in eff)
    n_stale_vis = min(len(stale_vis), 3) / 3.0
    said = float(turn.target_key in st.updated_keys)
    persistent = float(st.mode == "persistent")
    return {
        "cur_vis": cur_vis,
        "cur_inj_now": float(cur is not None and cur.id in chosen_ids),
        "n_stale_vis": n_stale_vis,
        "stale_inj_now": float(any(r.id in chosen_ids for r in stale)),
        "user_said_new": said,
        "cur_ctx_age": age(cur.id) if cur is not None and cur.id in eff else 0.0,
        "stale_ctx_age": max((age(r.id) for r in stale_vis), default=0.0),
        "stale_mentioned": float(any(st.mentions.get(r.id, 0) for r in stale)),
        "cur_mentioned": float(cur is not None and st.mentions.get(cur.id, 0) > 0),
        "stale_ever_injected": float(any(r.id in st.injected_at for r in stale)),
        "n_ctx": min(len(eff) / 10.0, 2.0),
        "persistent": persistent,
        "t_norm": st.t / 20.0,
        "stale_and_cur": float(n_stale_vis > 0) * cur_vis,
        "stale_and_said": float(n_stale_vis > 0) * said,
        "stale_vis_x_persistent": n_stale_vis * persistent,
    }


def mention_candidates(st, turn: Turn, chosen_ids: list[str], user_values: set[str]) -> list[MemoryRecord]:
    """Memories that could produce an unrelated mention this turn (plan §3.10)."""
    eff = st.effective_context(chosen_ids)
    ever = {i for i, n in st.mentions.items() if n > 0}
    pool = eff | ever | set(chosen_ids)
    out = []
    for r in st.store:
        if r.id not in pool or r.id in turn.relevant_ids or r.key == turn.target_key or r.value in user_values:
            continue
        out.append(r)
    return out


def mention_features(st, turn: Turn, chosen_ids: list[str], r: MemoryRecord, sim: float) -> dict[str, float]:
    eff = st.effective_context(chosen_ids)
    inj = st.injected_at.get(r.id, [])
    in_eff = float(r.id in eff)
    topic = float(turn.topic_key == r.key)
    f = {
        "in_eff": in_eff,
        "inj_now": float(r.id in chosen_ids),
        "ctx_age": min((st.t - inj[0]) / 10.0, 2.0) if inj else 0.0,
        "ever_injected": float(bool(inj)),
        "n_injections": min(len(inj), 5) / 5.0,
        "n_intr": min(st.intrusions.get(r.id, 0), 5) / 5.0,
        "n_ment": min(st.mentions.get(r.id, 0), 5) / 5.0,
        "last_ment": float(r.id in st.last_mentioned),
        "sim": float(sim),
        "topic_match": topic,
        "kind_query": float(turn.kind == "query"),
        "kind_chat": float(turn.kind == "chat"),
        "kind_update": float(turn.kind == "update"),
        "n_ctx": min(len(eff) / 10.0, 2.0),
        "persistent": float(st.mode == "persistent"),
        "in_eff_x_topic": in_eff * topic,
        "ment_hist_not_vis": float(st.mentions.get(r.id, 0) > 0 and not in_eff),
    }
    for c in CATEGORIES:
        f[f"cat_{c}"] = float(r.category == c)
    return f


def vec(d: dict[str, float], names: tuple[str, ...]) -> np.ndarray:
    return np.array([d[n] for n in names], dtype=np.float64)
