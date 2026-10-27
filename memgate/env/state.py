"""Session state shared by the live runner, the gym environment and the simulator (plan §3.7)."""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field

from memgate.schema import MemoryRecord, Mode, Outcome, SessionScript, Turn


@dataclass
class SessionState:
    script: SessionScript
    store: list[MemoryRecord]
    mode: Mode = "persistent"
    t: int = 0
    in_context: dict[str, int] = field(default_factory=dict)        # id -> turn it entered the context
    injected_at: dict[str, list[int]] = field(default_factory=dict)  # id -> every turn it was injected
    mentions: dict[str, int] = field(default_factory=dict)           # id -> times its value appeared in a reply
    intrusions: dict[str, int] = field(default_factory=dict)         # id -> unrelated mentions (n_m, C8)
    cand_seen: dict[str, int] = field(default_factory=dict)          # id -> times offered as a candidate
    last_mentioned: frozenset[str] = frozenset()
    updated_keys: set[str] = field(default_factory=set)              # keys the user changed this session
    conflict_flag: bool = False
    history: list[tuple[str, str]] = field(default_factory=list)     # (user content as stored, reply); live only

    @property
    def T(self) -> int:
        return self.script.T

    @property
    def turn(self) -> Turn:
        return self.script.turns[self.t]

    @property
    def done(self) -> bool:
        return self.t >= self.T

    def by_id(self) -> dict[str, MemoryRecord]:
        return {r.id: r for r in self.store}

    def effective_context(self, chosen_ids: list[str] | set[str]) -> set[str]:
        """Memories the model sees this turn, which is also what the λ term charges for."""
        if self.mode == "persistent":
            return set(self.in_context) | set(chosen_ids)
        return set(chosen_ids)

    def ever_mentioned(self) -> set[str]:
        return {i for i, n in self.mentions.items() if n > 0}

    def fork(self) -> SessionState:
        return copy.deepcopy(self)

    def snapshot(self) -> dict:
        """Everything the behaviour model may need, JSON-serialisable (plan §2.5, F1)."""
        return {
            "t": self.t, "T": self.T, "mode": self.mode,
            "store_ids": [r.id for r in self.store],
            "in_context": dict(self.in_context),
            "injected_at": {k: list(v) for k, v in self.injected_at.items()},
            "mentions": dict(self.mentions),
            "intrusions": dict(self.intrusions),
            "cand_seen": dict(self.cand_seen),
            "last_mentioned": sorted(self.last_mentioned),
            "updated_keys": sorted(self.updated_keys),
            "conflict_flag": self.conflict_flag,
        }


def snapshot_json(st: SessionState) -> str:
    return json.dumps(st.snapshot(), sort_keys=True)


def new_state(script: SessionScript, initial_store: list[MemoryRecord], mode: Mode) -> SessionState:
    return SessionState(script=script, store=list(initial_store), mode=mode)


def note_candidates_(st: SessionState, cand_ids: list[str]) -> None:
    for i in cand_ids:
        st.cand_seen[i] = st.cand_seen.get(i, 0) + 1


def advance_(st: SessionState, turn: Turn, chosen: list[MemoryRecord], out: Outcome,
             stored_user: str | None = None, reply: str | None = None) -> None:
    """In-place transition. Order fixed by plan §3.7. Candidate counts are recorded by the caller
    (note_candidates_) before the decision, because they are part of the observation."""
    for i in out.mentioned_ids:
        st.mentions[i] = st.mentions.get(i, 0) + 1
    for i in out.intrusion_ids:
        st.intrusions[i] = st.intrusions.get(i, 0) + 1
    st.last_mentioned = frozenset(out.mentioned_ids)
    for r in chosen:
        st.injected_at.setdefault(r.id, []).append(turn.idx)
        if st.mode == "persistent":
            st.in_context.setdefault(r.id, turn.idx)   # M6: never removed
    if st.mode == "ephemeral":
        st.in_context = {}
    st.conflict_flag = False
    if turn.kind == "update" and turn.new_record is not None:
        old_ids = {r.id for r in st.store if r.key == turn.target_key}
        exposed = set(st.in_context) if st.mode == "persistent" else set(st.injected_at)
        st.conflict_flag = bool(old_ids & exposed)
        st.store = st.store + [turn.new_record]
        st.updated_keys.add(turn.target_key)
    if stored_user is not None:
        st.history.append((stored_user, reply or ""))
    st.t += 1
