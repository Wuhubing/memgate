"""Data objects passed between modules (plan §2)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Literal

Category = Literal["preference", "project", "schedule", "contact", "trivia"]
CATEGORIES: tuple[str, ...] = ("preference", "project", "schedule", "contact", "trivia")
TurnKind = Literal["query", "chat", "update"]
Mode = Literal["persistent", "ephemeral"]
Answer = Literal["correct", "stale", "neither", "na"]


@dataclass(frozen=True)
class MemoryRecord:
    id: str                       # "u003-m017"
    user_id: str
    key: str                      # versions of one fact share a key, e.g. "kestrel.deadline"
    value: str                    # unique within the user's world, e.g. "Friday 14 November"
    text: str                     # the sentence that gets injected
    category: Category
    topic: str                    # the word a chat turn uses to mention this subject, e.g. "Kestrel"
    created_at: int               # virtual day
    superseded_by: str | None = None   # never visible to the controller
    tokens: int = 0


@dataclass(frozen=True)
class Turn:
    idx: int
    kind: TurnKind
    text: str
    target_key: str | None = None            # query / update
    answer_value: str | None = None          # query: current value
    stale_values: tuple[str, ...] = ()       # query: all old values of target_key
    relevant_ids: frozenset[str] = frozenset()
    new_record: MemoryRecord | None = None   # update
    topic_key: str | None = None             # chat that names a memory's subject


@dataclass(frozen=True)
class SessionScript:
    id: str
    user_id: str
    turns: tuple[Turn, ...]

    @property
    def T(self) -> int:
        return len(self.turns)


@dataclass(frozen=True)
class Candidate:
    record: MemoryRecord
    cos: float
    bm25: float          # raw score
    bm25_norm: float     # divided by the max among this turn's candidates
    rank: int            # 1..k after fusion
    fused: float         # raw RRF score


@dataclass(frozen=True)
class Outcome:
    answer: Answer
    mentioned_ids: frozenset[str]
    intrusion_ids: frozenset[str]
    r_quality: float
    r_stale: float
    r_intrusion: float
    r_tokens: float
    hedged: bool = False

    @property
    def reward(self) -> float:
        return self.r_quality + self.r_stale + self.r_intrusion + self.r_tokens


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str = ""
    system_fingerprint: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit: int | None = None
    cache_miss: int | None = None
    reasoning_tokens: int = 0
    finish_reason: str | None = None
    latency_s: float = 0.0
    from_cache: bool = False


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_hit: int = 0
    cache_miss: int = 0
    cost_cny: float = 0.0
    extra: dict = field(default_factory=dict)


# ---- (de)serialisation helpers -------------------------------------------------------------

def record_to_dict(r: MemoryRecord) -> dict:
    return asdict(r)


def record_from_dict(d: dict) -> MemoryRecord:
    return MemoryRecord(**d)


def turn_to_dict(t: Turn) -> dict:
    d = asdict(t)
    d["stale_values"] = list(t.stale_values)
    d["relevant_ids"] = sorted(t.relevant_ids)
    d["new_record"] = record_to_dict(t.new_record) if t.new_record else None
    return d


def turn_from_dict(d: dict) -> Turn:
    d = dict(d)
    d["stale_values"] = tuple(d.get("stale_values", ()))
    d["relevant_ids"] = frozenset(d.get("relevant_ids", ()))
    nr = d.get("new_record")
    d["new_record"] = record_from_dict(nr) if nr else None
    return Turn(**d)


def script_to_json(s: SessionScript) -> str:
    return json.dumps({"id": s.id, "user_id": s.user_id, "turns": [turn_to_dict(t) for t in s.turns]})


def script_from_json(line: str) -> SessionScript:
    d = json.loads(line)
    return SessionScript(d["id"], d["user_id"], tuple(turn_from_dict(t) for t in d["turns"]))
