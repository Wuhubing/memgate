"""Fictional users, their memory stores, and annotated session scripts (plan §3.2)."""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from memgate.config import WorldConfig
from memgate.retrieval.embed import EmbeddingStore, make_embedder
from memgate.schema import (
    MemoryRecord,
    SessionScript,
    Turn,
    record_from_dict,
    record_to_dict,
    script_from_json,
    script_to_json,
)
from memgate.text import count_tokens, normalize
from memgate.world import templates as tpl
from memgate.world.aliases import AliasIndex
from memgate.world.pools import MONTH_ABBR, MONTHS, WEEKDAYS, ValueFactory

TODAY = 400  # virtual "now" (days); initial records are older, in-session updates are TODAY + t


@dataclass(frozen=True)
class KeySpec:
    key: str
    kind: str            # changing | preference | trivia
    attr: str
    category: str
    np: str              # noun phrase with the project filled in
    topic: str           # chat topic word with the project filled in


@dataclass
class User:
    id: str
    split: str
    keys: list[KeySpec]
    initial_store: list[MemoryRecord]
    spares: dict[str, list[MemoryRecord]]          # changing key -> unused versions for updates
    aliases: dict[str, list[str]]                  # value -> surface forms
    _index: AliasIndex | None = field(default=None, repr=False, compare=False)

    @property
    def index(self) -> AliasIndex:
        if self._index is None:
            self._index = AliasIndex(self.aliases)
        return self._index

    def spec(self, key: str) -> KeySpec:
        return next(k for k in self.keys if k.key == key)

    @property
    def changing_keys(self) -> list[str]:
        return [k.key for k in self.keys if k.kind == "changing"]

    @property
    def queryable_keys(self) -> list[str]:
        return [k.key for k in self.keys if k.kind in ("changing", "preference")]


def current_record(store: list[MemoryRecord], key: str) -> MemoryRecord:
    return max((r for r in store if r.key == key), key=lambda r: (r.created_at, r.id))


def stale_records(store: list[MemoryRecord], key: str) -> list[MemoryRecord]:
    cur = current_record(store, key)
    return [r for r in store if r.key == key and r.id != cur.id]


# ---------------------------------------------------------------------------------------------
# world generation


def _make_user(uid: str, split: str, cfg: WorldConfig, vf: ValueFactory, rng: np.random.Generator) -> User:
    keys: list[KeySpec] = []
    records: list[MemoryRecord] = []
    spares: dict[str, list[MemoryRecord]] = {}
    aliases: dict[str, list[str]] = {}
    used_days: set[int] = set()
    n = 0

    def new_rec(spec: KeySpec, attr: tpl.Attr, value: str, created: int, project: str | None, rid: str):
        text = tpl.fill(attr.record, v=value, P=project or "")
        return MemoryRecord(rid, uid, spec.key, value, text, attr.category, spec.topic, created,
                            None, count_tokens(text))

    n_projects = cfg.n_changing // len(tpl.PROJECT_ATTRS)
    projects = [vf.core() for _ in range(n_projects)]
    for p in projects:
        for attr in tpl.PROJECT_ATTRS:
            spec = KeySpec(f"{p.lower()}.{attr.name}", "changing", attr.name, attr.category,
                           tpl.fill(attr.np, P=p), tpl.fill(attr.topic, P=p))
            keys.append(spec)
            n_old = int(rng.integers(1, cfg.max_old_versions + 1)) if rng.random() < cfg.p_stale else 0
            days = sorted(int(d) for d in rng.choice(np.arange(0, TODAY - 30), size=n_old + 1, replace=False))
            versions = []
            for d in days:
                value, al = vf.draw(attr.vtype, used_days)
                aliases[value] = al
                versions.append(new_rec(spec, attr, value, d, p, f"{uid}-m{n:03d}")); n += 1
            # link superseded_by along the chain (generator-only annotation)
            for i in range(len(versions) - 1):
                versions[i] = dataclasses.replace(versions[i], superseded_by=versions[i + 1].id)
            records.extend(versions)
            sp = []
            for j in range(cfg.spare_values):
                value, al = vf.draw(attr.vtype, used_days)
                aliases[value] = al
                sp.append(new_rec(spec, attr, value, TODAY, p, f"{uid}-x{len(keys) - 1:02d}{j:02d}"))
            spares[spec.key] = sp

    for kind, pool, count in (("preference", tpl.PREFERENCE_ATTRS, cfg.n_preference),
                              ("trivia", tpl.TRIVIA_ATTRS, cfg.n_trivia)):
        for attr in pool[:count]:
            spec = KeySpec(f"{kind}.{attr.name}", kind, attr.name, attr.category, attr.np, attr.topic)
            keys.append(spec)
            value, al = vf.draw(attr.vtype, used_days)
            aliases[value] = al
            records.append(new_rec(spec, attr, value, int(rng.integers(0, TODAY - 30)), None, f"{uid}-m{n:03d}"))
            n += 1
    return User(uid, split, keys, records, spares, aliases)


class World:
    def __init__(self, cfg: WorldConfig, users: dict[str, User], emb: EmbeddingStore):
        self.cfg = cfg
        self.users = users
        self.emb = emb
        self.path: Path | None = None
        self.records: dict[str, MemoryRecord] = {}
        for u in users.values():
            for r in u.initial_store:
                self.records[r.id] = r
            for sp in u.spares.values():
                for r in sp:
                    self.records[r.id] = r

    def split_users(self, split: str) -> list[User]:
        return [u for u in self.users.values() if u.split == split]

    def user(self, uid: str) -> User:
        return self.users[uid]

    # ---- persistence --------------------------------------------------------------------------
    def save(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        users = []
        for u in self.users.values():
            users.append({
                "id": u.id, "split": u.split,
                "keys": [dataclasses.asdict(k) for k in u.keys],
                "initial_store": [record_to_dict(r) for r in u.initial_store],
                "spares": {k: [record_to_dict(r) for r in v] for k, v in u.spares.items()},
                "aliases": u.aliases,
            })
        (out / "world.json").write_text(json.dumps({"config": self.cfg.model_dump(), "users": users}))
        self.emb.save(out / "embeddings.npz")

    @classmethod
    def load(cls, path: Path) -> World:
        path = Path(path)
        d = json.loads((path / "world.json").read_text())
        cfg = WorldConfig(**d["config"])
        users = {}
        for u in d["users"]:
            users[u["id"]] = User(
                u["id"], u["split"], [KeySpec(**k) for k in u["keys"]],
                [record_from_dict(r) for r in u["initial_store"]],
                {k: [record_from_dict(r) for r in v] for k, v in u["spares"].items()},
                u["aliases"])
        world = cls(cfg, users, EmbeddingStore.load(path / "embeddings.npz"))
        world.path = path
        return world


def generate_world(cfg: WorldConfig, embedder=None) -> World:
    rng = np.random.default_rng(cfg.seed)
    vf = ValueFactory(rng)
    n_tr, n_va = cfg.split.train, cfg.split.val
    assert n_tr + n_va + cfg.split.test == cfg.n_users
    users = {}
    for i in range(cfg.n_users):
        split = "train" if i < n_tr else "val" if i < n_tr + n_va else "test"
        uid = f"u{i:03d}"
        users[uid] = _make_user(uid, split, cfg, vf, rng)
    emb = EmbeddingStore(cfg.embedder)
    world = World(cfg, users, emb)
    texts = [r.text for r in world.records.values()]
    for u in users.values():
        texts.extend(all_turn_texts(u))
    emb.add_texts(texts, embedder or make_embedder(cfg.embedder))
    validate_world(world)
    return world


# ---------------------------------------------------------------------------------------------
# sessions


def all_turn_texts(u: User) -> list[str]:
    """Every text the session generator can emit for this user (so embeddings can be precomputed)."""
    out = []
    for k in u.queryable_keys:
        out += [tpl.fill(q, np=u.spec(k).np) for q in tpl.QUERY]
    for k in u.changing_keys:
        for r in u.spares[k]:
            out += [tpl.fill(t, np=u.spec(k).np, v=r.value) for t in tpl.UPDATE]
    for topic in {k.topic for k in u.keys}:
        out += [tpl.fill(c, T=topic) for c in tpl.CHAT_TOPIC]
    out += list(tpl.CHAT_PLAIN)
    return out


def zipf_pick(keys: list[str], rng: np.random.Generator, a: float, order: np.ndarray) -> str:
    ranks = np.arange(1, len(keys) + 1, dtype=float)
    p = ranks ** -a
    p /= p.sum()
    return keys[order[rng.choice(len(keys), p=p)]]


class SessionFactory:
    """Generates annotated session scripts for one split. Deterministic given (seed, counter)."""

    def __init__(self, world: World, split: str, seed: int, p_kind: dict[str, float] | None = None):
        self.world = world
        self.users = world.split_users(split)
        self.split = split
        self.seed = seed
        self.counter = 0
        self.p_kind = p_kind or world.cfg.p_kind

    def make(self, T: int | None = None, user: User | None = None) -> SessionScript:
        rng = np.random.default_rng([self.seed, self.counter])
        self.counter += 1
        u = user or self.users[int(rng.integers(len(self.users)))]
        sid = f"{self.split}-{u.id}-s{self.seed}-{self.counter - 1:05d}"
        return make_session(u, rng, T or self.world.cfg.T_default, self.world.cfg, sid, self.p_kind)


def make_session(u: User, rng: np.random.Generator, T: int, cfg: WorldConfig, sid: str,
                 p_kind: dict[str, float] | None = None) -> SessionScript:
    p_kind = p_kind or cfg.p_kind
    kinds = list(p_kind)
    probs = np.array([p_kind[k] for k in kinds], dtype=float)
    probs /= probs.sum()
    store = list(u.initial_store)
    spare_left = {k: list(v) for k, v in u.spares.items()}
    qkeys = u.queryable_keys
    order = rng.permutation(len(qkeys))
    turns = []
    for t in range(T):
        kind = kinds[rng.choice(len(kinds), p=probs)]
        if kind == "query":
            key = zipf_pick(qkeys, rng, cfg.zipf_a, order)
            cur = current_record(store, key)
            stale = tuple(r.value for r in stale_records(store, key))
            text = tpl.fill(tpl.QUERY[rng.integers(len(tpl.QUERY))], np=u.spec(key).np)
            turns.append(Turn(t, "query", text, key, cur.value, stale, frozenset({cur.id})))
        elif kind == "chat":
            if rng.random() < cfg.p_chat_topic:
                spec = u.keys[rng.integers(len(u.keys))]
                text = tpl.fill(tpl.CHAT_TOPIC[rng.integers(len(tpl.CHAT_TOPIC))], T=spec.topic)
                turns.append(Turn(t, "chat", text, topic_key=spec.key))
            else:
                turns.append(Turn(t, "chat", tpl.CHAT_PLAIN[rng.integers(len(tpl.CHAT_PLAIN))]))
        else:
            key = u.changing_keys[rng.integers(len(u.changing_keys))]
            if not spare_left[key]:
                raise RuntimeError(f"spare values exhausted for {key} (B8)")
            new = spare_left[key].pop(int(rng.integers(len(spare_left[key]))))
            new = dataclasses.replace(new, created_at=TODAY + t)
            old = current_record(store, key)
            store = [dataclasses.replace(r, superseded_by=new.id) if r.id == old.id else r for r in store] + [new]
            text = tpl.fill(tpl.UPDATE[rng.integers(len(tpl.UPDATE))], np=u.spec(key).np, v=new.value)
            turns.append(Turn(t, "update", text, key, None, (), frozenset(), new))
    return SessionScript(sid, u.id, tuple(turns))


def make_pilot_session(u: User, rng: np.random.Generator, sid: str) -> tuple[SessionScript, str, str, str]:
    """E0 fixed 8-turn structure (plan §3.9). Returns (script, K, current id, distractor id R)."""
    stale_keys = [k for k in u.changing_keys if stale_records(u.initial_store, k)]
    if not stale_keys:
        raise ValueError("user has no changing key with an old version")
    K = stale_keys[rng.integers(len(stale_keys))]
    trivia = [r for r in u.initial_store if r.category == "trivia"]
    R = trivia[rng.integers(len(trivia))]
    store = list(u.initial_store)
    cur0 = current_record(store, K)
    np_K = u.spec(K).np

    def q(t):
        cur = current_record(store, K)
        stale = tuple(r.value for r in stale_records(store, K))
        return Turn(t, "query", tpl.fill(tpl.QUERY[rng.integers(len(tpl.QUERY))], np=np_K), K, cur.value, stale,
                    frozenset({cur.id}))

    def chat_R(t):
        return Turn(t, "chat", tpl.fill(tpl.CHAT_TOPIC[rng.integers(len(tpl.CHAT_TOPIC))], T=R.topic),
                    topic_key=R.key)

    def chat_plain(t):
        return Turn(t, "chat", tpl.CHAT_PLAIN[rng.integers(len(tpl.CHAT_PLAIN))])

    turns = [q(0), chat_R(1), chat_plain(2)]
    new = u.spares[K][int(rng.integers(len(u.spares[K])))]
    new = dataclasses.replace(new, created_at=TODAY + 3)
    store = [dataclasses.replace(r, superseded_by=new.id) if r.id == cur0.id else r for r in store] + [new]
    turns.append(Turn(3, "update", tpl.fill(tpl.UPDATE[rng.integers(len(tpl.UPDATE))], np=np_K, v=new.value),
                      K, None, (), frozenset(), new))
    turns += [chat_plain(4), q(5), chat_R(6), q(7)]
    return SessionScript(sid, u.id, tuple(turns)), K, cur0.id, R.id


# ---------------------------------------------------------------------------------------------
# validation (plan §3.2)


class WorldInvalid(Exception):
    pass


_DATE_WORDS = {w.lower() for w in MONTHS + MONTH_ABBR + WEEKDAYS}


def validate_world(world: World) -> None:
    # 1. templates never contain date words or {v} outside update templates
    for t in tpl.QUERY + tpl.CHAT_TOPIC + tpl.CHAT_PLAIN:
        bad = set(normalize(t).split()) & (_DATE_WORDS - {"may", "monday"})
        if bad:
            raise WorldInvalid(f"template contains date words {bad}: {t!r}")
    for u in world.users.values():
        # 2. aliases of different values never contain one another (word boundary)
        joined = {v: "|".join(normalize(a) for a in {v, *al}) for v, al in u.aliases.items()}
        for v, al in u.aliases.items():
            for a in {v, *al}:
                na = normalize(a)
                for w, j in joined.items():
                    if w != v and na in j:
                        raise WorldInvalid(f"{u.id}: alias {a!r} of {v!r} occurs in aliases of {w!r}")
        # 3. chat/query texts never mention a value; update texts mention exactly the new one
        for k in u.queryable_keys:
            for q in tpl.QUERY:
                text = tpl.fill(q, np=u.spec(k).np)
                if u.index.values_in(text):
                    raise WorldInvalid(f"{u.id}: query mentions a value: {text!r}")
        for topic in {k.topic for k in u.keys}:
            for c in tpl.CHAT_TOPIC:
                text = tpl.fill(c, T=topic)
                if u.index.values_in(text):
                    raise WorldInvalid(f"{u.id}: chat mentions a value: {text!r}")
        for c in tpl.CHAT_PLAIN:
            if u.index.values_in(c):
                raise WorldInvalid(f"{u.id}: chat mentions a value: {c!r}")
        for k in u.changing_keys:
            for r in u.spares[k]:
                text = tpl.fill(tpl.UPDATE[0], np=u.spec(k).np, v=r.value)
                if u.index.values_in(text) != {r.value}:
                    raise WorldInvalid(f"{u.id}: update text mentions other values: {text!r}")
        # 4. every changing key has enough spare values
        for k in u.changing_keys:
            if len(u.spares[k]) < 3:
                raise WorldInvalid(f"{u.id}: too few spare values for {k}")
    splits = {}
    for u in world.users.values():
        splits.setdefault(u.split, set()).add(u.id)
    if sum(len(s) for s in splits.values()) != len(world.users):
        raise WorldInvalid("split overlap")


def validate_script(world: World, s: SessionScript) -> None:
    """Every query's answer is the latest value of its key at that moment; chat/query never name a value."""
    u = world.user(s.user_id)
    store = list(u.initial_store)
    for turn in s.turns:
        if turn.kind == "query":
            cur = current_record(store, turn.target_key)
            if cur.value != turn.answer_value or turn.relevant_ids != frozenset({cur.id}):
                raise WorldInvalid(f"{s.id} t{turn.idx}: answer is not the current value")
            if u.index.values_in(turn.text):
                raise WorldInvalid(f"{s.id} t{turn.idx}: query names a value")
        elif turn.kind == "chat" and u.index.values_in(turn.text):
            raise WorldInvalid(f"{s.id} t{turn.idx}: chat names a value")
        elif turn.kind == "update":
            store = store + [turn.new_record]


def save_scripts(path: Path, scripts: list[SessionScript]) -> None:
    with open(path, "w") as f:
        for s in scripts:
            f.write(script_to_json(s) + "\n")


def load_scripts(path: Path) -> list[SessionScript]:
    with open(path) as f:
        return [script_from_json(line) for line in f if line.strip()]


def fixed_scripts(world: World, split: str, per_user: int, T: int, seed: int) -> list[SessionScript]:
    """Deterministic evaluation set: the same seed always gives the same scripts."""
    fac = SessionFactory(world, split, seed)
    return [fac.make(T, user=u) for u in fac.users for _ in range(per_user)]


def log_scripts(world: World, n: int, T_choices: list[int], seed: int) -> list[SessionScript]:
    fac = SessionFactory(world, "train", seed)
    rng = np.random.default_rng(seed + 7)
    return [fac.make(int(rng.choice(T_choices))) for _ in range(n)]
