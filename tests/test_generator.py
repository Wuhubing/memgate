from __future__ import annotations

import numpy as np

from memgate.retrieval.embed import HashEmbedder
from memgate.text import normalize
from memgate.world.generator import (
    SessionFactory,
    World,
    current_record,
    fixed_scripts,
    generate_world,
    load_scripts,
    make_pilot_session,
    save_scripts,
    stale_records,
    validate_script,
)

from .conftest import small_cfg


def test_deterministic(world):
    again = generate_world(small_cfg(), HashEmbedder())
    a = sorted((r.id, r.value, r.text) for r in world.records.values())
    b = sorted((r.id, r.value, r.text) for r in again.records.values())
    assert a == b


def test_splits_disjoint_and_sized(world):
    sizes = {s: len(world.split_users(s)) for s in ("train", "val", "test")}
    assert sizes == {"train": 6, "val": 2, "test": 2}
    ids = [u.id for s in ("train", "val", "test") for u in world.split_users(s)]
    assert len(ids) == len(set(ids)) == 10


def test_values_unique_within_user(world):
    for u in world.users.values():
        recs = list(u.initial_store) + [r for sp in u.spares.values() for r in sp]
        vals = [normalize(r.value) for r in recs]
        assert len(vals) == len(set(vals)), u.id


def test_stale_ratio_close_to_config(world):
    keys = [(u, k) for u in world.users.values() for k in u.changing_keys]
    share = np.mean([bool(stale_records(u.initial_store, k)) for u, k in keys])
    assert 0.25 <= share <= 0.75     # p_stale = 0.5 with 60 keys


def test_sessions_valid_every_split(world):
    for split in ("train", "val", "test"):
        fac = SessionFactory(world, split, seed=1)
        for _ in range(60):
            s = fac.make(12)
            assert s.T == 12
            validate_script(world, s)
            assert world.user(s.user_id).split == split


def test_query_answer_follows_updates(world):
    fac = SessionFactory(world, "train", seed=2, p_kind={"query": 0.5, "chat": 0.0, "update": 0.5})
    seen_update_then_query = False
    for _ in range(40):
        s = fac.make(12)
        store = list(world.user(s.user_id).initial_store)
        updated = set()
        for t in s.turns:
            if t.kind == "update":
                assert t.new_record.value in t.text or normalize(t.new_record.value) in normalize(t.text)
                assert t.new_record.value not in {r.value for r in store}
                store.append(t.new_record)
                updated.add(t.target_key)
            elif t.kind == "query":
                assert t.answer_value == current_record(store, t.target_key).value
                assert t.answer_value not in t.stale_values
                seen_update_then_query |= t.target_key in updated
    assert seen_update_then_query


def test_fixed_scripts_deterministic(world):
    a = fixed_scripts(world, "test", 3, 8, seed=9)
    b = fixed_scripts(world, "test", 3, 8, seed=9)
    assert [s.turns for s in a] == [s.turns for s in b]
    assert len(a) == 3 * len(world.split_users("test"))


def test_pilot_session_shape(world, rng):
    u = next(u for u in world.split_users("train") if any(stale_records(u.initial_store, k) for k in u.changing_keys))
    s, K, cur_id, R_id = make_pilot_session(u, rng, "p0")
    assert s.T == 8
    assert world.records[R_id].key != K
    assert current_record(u.initial_store, K).id == cur_id or any(t.kind == "update" for t in s.turns)


def test_world_and_scripts_roundtrip(world, tmp_path):
    world.save(tmp_path / "w")
    w2 = World.load(tmp_path / "w")
    assert {r.id: r for r in w2.records.values()} == {r.id: r for r in world.records.values()}
    t = "What is the plan?"
    assert np.allclose(w2.emb.get(t), world.emb.get(t))
    scripts = fixed_scripts(world, "val", 2, 6, seed=1)
    save_scripts(tmp_path / "s.jsonl", scripts)
    assert load_scripts(tmp_path / "s.jsonl") == scripts
