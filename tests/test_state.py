from __future__ import annotations

import json

from memgate.env.state import advance_, new_state, snapshot_json
from memgate.harness.scorer import compute_reward
from memgate.schema import SessionScript, Turn
from memgate.world.generator import stale_records

from .conftest import query_turn, user_with_stale


def _script(u, k):
    new = u.spares[k][0]
    return SessionScript("s", u.id, (
        query_turn(u, k, 0),
        Turn(1, "chat", "Tell me a joke."),
        Turn(2, "update", f"Update: {u.spec(k).np} is now {new.value}.", k, new_record=new),
        Turn(3, "chat", "Anything else?"),
    ))


def run(st, reward, inject_at0):
    for t in st.script.turns:
        chosen = inject_at0 if t.idx == 0 else []
        out = compute_reward(st, t, chosen, "na", frozenset(), frozenset(), reward)
        advance_(st, t, chosen, out, "u", "a")


def test_persistent_keeps_context_and_flags_conflict(world, reward):
    u, k = user_with_stale(world)
    old = stale_records(u.initial_store, k)[0]
    st = new_state(_script(u, k), u.initial_store, "persistent")
    run(st, reward, [old])
    assert st.in_context == {old.id: 0}
    assert st.injected_at == {old.id: [0]}
    assert k in st.updated_keys
    assert st.store[-1].id == u.spares[k][0].id
    assert st.t == 4 and st.done and len(st.history) == 4


def test_conflict_flag_set_on_update_turn(world, reward):
    u, k = user_with_stale(world)
    old = stale_records(u.initial_store, k)[0]
    st = new_state(_script(u, k), u.initial_store, "persistent")
    for t in st.script.turns[:3]:
        chosen = [old] if t.idx == 0 else []
        advance_(st, t, chosen, compute_reward(st, t, chosen, "na", frozenset(), frozenset(), reward))
    assert st.conflict_flag
    t = st.script.turns[3]
    advance_(st, t, [], compute_reward(st, t, [], "na", frozenset(), frozenset(), reward))
    assert not st.conflict_flag        # only on the turn of the update


def test_ephemeral_clears_context(world, reward):
    u, k = user_with_stale(world)
    rec = u.initial_store[0]
    st = new_state(_script(u, k), u.initial_store, "ephemeral")
    run(st, reward, [rec])
    assert st.in_context == {}
    assert st.injected_at == {rec.id: [0]}
    assert st.effective_context(["x"]) == {"x"}


def test_snapshot_is_json_and_independent(world, reward):
    u, k = user_with_stale(world)
    st = new_state(_script(u, k), u.initial_store, "persistent")
    snap = json.loads(snapshot_json(st))
    assert snap["t"] == 0 and snap["in_context"] == {}
    run(st, reward, [u.initial_store[0]])
    assert snap["in_context"] == {}           # earlier snapshot not mutated
    assert json.loads(snapshot_json(st))["t"] == 4


def test_fork_is_deep(world, reward):
    u, k = user_with_stale(world)
    st = new_state(_script(u, k), u.initial_store, "persistent")
    f = st.fork()
    run(f, reward, [u.initial_store[0]])
    assert st.t == 0 and st.in_context == {} and len(st.store) == len(u.initial_store)
