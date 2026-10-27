"""compute_reward: the one reward function both backends share (plan §1.6)."""
from __future__ import annotations

import pytest

from memgate.config import load_reward
from memgate.env.state import advance_, new_state
from memgate.harness.scorer import compute_reward
from memgate.schema import SessionScript, Turn

from .conftest import query_turn, user_with_stale


@pytest.fixture
def st_turn(world):
    u, k = user_with_stale(world)
    turn = query_turn(u, k)
    turns = tuple(Turn(i, "chat", "Tell me a joke.") if i else turn for i in range(4))
    return u, new_state(SessionScript("s", u.id, turns), u.initial_store, "persistent"), turn


def test_terms(st_turn):
    u, st, turn = st_turn
    cfg = load_reward("R_full")
    r1, r2 = u.initial_store[0], u.initial_store[-1]
    out = compute_reward(st, turn, [r1, r2], "stale", frozenset({r2.id}), frozenset({r2.id}), cfg)
    assert out.r_quality == 0 and out.r_stale == -cfg.delta
    assert out.r_intrusion == pytest.approx(-cfg.alpha * cfg.beta ** 0)
    assert out.r_tokens == pytest.approx(-cfg.lam * (r1.tokens + r2.tokens))
    assert out.reward == pytest.approx(out.r_quality + out.r_stale + out.r_intrusion + out.r_tokens)
    ok = compute_reward(st, turn, [], "correct", frozenset(), frozenset(), cfg)
    assert ok.reward == 1.0


def test_intrusion_penalty_grows_with_earlier_intrusions(st_turn):
    u, st, turn = st_turn
    cfg = load_reward("R_full")
    r = u.initial_store[-1]
    pens = []
    for _ in range(3):
        out = compute_reward(st, st.turn, [], "na" if st.turn.kind != "query" else "neither",
                             frozenset({r.id}), frozenset({r.id}), cfg)
        pens.append(out.r_intrusion)
        advance_(st, st.turn, [], out)
    assert pens == pytest.approx([-cfg.alpha * cfg.beta ** n for n in range(3)])
    assert st.intrusions[r.id] == 3


def test_relevant_mentions_do_not_count_toward_n_m(st_turn):
    """n_m counts earlier *unrelated* mentions only, not all mentions (§1.6)."""
    u, st, turn = st_turn
    cfg = load_reward("R_full")
    cur = next(iter(turn.relevant_ids))
    out = compute_reward(st, turn, [], "correct", frozenset({cur}), frozenset(), cfg)
    advance_(st, turn, [], out)
    assert st.mentions[cur] == 1 and st.intrusions.get(cur, 0) == 0


def test_behavior_variant_has_flat_intrusions_and_no_tokens(st_turn):
    u, st, turn = st_turn
    cfg = load_reward("R_behavior")
    assert cfg.lam == 0 and cfg.beta == 1
    st.intrusions[u.initial_store[-1].id] = 5
    out = compute_reward(st, turn, list(u.initial_store[:3]), "neither", frozenset({u.initial_store[-1].id}),
                         frozenset({u.initial_store[-1].id}), cfg)
    assert out.r_tokens == 0 and out.r_intrusion == pytest.approx(-cfg.alpha)


def test_token_cost_uses_effective_context(world):
    u, k = user_with_stale(world)
    cfg = load_reward("R_full")
    turns = (query_turn(u, k, 0), query_turn(u, k, 1))
    a, b = u.initial_store[0], u.initial_store[1]
    for mode, expect in (("persistent", a.tokens + b.tokens), ("ephemeral", b.tokens)):
        st = new_state(SessionScript("s", u.id, turns), u.initial_store, mode)
        out = compute_reward(st, turns[0], [a], "correct", frozenset(), frozenset(), cfg)
        advance_(st, turns[0], [a], out)
        out = compute_reward(st, turns[1], [b], "correct", frozenset(), frozenset(), cfg)
        assert out.r_tokens == pytest.approx(-cfg.lam * expect), mode
