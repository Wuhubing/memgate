"""Scorer: string matching of values and aliases in replies (plan §3.6). At least 30 hand-written replies."""
from __future__ import annotations

import pytest

from memgate.env.state import new_state
from memgate.harness.scorer import classify_answer, intrusion_ids, mentioned_ids, score_response
from memgate.schema import SessionScript, Turn
from memgate.world.generator import current_record, stale_records

from .conftest import query_turn, user_with_stale


@pytest.fixture(scope="module")
def setup(world):
    u, k = user_with_stale(world)
    cur = current_record(u.initial_store, k)
    old = stale_records(u.initial_store, k)[0]
    other = next(r for r in u.initial_store if r.key.split(".")[0] != k.split(".")[0] and r.key != k)
    return u, k, cur, old, other


def forms(u, value):
    return sorted({value, *u.aliases[value]})


# (reply template, expected answer, expected hedged); {cur} / {old} / {oth} are filled with surface forms
CASES = [
    ("It is {cur}.", "correct", False),
    ("The deadline is {cur}, as far as I know.", "correct", False),
    ("{cur}", "correct", False),
    ("It used to be {old}, but now it is {cur}.", "correct", True),
    ("It was moved from {old} to {cur}.", "correct", True),
    ("It is {old}.", "stale", False),
    ("I believe it's {old}!", "stale", False),
    ("({old})", "stale", False),
    ("I don't have that information.", "neither", False),
    ("Could you remind me?", "neither", False),
    ("It is {oth}.", "neither", False),
    ("IT IS {cur_upper}.", "correct", False),
    ("it is {cur_lower}", "correct", False),
    ("Sure - {cur}; let me know if that changes.", "correct", False),
    ("Either {old} or {cur}, I'm not sure.", "correct", True),
    ("Your contact {oth} mentioned it is {cur}.", "correct", False),
    ("Old: {old}. New: {cur}.", "correct", True),
    ("\"{old}\"", "stale", False),
    ("{old}, I think, but {oth} may know more.", "stale", False),
    ("Nothing on file.", "neither", False),
    ("**{cur}**", "correct", False),
    ("- Deadline: *{old}*", "stale", False),
    ("`{cur}` (was `{old}`)", "correct", True),
]


def fill(tpl, cur, old, oth):
    return tpl.format(cur=cur, old=old, oth=oth, cur_upper=cur.upper(), cur_lower=cur.lower())


def all_cases(setup):
    u, k, cur, old, other = setup
    out = []
    for tpl, ans, hedged in CASES:
        out.append((fill(tpl, cur.value, old.value, other.value), ans, hedged))
    # every surface form of the current and the old value is recognised
    for f in forms(u, cur.value):
        out.append((f"It is {f}.", "correct", False))
    for f in forms(u, old.value):
        out.append((f"It is {f}.", "stale", False))
    return out


def test_at_least_30_replies(setup):
    assert len(all_cases(setup)) >= 30


def test_classify(setup, reward):
    u, k, cur, old, other = setup
    turn = query_turn(u, k)
    st = new_state(SessionScript("s", u.id, (turn,)), u.initial_store, "persistent")
    for text, ans, hedged in all_cases(setup):
        out = score_response(st, turn, [], text, reward, u.index)
        assert (out.answer, out.hedged) == (ans, hedged), text


def test_word_boundaries(setup):
    u, k, cur, old, other = setup
    core = other.value.split()[-1]
    assert not u.index.values_in(f"{core}xyz and pre{core}")
    assert u.index.values_in(f"... {other.value}!") == {other.value}


def test_date_prefix_is_not_a_match(world):
    """'Nov 1' must not match 'Nov 14' (and the reverse)."""
    n = 0
    for u in world.users.values():
        for v, al in u.aliases.items():
            for a in al:
                parts = a.split()
                if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 2:
                    short = f"{parts[0]} {parts[1][0]}"
                    assert v not in u.index.values_in(f"It is {short}."), (a, short)
                    n += 1
    assert n > 0


def test_non_query_turn_is_na(setup):
    u, k, cur, old, other = setup
    t = Turn(0, "chat", "Tell me a joke.")
    assert classify_answer(t, {cur.value}) == ("na", False)


def test_mentions_and_intrusions(setup, reward):
    u, k, cur, old, other = setup
    turn = query_turn(u, k)
    st = new_state(SessionScript("s", u.id, (turn,)), u.initial_store, "persistent")
    vals = u.index.values_in(f"It is {cur.value}, before it was {old.value}; ask {other.value}.")
    ment = mentioned_ids(st, turn, vals)
    assert {cur.id, old.id, other.id} <= ment
    intr = intrusion_ids(st, turn, ment, u.index)
    assert other.id in intr                      # unrelated record
    assert cur.id not in intr and old.id not in intr   # the turn's own key is never an intrusion


def test_value_the_user_just_said_is_not_an_intrusion(setup):
    u, k, cur, old, other = setup
    turn = Turn(0, "chat", f"I met {other.value} today, any tips?")
    st = new_state(SessionScript("s", u.id, (turn,)), u.initial_store, "persistent")
    ment = mentioned_ids(st, turn, {other.value})
    assert other.id in ment
    assert other.id not in intrusion_ids(st, turn, ment, u.index)


def test_update_turn_new_record_counted(setup):
    u, k, cur, old, other = setup
    new = u.spares[k][0]
    turn = Turn(0, "update", f"Update: {u.spec(k).np} is now {new.value}.", k, new_record=new)
    st = new_state(SessionScript("s", u.id, (turn,)), u.initial_store, "persistent")
    ment = mentioned_ids(st, turn, {new.value})
    assert ment == {new.id}
    assert not intrusion_ids(st, turn, ment, u.index)
