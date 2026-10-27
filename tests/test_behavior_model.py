from __future__ import annotations

import numpy as np
import pytest

from memgate.env.state import new_state
from memgate.schema import SessionScript
from memgate.sim.behavior_model import BehaviorModel
from memgate.sim.dataset import build_datasets
from memgate.sim.outcome_features import ANSWER_CLASSES, ANSWER_FEATURES, MENTION_FEATURES

from .conftest import query_turn, user_with_stale


def synthetic(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    Xa = rng.normal(size=(n, len(ANSWER_FEATURES)))
    logits = np.stack([2 * Xa[:, 0], 2 * Xa[:, 2], np.zeros(n)], axis=1)   # cur_vis -> correct, stale -> stale
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    ya = np.array([ANSWER_CLASSES[rng.choice(3, p=pi)] for pi in p])
    Xm = rng.normal(size=(n, len(MENTION_FEATURES)))
    ym = (rng.random(n) < 1 / (1 + np.exp(-(1.5 * Xm[:, 0] - 1)))).astype(int)
    Xl = rng.normal(size=(n, 4))
    yl = 30 + 5 * Xl[:, 0] + rng.normal(size=n)
    return Xa, ya, Xm, ym, Xl, yl


def test_recovers_effects():
    m = BehaviorModel.fit(*synthetic())
    x = np.zeros((2, len(ANSWER_FEATURES)))
    x[1, 0] = 2.0
    p = m.answer_probs_x(x)
    assert np.allclose(p.sum(1), 1)
    assert p[1, ANSWER_CLASSES.index("correct")] > p[0, ANSWER_CLASSES.index("correct")] + 0.3
    xm = np.zeros((2, len(MENTION_FEATURES)))
    xm[1, 0] = 2.0
    pm = m.mention_probs_x(xm)
    assert pm[1] > pm[0] + 0.3
    assert m.length_coef[1] == pytest.approx(5, abs=0.3)


def test_inputs_clipped_to_logged_range():
    Xa, ya, Xm, ym, Xl, yl = synthetic()
    m = BehaviorModel.fit(Xa, ya, Xm, ym, Xl, yl)
    far = np.zeros((1, len(ANSWER_FEATURES)))
    far[0, 0] = 1e6
    edge = np.zeros((1, len(ANSWER_FEATURES)))
    edge[0, 0] = Xa[:, 0].max()
    assert np.allclose(m.answer_probs_x(far), m.answer_probs_x(edge))


def test_binary_answer_classes_handled():
    Xa, ya, Xm, ym, Xl, yl = synthetic()
    ya = np.where(ya == "stale", "neither", ya)
    m = BehaviorModel.fit(Xa, ya, Xm, ym, Xl, yl)
    p = m.answer_probs_x(np.zeros((3, len(ANSWER_FEATURES))))
    assert np.allclose(p.sum(1), 1) and np.all(p[:, ANSWER_CLASSES.index("stale")] == 0)


def test_needs_two_classes():
    Xa, ya, Xm, ym, Xl, yl = synthetic()
    with pytest.raises(ValueError):
        BehaviorModel.fit(Xa, np.full(len(ya), "correct"), Xm, ym)


def test_save_load(tmp_path):
    m = BehaviorModel.fit(*synthetic())
    m.save(tmp_path / "m.pkl")
    m2 = BehaviorModel.load(tmp_path / "m.pkl")
    x = np.random.default_rng(1).normal(size=(5, len(ANSWER_FEATURES)))
    assert np.allclose(m.answer_probs_x(x), m2.answer_probs_x(x))


def test_dataset_from_logs(world, fake_logs):
    ds = build_datasets(fake_logs, world)
    assert len(ds.Xa) == (fake_logs.kind == "query").sum()
    assert ds.Xa.shape[1] == len(ANSWER_FEATURES) and ds.Xm.shape[1] == len(MENTION_FEATURES)
    assert set(ds.ya) <= set(ANSWER_CLASSES) and 0 < ds.ym.mean() < 1
    held = set(list(fake_logs.session_id.unique())[:20])
    tr, te = ds.split(held)
    assert len(tr.Xa) + len(te.Xa) == len(ds.Xa)
    assert set(te.sa) <= held and not set(tr.sa) & held


def test_no_stale_class_without_old_values(world, bm, rng):
    u, k = user_with_stale(world)
    t = query_turn(u, k)
    t = t.__class__(**{**t.__dict__, "stale_values": ()})
    st = new_state(SessionScript("s", u.id, (t,)), u.initial_store, "persistent")
    p_ans, _ = bm.probs(st, t, [], {}, set())
    assert p_ans["stale"] == 0 and sum(p_ans.values()) == pytest.approx(1)
    for _ in range(50):
        ans, ment = bm.sample(st, t, [], {}, set(), rng)
        assert ans != "stale"
        if ans == "correct":
            assert next(iter(t.relevant_ids)) in ment     # a correct reply names the current value
