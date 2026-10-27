"""Behaviour model fitted on real-LLM logs; the simulator samples from it (plan §3.10).

Parameters are kept as plain numpy arrays so `probs` is a few dot products (the simulator calls
it hundreds of thousands of times).
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from memgate.sim.outcome_features import (
    ANSWER_CLASSES,
    ANSWER_FEATURES,
    MENTION_FEATURES,
    answer_features,
    mention_candidates,
    mention_features,
    vec,
)


@dataclass
class Linear:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray        # (n_classes or 1, n_features)
    intercept: np.ndarray
    lo: np.ndarray | None = None   # feature range seen in the logs; inputs are clipped to it (F3)
    hi: np.ndarray | None = None

    def logits(self, x: np.ndarray) -> np.ndarray:
        if self.lo is not None:
            x = np.clip(x, self.lo, self.hi)
        z = (x - self.mean) / self.scale
        return z @ self.coef.T + self.intercept

    @classmethod
    def from_sklearn(cls, scaler, lr, X: np.ndarray | None = None) -> Linear:
        lo = X.min(axis=0) if X is not None else None
        hi = X.max(axis=0) if X is not None else None
        return cls(scaler.mean_.copy(), np.where(scaler.scale_ > 0, scaler.scale_, 1.0), lr.coef_.copy(),
                   lr.intercept_.copy(), lo, hi)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


@dataclass
class BehaviorModel:
    answer: Linear
    answer_classes: tuple[str, ...]
    mention: Linear
    length_coef: np.ndarray
    meta: dict = field(default_factory=dict)

    # ---- fitting -----------------------------------------------------------------------------
    @classmethod
    def fit(cls, Xa: np.ndarray, ya: np.ndarray, Xm: np.ndarray, ym: np.ndarray, Xl: np.ndarray | None = None,
            yl: np.ndarray | None = None, C: float = 1.0, seed: int = 0) -> BehaviorModel:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        sa = StandardScaler().fit(Xa)
        classes = sorted(set(ya.tolist()), key=ANSWER_CLASSES.index)
        if len(classes) < 2:
            raise ValueError("answer model needs at least two outcome classes in the logs")
        la = LogisticRegression(C=C, max_iter=2000, random_state=seed).fit(sa.transform(Xa), ya)
        lin_a = Linear.from_sklearn(sa, la, Xa)
        if len(la.classes_) == 2:   # sklearn returns one row for binary problems
            lin_a = Linear(lin_a.mean, lin_a.scale, np.vstack([-lin_a.coef, lin_a.coef]) / 2,
                           np.array([-lin_a.intercept[0], lin_a.intercept[0]]) / 2, lin_a.lo, lin_a.hi)
        sm = StandardScaler().fit(Xm)
        if len(set(ym.tolist())) < 2:
            raise ValueError("mention model needs both mentioned and not-mentioned samples")
        lm = LogisticRegression(C=C, max_iter=2000, random_state=seed).fit(sm.transform(Xm), ym)
        if Xl is not None and len(Xl):
            A = np.c_[np.ones(len(Xl)), Xl]
            length_coef = np.linalg.lstsq(A, yl, rcond=None)[0]
        else:
            length_coef = np.zeros(1)
        return cls(lin_a, tuple(str(c) for c in la.classes_), Linear.from_sklearn(sm, lm, Xm), length_coef,
                   {"C": C, "n_answer": len(ya), "n_mention": len(ym)})

    # ---- prediction --------------------------------------------------------------------------
    def answer_probs_x(self, X: np.ndarray) -> np.ndarray:
        """Probabilities over ANSWER_CLASSES (missing classes get 0)."""
        p = _softmax(np.atleast_2d(self.answer.logits(X)))
        out = np.zeros((p.shape[0], len(ANSWER_CLASSES)))
        for j, c in enumerate(self.answer_classes):
            out[:, ANSWER_CLASSES.index(c)] = p[:, j]
        return out

    def mention_probs_x(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.zeros(0)
        z = self.mention.logits(np.atleast_2d(X))[:, 0]
        return 1.0 / (1.0 + np.exp(-z))

    def probs(self, st, turn, chosen_ids: list[str], sims: dict[str, float], user_values: set[str]
              ) -> tuple[dict[str, float], dict[str, float]]:
        """(P(answer class), P(mentioned) per mentionable record). Used by DP-style planners and sim."""
        if turn.kind == "query":
            pa = self.answer_probs_x(vec(answer_features(st, turn, chosen_ids), ANSWER_FEATURES)[None])[0]
            if not turn.stale_values:   # no old value exists, so the reply cannot use one
                pa[ANSWER_CLASSES.index("stale")] = 0.0
                pa = pa / pa.sum()
            p_ans = dict(zip(ANSWER_CLASSES, pa.tolist(), strict=True))
        else:
            p_ans = {"na": 1.0}
        cands = mention_candidates(st, turn, chosen_ids, user_values)
        if cands:
            X = np.stack([vec(mention_features(st, turn, chosen_ids, r, sims.get(r.id, 0.0)), MENTION_FEATURES)
                          for r in cands])
            pm = self.mention_probs_x(X)
            p_m = {r.id: float(p) for r, p in zip(cands, pm, strict=True)}
        else:
            p_m = {}
        return p_ans, p_m

    def sample(self, st, turn, chosen_ids: list[str], sims: dict[str, float], user_values: set[str],
               rng: np.random.Generator) -> tuple[str, frozenset[str]]:
        p_ans, p_m = self.probs(st, turn, chosen_ids, sims, user_values)
        classes = list(p_ans)
        answer = classes[rng.choice(len(classes), p=np.array([p_ans[c] for c in classes]))]
        mentioned = {i for i, p in p_m.items() if rng.random() < p}
        # the reply also names the value it answered with; these are not intrusions
        by_val = {r.value: r.id for r in st.store if r.key == turn.target_key}
        if answer == "correct" and turn.answer_value in by_val:
            mentioned.add(by_val[turn.answer_value])
        elif answer == "stale":
            olds = [by_val[v] for v in turn.stale_values if v in by_val]
            eff = st.effective_context(chosen_ids)
            vis = [i for i in olds if i in eff] or olds
            if vis:
                mentioned.add(vis[int(rng.integers(len(vis)))])
        return answer, frozenset(mentioned)

    # ---- io ----------------------------------------------------------------------------------
    def save(self, path: Path) -> None:
        import sklearn
        self.meta["sklearn"] = sklearn.__version__
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> BehaviorModel:
        with open(path, "rb") as f:
            return pickle.load(f)
