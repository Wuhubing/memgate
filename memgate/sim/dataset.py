"""Log rows -> design matrices for the behaviour model. Uses the logged state snapshots (F1)."""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from memgate.schema import turn_from_dict
from memgate.sim.outcome_features import (
    ANSWER_FEATURES,
    MENTION_FEATURES,
    StateView,
    answer_features,
    mention_candidates,
    mention_features,
    vec,
)

LENGTH_FEATURES = ("n_ctx", "kind_query", "kind_chat", "t_norm")


@dataclass
class Datasets:
    Xa: np.ndarray
    ya: np.ndarray
    sa: np.ndarray          # session id per answer row
    Xm: np.ndarray
    ym: np.ndarray
    sm: np.ndarray
    Xl: np.ndarray
    yl: np.ndarray
    sl: np.ndarray

    def split(self, held_out: set[str]) -> tuple[Datasets, Datasets]:
        def part(keep):
            ma, mm, ml = (np.array([(s in held_out) != keep for s in arr]) for arr in (self.sa, self.sm, self.sl))
            return Datasets(self.Xa[ma], self.ya[ma], self.sa[ma], self.Xm[mm], self.ym[mm], self.sm[mm],
                            self.Xl[ml], self.yl[ml], self.sl[ml])
        return part(True), part(False)


def build_datasets(df: pd.DataFrame, world) -> Datasets:
    Xa, ya, sa, Xm, ym, sm, Xl, yl, sl = [], [], [], [], [], [], [], [], []
    recs = dict(world.records)
    for row in df.itertuples(index=False):
        snap = json.loads(row.state_json)
        turn = turn_from_dict(json.loads(row.turn_json))
        if turn.new_record is not None:
            recs[turn.new_record.id] = turn.new_record
        view = StateView.from_snapshot(snap, recs)
        chosen = json.loads(row.injected_ids)
        mentioned = set(json.loads(row.mentioned_ids))
        user = world.user(row.user_id)
        if turn.kind == "query":
            Xa.append(vec(answer_features(view, turn, chosen), ANSWER_FEATURES))
            ya.append(row.answer)
            sa.append(row.session_id)
        q = world.emb.get(turn.text)
        uv = user.index.values_in(turn.text)
        for r in mention_candidates(view, turn, chosen, uv):
            sim = float(world.emb.get(r.text) @ q)
            Xm.append(vec(mention_features(view, turn, chosen, r, sim), MENTION_FEATURES))
            ym.append(int(r.id in mentioned))
            sm.append(row.session_id)
        eff = view.effective_context(chosen)
        Xl.append([min(len(eff) / 10, 2), float(turn.kind == "query"), float(turn.kind == "chat"), view.t / 20])
        yl.append(row.completion_tokens)
        sl.append(row.session_id)
    f = lambda x, d: np.array(x, dtype=float).reshape(-1, d)  # noqa: E731
    return Datasets(f(Xa, len(ANSWER_FEATURES)), np.array(ya), np.array(sa), f(Xm, len(MENTION_FEATURES)),
                    np.array(ym), np.array(sm), f(Xl, len(LENGTH_FEATURES)), np.array(yl, dtype=float), np.array(sl))
