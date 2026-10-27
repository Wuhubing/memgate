"""Statistics for the report (plan §3.13). Only PRIMARY_CONTRASTS may back the main conclusions."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

COMPONENT_COLS = ("quality", "stale", "intrusion", "tokens")


def bootstrap_ci(x, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean and percentile CI, resampling sessions (plan: bootstrap by session)."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_diff(df: pd.DataFrame, a: dict, b: dict, value: str = "return", n_boot: int = 1000) -> dict:
    """Mean of (a - b) over the sessions both were evaluated on. `a`/`b` are column filters.

    With several seeds per method, sessions are first averaged within each method over seeds.
    """
    def pick(f):
        m = np.ones(len(df), dtype=bool)
        for k, v in f.items():
            m &= (df[k] == v).to_numpy()
        return df[m].groupby("session_id")[value].mean()

    sa, sb = pick(a), pick(b)
    common = sa.index.intersection(sb.index)
    d = (sa.loc[common] - sb.loc[common]).to_numpy()
    mean, lo, hi = bootstrap_ci(d, n_boot)
    return {"diff": mean, "lo": lo, "hi": hi, "n_sessions": len(common)}


def seed_summary(df: pd.DataFrame, by: list[str], value: str = "return") -> pd.DataFrame:
    """Per seed: mean over sessions; then mean +/- std across seeds (plan §3.13)."""
    per_seed = df.groupby([*by, "seed"])[value].mean().reset_index()
    g = per_seed.groupby(by)[value]
    return pd.DataFrame({"mean": g.mean(), "std": g.std(ddof=1).fillna(0.0), "n_seeds": g.size()}).reset_index()


def session_returns(turns: pd.DataFrame) -> pd.DataFrame:
    """Per-turn log rows -> one row per session with the undiscounted return and its parts."""
    g = turns.groupby("session_id")
    out = pd.DataFrame({
        "return": g["reward"].sum(), "quality": g["r_quality"].sum(), "stale": g["r_stale"].sum(),
        "intrusion": g["r_intrusion"].sum(), "tokens": g["r_tokens"].sum(),
    })
    return out.reset_index()


@dataclass(frozen=True)
class Contrast:
    name: str
    description: str
    source: str            # "dp" (exact, small instance) or "sim" (full-scale evaluation)
    a: dict
    b: dict


# The only comparisons the report body may cite (plan §3.13, J5, J6).
PRIMARY_CONTRASTS: tuple[Contrast, ...] = (
    Contrast("delta", "V^DP - V^greedy: pure myopia loss, split by reward component", "dp",
             {"policy": "dp", "reward": "R_full"}, {"policy": "greedy", "reward": "R_full"}),
    Contrast("delta_amort", "V^DP - V^amortized-greedy: loss left after amortising token cost", "dp",
             {"policy": "dp", "reward": "R_full"}, {"policy": "amortized_greedy", "reward": "R_full"}),
    Contrast("delta_behavior", "Delta re-solved under R_behavior (lambda=0, beta=1): LLM behaviour only", "dp",
             {"policy": "dp", "reward": "R_behavior"}, {"policy": "greedy", "reward": "R_behavior"}),
    Contrast("dqn_gamma", "DQN gamma=0.9 - DQN gamma=0, full reward", "sim",
             {"method": "dqn", "gamma": 0.9, "reward": "R_full", "mask_history": False},
             {"method": "dqn", "gamma": 0.0, "reward": "R_full", "mask_history": False}),
    Contrast("dqn_gamma_behavior", "DQN gamma=0.9 - DQN gamma=0 under R_behavior", "sim",
             {"method": "dqn", "gamma": 0.9, "reward": "R_behavior", "mask_history": False},
             {"method": "dqn", "gamma": 0.0, "reward": "R_behavior", "mask_history": False}),
    Contrast("h3_history", "DQN gamma=0.9 - same without history features (H3)", "sim",
             {"method": "dqn", "gamma": 0.9, "reward": "R_full", "mask_history": False},
             {"method": "dqn", "gamma": 0.9, "reward": "R_full", "mask_history": True}),
)
# H4 is every contrast above computed separately for mode=persistent and mode=ephemeral.

REFERENCE_METHODS = ("always", "never", "threshold", "linucb", "linucb_risk", "model_greedy")


def dp_contrasts(delta: pd.DataFrame) -> pd.DataFrame:
    """delta.csv rows (T, p_stale, mode, reward, policy, total, components) -> primary DP contrasts."""
    rows = []
    keys = ["T", "p_stale", "mode"]
    for c in PRIMARY_CONTRASTS:
        if c.source != "dp":
            continue
        A = delta[(delta.policy == c.a["policy"]) & (delta.reward == c.a["reward"])].set_index(keys)
        B = delta[(delta.policy == c.b["policy"]) & (delta.reward == c.b["reward"])].set_index(keys)
        common = A.index.intersection(B.index)
        for k in common:
            r = {"contrast": c.name, **dict(zip(keys, k, strict=True)), "value": A.loc[k, "total"] - B.loc[k, "total"]}
            for comp in COMPONENT_COLS:
                r[comp] = A.loc[k, comp] - B.loc[k, comp]
            rows.append(r)
    return pd.DataFrame(rows)


def sim_contrasts(ev: pd.DataFrame) -> pd.DataFrame:
    """Primary sim contrasts per (mode, T); sessions are paired across the two methods."""
    rows = []
    keys = ["mode", "T"] if "T" in ev else ["mode"]
    for c in PRIMARY_CONTRASTS:
        if c.source != "sim":
            continue
        for k, sub in ev.groupby(keys):
            k = k if isinstance(k, tuple) else (k,)
            r = paired_diff(sub, c.a, c.b)
            if r["n_sessions"] == 0:
                continue
            rows.append({"contrast": c.name, **dict(zip(keys, k, strict=True)), **r})
    return pd.DataFrame(rows)


def rank_agreement(sim: pd.DataFrame, live: pd.DataFrame, by: str = "label") -> tuple[float, pd.DataFrame]:
    """Spearman correlation of per-method mean return, simulator vs real LLM (G5, J3).

    The simulator side is restricted to the runs the live evaluation used.
    """
    from scipy.stats import spearmanr
    s = sim[sim["run"].isin(set(live["run"]))].groupby(by)["return"].mean().rename("sim")
    v = live.groupby(by)["return"].mean().rename("live")
    both = pd.concat([s, v], axis=1, join="inner").reset_index()
    rho = float(spearmanr(both["sim"], both["live"]).statistic) if len(both) >= 3 else float("nan")
    return rho, both
