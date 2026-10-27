"""R10: every figure and table of the report from saved results (plan §3.13). Recomputes nothing.

Inputs that do not exist yet are skipped with a note, so this can run at any stage.
"""
from __future__ import annotations

import argparse
import json
import shutil

import pandas as pd

from memgate.analysis.plots import (
    case_figure,
    fig1_pilot,
    fig2_gamma,
    fig3_delta,
    fig4_gamma_T,
    fig5_sim_vs_live,
    table1,
)
from memgate.analysis.stats import PRIMARY_CONTRASTS, paired_diff, rank_agreement, sim_contrasts
from memgate.experiment import resolve
from memgate.harness.runner import read_logs


def skip(what: str, path) -> None:
    print(f"[figures] skip {what}: {path} not found")


def main_rows(ev: pd.DataFrame, T_main: int) -> pd.DataFrame:
    """The default setting: T = T_main and not the E3 T-sweep (its seeds would mix with E2's)."""
    return ev[(ev["T"] == T_main) & (~ev["sweep"].astype(str).str.startswith("e3_full"))]


def live_contrasts(live: pd.DataFrame) -> pd.DataFrame:
    """J7: the real-LLM counterparts of the planning-gain contrasts (rankings only, not Delta itself)."""
    pairs = [(c.name, c.a, c.b) for c in PRIMARY_CONTRASTS if c.source == "sim"]
    pairs.append(("amortized_minus_greedy", {"method": "amortized_greedy"}, {"method": "model_greedy"}))
    rows = []
    for mode, sub in live.groupby("mode"):
        for name, a, b in pairs:
            r = paired_diff(sub, a, b)
            if r["n_sessions"]:
                rows.append({"contrast": name, "mode": mode, **r})
    return pd.DataFrame(rows)


def case_study(logs_root, live: pd.DataFrame, out) -> str | None:
    """The test session where DQN gamma=0.9 and gamma=0 differ most on the real LLM."""
    def run_of(gamma):
        r = live[(live.method == "dqn") & (live.gamma == gamma) & (~live.mask_history.astype(bool))]
        return r["run"].iloc[0] if len(r) else None

    ra, rb = run_of(0.9), run_of(0.0)
    if ra is None or rb is None:
        return None
    a, b = read_logs(logs_root / ra), read_logs(logs_root / rb)
    ret = pd.concat([a.groupby("session_id").reward.sum().rename("a"),
                     b.groupby("session_id").reward.sum().rename("b")], axis=1, join="inner")
    sid = (ret.a - ret.b).idxmax()

    def rows(df):
        g = df[df.session_id == sid].copy()
        g["injected"] = g.injected_ids.map(lambda s: ", ".join(json.loads(s)))
        return g[["t", "kind", "injected", "reward"]]

    case_figure(rows(a), rows(b), "DQN γ=0.9", "DQN γ=0", out)
    return str(sid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", default="results/e0")
    ap.add_argument("--dp", default="runs/e3")
    ap.add_argument("--sim-eval", default="results/sim_eval.parquet")
    ap.add_argument("--live-eval", default="results/live_eval.parquet")
    ap.add_argument("--live-logs", default="results/live_logs")
    ap.add_argument("--T-main", type=int, default=12)
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    out = resolve(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary: dict = {}

    p = resolve(a.pilot) / "summary.csv"
    if p.exists():
        s = pd.read_csv(p)
        fig1_pilot(s[s["mode"] == "persistent"], out / "fig1_pilot.png")
        if (s["mode"] == "ephemeral").any():
            fig1_pilot(s[s["mode"] == "ephemeral"], out / "fig1_pilot_ephemeral.png")
        g0 = resolve(a.pilot) / "g0.json"
        if g0.exists():
            summary["e0"] = json.loads(g0.read_text())
    else:
        skip("figure 1", p)

    d = resolve(a.dp)
    if (d / "contrasts.csv").exists():
        con = pd.read_csv(d / "contrasts.csv")
        fig3_delta(con, out / "fig3_delta.png")
        shutil.copy(d / "contrasts.csv", out / "contrasts_dp.csv")
        shutil.copy(d / "delta.csv", out / "delta_dp.csv")
        summary["dp"] = con.pivot_table(index=["mode", "T", "p_stale"], columns="contrast", values="value"
                                        ).round(4).reset_index().to_dict("records")
    else:
        skip("figure 3", d / "contrasts.csv")

    sim = None
    p = resolve(a.sim_eval)
    if p.exists():
        sim = pd.read_parquet(p)
        main = main_rows(sim, a.T_main)
        table1(main, out / "table1.csv")
        fig2_gamma(main, out / "fig2_gamma.png")
        sc = sim_contrasts(sim[~sim["sweep"].astype(str).str.startswith("e3_full")])
        sc.to_csv(out / "contrasts_sim.csv", index=False)
        summary["sim_contrasts"] = sc.round(4).to_dict("records")
        e3 = sim[sim["sweep"].astype(str).str.startswith("e3_full")]
        if len(e3):
            fig4_gamma_T(e3, out / "fig4_gamma_T.png")
        else:
            skip("figure 4", "e3_full_* runs in sim_eval")
    else:
        skip("table 1 / figures 2, 4", p)

    p = resolve(a.live_eval)
    if p.exists():
        live = pd.read_parquet(p)
        table1(live, out / "table_live.csv")
        lc = live_contrasts(live)
        lc.to_csv(out / "contrasts_live.csv", index=False)
        summary["live_contrasts"] = lc.round(4).to_dict("records")
        if sim is not None:
            rho, both = rank_agreement(sim, live)
            both.to_csv(out / "rank_sim_vs_live.csv", index=False)
            fig5_sim_vs_live(both, rho, out / "fig5_sim_vs_live.png")
            summary["spearman_sim_live"] = rho
        sid = case_study(resolve(a.live_logs), live, out / "fig_case.png")
        summary["case_session"] = sid
        if sid is None:
            skip("case figure", "DQN γ=0.9 and γ=0 live runs")
    else:
        skip("live table / figure 5 / case figure", p)

    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"[figures] wrote {out}")


if __name__ == "__main__":
    main()
