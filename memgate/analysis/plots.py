"""Figures 1-3, Table 1 and the case figure. Reads results only; recomputes nothing (plan §3.13)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from memgate.analysis.stats import seed_summary  # noqa: E402


def fig1_pilot(summary: pd.DataFrame, out: Path) -> None:
    """E0: R mention rate and stale-use rate by turn and condition, with CIs."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for cond, g in summary.groupby("condition"):
        r = g[g.metric == "R_mention"].sort_values("t")
        axes[0].errorbar(r.t, r["mean"], yerr=[r["mean"] - r.lo, r.hi - r["mean"]], label=cond, marker="o",
                         capsize=2)
        s = g[g.metric == "stale_used"].sort_values("t")
        axes[1].errorbar(s.t, s["mean"], yerr=[s["mean"] - s.lo, s.hi - s["mean"]], label=cond, marker="o",
                         capsize=2)
    axes[0].set(title="Distractor R mentioned", xlabel="turn", ylabel="rate")
    axes[1].set(title="Old value used (query turns)", xlabel="turn", ylabel="rate")
    axes[1].axvline(3, color="grey", ls=":", lw=1)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def table1(ev: pd.DataFrame, out: Path) -> pd.DataFrame:
    cols = ["return", "quality", "stale", "intrusion", "tokens", "acc", "stale_rate", "n_intrusions",
            "mem_tokens", "hedged_rate"]
    by = ["mode", "reward", "label"]
    tabs = []
    for c in cols:
        if c in ev:
            s = seed_summary(ev, by, c).rename(columns={"mean": c, "std": f"{c}_sd"})
            tabs.append(s.set_index(by)[[c, f"{c}_sd"]])
    t = pd.concat(tabs, axis=1).reset_index()
    t.to_csv(out, index=False)
    return t


def fig2_gamma(ev: pd.DataFrame, out: Path) -> None:
    d = ev[(ev.method == "dqn") & (~ev.mask_history.astype(bool))]
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for (mode, reward), g in d.groupby(["mode", "reward"]):
        s = seed_summary(g, ["gamma"]).sort_values("gamma")
        ax.errorbar(s.gamma, s["mean"], yerr=s["std"], marker="o", capsize=2, label=f"{mode} / {reward}")
    ax.set(xlabel="gamma", ylabel="session return (undiscounted)", title="Return vs gamma (DQN)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig3_delta(contrasts: pd.DataFrame, out: Path) -> None:
    """Delta, Delta_amort and behaviour-only Delta vs T (one panel per mode), and vs p_stale."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    ps = sorted(contrasts.p_stale.unique()) if len(contrasts) else [0.5]
    p_default = min(ps, key=lambda p: abs(p - 0.5))      # the main world's stale ratio
    for (name, mode), g in contrasts.groupby(["contrast", "mode"]):
        a = g[g.p_stale == p_default].sort_values("T")
        axes[0].plot(a["T"], a["value"], marker="o", ls="-" if mode == "persistent" else "--", label=f"{name} ({mode})")
        T_max = g["T"].max()
        b = g[g["T"] == T_max].sort_values("p_stale")
        axes[1].plot(b.p_stale, b["value"], marker="o", ls="-" if mode == "persistent" else "--",
                     label=f"{name} ({mode})")
    axes[0].set(xlabel="session length T", ylabel="value gap", title=f"Myopia loss vs T (p_stale={p_default})")
    axes[1].set(xlabel="stale ratio", ylabel="value gap", title="Myopia loss vs stale ratio (largest T)")
    for ax in axes:
        ax.axhline(0, color="grey", lw=0.8)
    axes[0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def case_figure(rows_a: pd.DataFrame, rows_b: pd.DataFrame, label_a: str, label_b: str, out: Path) -> None:
    """Same session, two policies: what each injected per turn. Rows need t, kind, injected, reward."""
    a = rows_a.sort_values("t").reset_index(drop=True)
    b = rows_b.sort_values("t").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 0.45 * (len(a) + 2)))
    ax.axis("off")
    cell = [[ra.t, ra.kind, ra.injected or "-", f"{ra.reward:.2f}", rb.injected or "-", f"{rb.reward:.2f}"]
            for ra, rb in zip(a.itertuples(), b.itertuples(), strict=False)]
    cell.append(["", "total", "", f"{a.reward.sum():.2f}", "", f"{b.reward.sum():.2f}"])
    tab = ax.table(cellText=cell, colLabels=["t", "kind", f"{label_a}: injected", "r", f"{label_b}: injected", "r"],
                   loc="center", cellLoc="left")
    tab.auto_set_font_size(False)
    tab.set_fontsize(7)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig4_gamma_T(ev: pd.DataFrame, out: Path) -> None:
    """E3 at full scale: DQN gamma=0.9 minus gamma=0 return, per T, mean +/- std over seeds."""
    d = ev[(ev.method == "dqn") & (~ev.mask_history.astype(bool)) & ev.gamma.isin([0.0, 0.9])]
    s = seed_summary(d, ["mode", "reward", "T", "gamma"])
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    for (mode, reward), g in s.groupby(["mode", "reward"]):
        w = g.pivot_table(index="T", columns="gamma", values=["mean", "std"])
        if ("mean", 0.9) not in w or ("mean", 0.0) not in w:
            continue
        diff = w[("mean", 0.9)] - w[("mean", 0.0)]
        err = (w[("std", 0.9)] ** 2 + w[("std", 0.0)] ** 2) ** 0.5
        ax.errorbar(diff.index, diff.values, yerr=err.values, marker="o", capsize=2, label=f"{mode} / {reward}")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set(xlabel="session length T", ylabel="return(γ=0.9) − return(γ=0)", title="Planning gain vs T (DQN)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig5_sim_vs_live(both: pd.DataFrame, rho: float, out: Path) -> None:
    """Per-method mean return on the simulator vs on the real LLM (E6 ranking check)."""
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.scatter(both["sim"], both["live"])
    for r in both.itertuples():
        ax.annotate(r.label, (r.sim, r.live), fontsize=7, xytext=(3, 3), textcoords="offset points")
    lo = min(both["sim"].min(), both["live"].min())
    hi = max(both["sim"].max(), both["live"].max())
    ax.plot([lo, hi], [lo, hi], color="grey", lw=0.8, ls=":")
    ax.set(xlabel="simulator return", ylabel="real-LLM return", title=f"Sim vs real (Spearman ρ = {rho:.2f})")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
