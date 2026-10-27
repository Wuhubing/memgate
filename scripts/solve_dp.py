"""R6 / E3: exact myopia loss on the small instance (plan §3.11, §3.13)."""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import get_context

import pandas as pd

from memgate.analysis.stats import dp_contrasts
from memgate.config import load_reward, load_world, save_run_metadata
from memgate.experiment import load_sim, resolve
from memgate.planning.dp import COMPONENTS, SmallInstance


def solve_one(job: dict) -> list[dict]:
    """All rewards and policies for one (mode, T); the outcome cache is shared across them."""
    bm = load_sim(job["sim"])
    mode, T = job["mode"], job["T"]
    inst = SmallInstance(bm, T, mode, job["p_kind"])
    rows = []
    for rname in job["rewards"]:
        cfg = load_reward(rname)
        t0 = time.monotonic()
        pols = {"dp": inst.solve(cfg, job["gamma"]), "greedy": inst.greedy(cfg),
                "amortized_greedy": inst.greedy(cfg, amortize=True), "always": inst.always(),
                "never": inst.never()}
        for name, pi in pols.items():
            for p_stale, v in inst.evaluate_many(pi, cfg, job["stale"]).items():
                rows.append({"T": T, "p_stale": p_stale, "mode": mode, "reward": rname, "policy": name,
                             "total": float(v.sum()), **dict(zip(COMPONENTS, map(float, v), strict=True))})
        print(f"[dp] mode={mode} T={T} {rname}: {time.monotonic() - t0:.1f}s, "
              f"{inst.n_states} states cached", flush=True)
        inst._rew.clear()     # reward tables are per variant; outcome tables are kept for the next one
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True)
    ap.add_argument("--T", type=int, nargs="+", default=[4, 6, 8, 10])
    ap.add_argument("--stale", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.5, 0.6])
    ap.add_argument("--mode", nargs="+", default=["persistent", "ephemeral"])
    ap.add_argument("--reward", nargs="+", default=["R_full", "R_notok", "R_behavior"])
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=min(2, os.cpu_count() or 1),
                    help="one process per (mode, T); a T=10 job peaks at ~5-8 GB of RAM")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    p_kind = load_world().p_kind
    out = resolve(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # largest T first so the slowest jobs start immediately
    jobs = [{"sim": a.sim, "mode": m, "T": T, "p_kind": p_kind, "rewards": a.reward, "stale": a.stale,
             "gamma": a.gamma} for T in sorted(a.T, reverse=True) for m in a.mode]
    if a.workers > 1 and len(jobs) > 1:
        with get_context("spawn").Pool(min(a.workers, len(jobs))) as pool:
            parts = pool.map(solve_one, jobs, chunksize=1)
    else:
        parts = [solve_one(j) for j in jobs]
    df = pd.DataFrame([r for p in parts for r in p]).sort_values(["mode", "T", "reward", "policy", "p_stale"])
    df.to_csv(out / "delta.csv", index=False)
    con = dp_contrasts(df)
    con.to_csv(out / "contrasts.csv", index=False)
    # sanity relations that must hold (plan §5 / 原文档 "必须成立的关系")
    bad = []
    for (T, p, mode, r), g in df.groupby(["T", "p_stale", "mode", "reward"]):
        v = g.set_index("policy")["total"]
        if any(v["dp"] < v[x] - 1e-9 for x in v.index):
            bad.append((T, p, mode, r))
    (out / "checks.json").write_text(json.dumps({"dp_not_optimal": bad}))
    save_run_metadata(out, args=vars(a))
    print(con.pivot_table(index=["mode", "T"], columns="contrast", values="value", aggfunc="mean").round(4))
    if bad:
        raise SystemExit(f"DP is not >= every other policy in {len(bad)} settings: bug (see checks.json)")


if __name__ == "__main__":
    main()
