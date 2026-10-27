"""R7: train / tune every method of a sweep on the simulator (plan §3.12, §6.2).

A sweep YAML expands into runs (grid x defaults + extra). Runs execute in parallel processes,
one torch thread each (E12). A run directory with run.json is skipped unless --overwrite.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from memgate.config import load_yaml, save_run_metadata
from memgate.experiment import (
    eval_scripts,
    evaluate_sim,
    expand_sweep,
    label_of,
    load_policy,
    load_sim,
    load_world,
    make_sim_env,
    resolve,
    run_name,
)
from memgate.policies.heuristics import ThresholdPolicy
from memgate.policies.linucb import train_linucb
from memgate.retrieval.retriever import Retriever


def val_return(policy, world, bm, r, n: int) -> float:
    scripts = eval_scripts(world, "val")[:n]
    df, _ = evaluate_sim(policy, world, bm, scripts, mode=r["mode"], reward=r["reward"],
                         mask_history=r.get("mask_history", False), seed=7)
    return float(df["return"].mean())


def tune_threshold(world, bm, r, n_val: int) -> tuple[float, float]:
    ret = Retriever(world.emb)
    fused = []
    for s in eval_scripts(world, "val")[:100]:
        store = world.user(s.user_id).initial_store
        for t in s.turns:
            fused += [c.fused for c in ret.topk(t.text, store) if c is not None]
    grid = np.quantile(fused, np.linspace(0, 1, 20))
    best = max((val_return(ThresholdPolicy(float(tau)), world, bm, r, n_val), float(tau)) for tau in grid)
    return best[1], best[0]


def run_one(r: dict) -> tuple[str, str]:
    out = resolve(r["runs_root"]) / r["sweep"] / run_name(r)
    if (out / "run.json").exists() and not r.get("overwrite"):
        return str(out), "skipped"
    try:
        import torch
        torch.set_num_threads(1)
        t0 = time.monotonic()
        world, bm = load_world(r["world"]), load_sim(r["sim"])
        out.mkdir(parents=True, exist_ok=True)
        m, seed, n_val = r["method"], int(r.get("seed", 0)), int(r.get("n_val", 300))
        T_choices = r.get("T_choices", [world.cfg.T_default])
        meta = {**r, "label": label_of(r), "gamma": float(r.get("gamma", 0.0)),
                "mask_history": bool(r.get("mask_history", False))}

        def mk(i, split="train"):
            return make_sim_env(world, bm, split=split, mode=r["mode"], reward=r["reward"], T_choices=T_choices,
                                mask_history=meta["mask_history"], seed=seed * 100 + i)

        if m in ("dqn", "ppo"):
            from memgate.policies.sb3 import train_sb3
            eval_env = make_sim_env(world, bm, split="val", mode=r["mode"], reward=r["reward"], T_choices=T_choices,
                                    mask_history=meta["mask_history"], seed=seed + 999,
                                    scripts=eval_scripts(world, "val")[:200])
            train_sb3(m, mk, gamma=meta["gamma"], total_steps=int(r["steps"]), seed=seed, run_dir=out,
                      n_envs=int(r.get("n_envs", 8)), eval_env=eval_env, eval_freq=int(r.get("eval_freq", 10_000)),
                      overrides=r.get("hparams"))
        elif m in ("linucb", "linucb_risk"):
            best = None
            for alpha in r.get("alphas", [0.1, 0.5, 1.0]):
                pol = train_linucb(mk(0), int(r["steps"]), alpha, risk_sensitive=m == "linucb_risk", seed=seed)
                v = val_return(pol, world, bm, r, n_val)
                if best is None or v > best[0]:
                    best = (v, alpha, pol)
            meta["alpha_ucb"] = best[1]
            best[2].save(out / "policy.pkl")
        elif m == "threshold":
            meta["tau"], _ = tune_threshold(world, bm, r, n_val)
        elif m not in ("always", "never", "model_greedy", "amortized_greedy"):
            raise ValueError(f"unknown method {m}")
        (out / "run.json").write_text(json.dumps(meta, default=float))   # load_policy reads it
        pol, _ = load_policy(out, world, bm)
        meta["val_return"] = val_return(pol, world, bm, r, n_val)
        meta["train_seconds"] = time.monotonic() - t0
        (out / "run.json").write_text(json.dumps(meta, indent=2, default=float))
        save_run_metadata(out, run=meta)
        return str(out), "ok"
    except Exception:
        (out / "error.txt").write_text(traceback.format_exc())
        return str(out), "failed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True, nargs="+")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--steps", type=int, default=None, help="override total steps (quick checks)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="override the seed list")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    runs = []
    for path in a.sweep:
        for r in expand_sweep(load_yaml(resolve(path))):
            r["runs_root"] = a.runs_root
            r["overwrite"] = a.overwrite
            if a.steps:
                r["steps"] = a.steps
            runs.append(r)
    if a.seeds is not None:
        base = {json.dumps({k: v for k, v in r.items() if k != "seed"}, sort_keys=True, default=str): r for r in runs}
        runs = [{**r, "seed": s} for r in base.values() for s in a.seeds]
    print(f"{len(runs)} runs, {a.workers} workers")
    ctx = get_context("spawn")
    with ctx.Pool(a.workers) as pool:
        results = pool.map(run_one, runs, chunksize=1)
    failed = [p for p, s in results if s == "failed"]
    for p, s in results:
        print(f"  {s:8s} {p}")
    if failed:
        Path(resolve(a.runs_root) / "_failed.txt").write_text("\n".join(failed))
        raise SystemExit(f"{len(failed)} runs failed; see error.txt in each (list in runs/_failed.txt)")


if __name__ == "__main__":
    main()
