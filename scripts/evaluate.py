"""R8 / R9: evaluate trained runs on the test users, on the simulator or on the real LLM (plan §6.2).

sim : every run under --runs, every seed, the full fixed test set -> results/sim_eval.parquet
live: the runs a selection YAML names (configs/exp/e6_live.yaml), best seed by val_return, the
      first --n-sessions test scripts, the same scripts for every method -> results/live_eval.parquet

Both write one row per (run, session) with the tags make_figures.py groups by.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

from memgate.config import load_reward, load_yaml, save_run_metadata
from memgate.experiment import (
    eval_scripts,
    evaluate_sim,
    label_of,
    load_policy,
    load_sim,
    load_world,
    resolve,
    run_dirs,
)
from memgate.harness.runner import policy_chooser, read_logs, run_sessions

TAGS = ("method", "gamma", "seed", "mode", "reward", "mask_history", "sweep")


def run_id(run_dir: Path) -> str:
    return f"{run_dir.parent.name}/{run_dir.name}"     # sweep/run: run names repeat across sweeps


def eval_T(meta: dict, world) -> int:
    T = meta.get("T_choices") or [world.cfg.T_default]
    return int(T[0]) if len(T) == 1 else world.cfg.T_default


def tags_of(meta: dict, run_dir: Path, T: int) -> dict:
    t = {k: meta.get(k) for k in TAGS}
    t["gamma"] = float(t["gamma"] or 0.0)
    t["seed"] = int(t["seed"] or 0)
    t["mask_history"] = bool(t["mask_history"])
    t.update(label=label_of(meta), T=T, run=run_id(run_dir), val_return=meta.get("val_return"))
    return t


def matches(meta: dict, sel: dict) -> bool:
    for k, v in sel.items():
        m = meta.get(k)
        if k == "gamma":
            m, v = float(m or 0.0), float(v)
        elif k == "mask_history":
            m, v = bool(m), bool(v)
        if m != v:
            return False
    return True


def group_key(meta: dict) -> tuple:
    return (meta["method"], float(meta.get("gamma") or 0.0), meta["mode"], meta["reward"],
            bool(meta.get("mask_history")), json.dumps(meta.get("T_choices")), meta.get("sweep"))


def best_seed(dirs: list[Path]) -> list[Path]:
    """One run per method group: the seed with the highest validation return (never the test return)."""
    groups: dict[tuple, tuple[float, Path]] = {}
    for d in dirs:
        meta = json.loads((d / "run.json").read_text())
        v = meta.get("val_return")
        v = -np.inf if v is None else float(v)
        k = group_key(meta)
        if k not in groups or v > groups[k][0]:
            groups[k] = (v, d)
    return sorted(p for _, p in groups.values())


# ---- sim ---------------------------------------------------------------------------------------

def sim_one(job: dict) -> pd.DataFrame | str:
    import traceback

    import torch
    torch.set_num_threads(1)
    try:
        run_dir = Path(job["run_dir"])
        meta = json.loads((run_dir / "run.json").read_text())
        world = load_world(job["world"] or meta["world"])
        bm = load_sim(job["sim"] or meta["sim"])
        pol, _ = load_policy(run_dir, world, bm)
        T = job["T"] or eval_T(meta, world)
        scripts = eval_scripts(world, "test", T)
        df, _ = evaluate_sim(pol, world, bm, scripts, mode=meta["mode"], reward=meta["reward"],
                             mask_history=bool(meta.get("mask_history")))
        for k, v in tags_of(meta, run_dir, T).items():
            df[k] = v
        return df
    except Exception:
        return f"{job['run_dir']}\n{traceback.format_exc()}"


def run_sim(a, dirs: list[Path]) -> pd.DataFrame:
    jobs = [{"run_dir": str(d), "world": a.world, "sim": a.sim, "T": a.T} for d in dirs]
    with get_context("spawn").Pool(a.workers) as pool:
        res = pool.map(sim_one, jobs, chunksize=1)
    errs = [r for r in res if isinstance(r, str)]
    for e in errs:
        print(f"[eval] FAILED {e}")
    if errs:
        raise SystemExit(f"{len(errs)} runs failed to evaluate")
    return pd.concat(res, ignore_index=True)


# ---- live --------------------------------------------------------------------------------------

def live_session_table(logs: pd.DataFrame, world) -> pd.DataFrame:
    """Per-turn real-LLM logs -> the same per-session columns evaluate_sim produces."""
    toks = {r.id: r.tokens for r in world.records.values()}
    rows = []
    for sid, g in logs.groupby("session_id"):
        g = g.sort_values("t")
        q = g[g.kind == "query"]
        mem = 0.0
        for row in g.itertuples(index=False):
            snap = json.loads(row.state_json)
            inj = set(json.loads(row.injected_ids))
            eff = (set(snap["in_context"]) | inj) if row.mode == "persistent" else inj
            mem += sum(toks.get(i, 0) for i in eff)
        rows.append({
            "session_id": sid, "return": g.reward.sum(), "quality": g.r_quality.sum(), "stale": g.r_stale.sum(),
            "intrusion": g.r_intrusion.sum(), "tokens": g.r_tokens.sum(),
            "acc": q.correct.mean() if len(q) else np.nan, "stale_rate": q.stale_used.mean() if len(q) else np.nan,
            "n_intrusions": int(sum(len(json.loads(x)) for x in g.intrusion_ids)), "mem_tokens": mem / len(g),
            "hedged_rate": q.hedged.mean() if len(q) else np.nan,
            "prompt_tokens": int(g.prompt_tokens.sum()), "completion_tokens": int(g.completion_tokens.sum()),
        })
    return pd.DataFrame(rows)


async def run_live(a, dirs: list[Path], sel_cfg: dict) -> pd.DataFrame:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from _common import make_llm

    metas = [json.loads((d / "run.json").read_text()) for d in dirs]
    world = load_world(a.world or metas[0]["world"])
    bm = load_sim(a.sim or metas[0]["sim"])
    llm, cfg = make_llm(a.llm_config, world, a.fake)
    n = a.n_sessions or int(sel_cfg.get("n_sessions", 300))
    out_root = resolve(a.live_dir)
    tables = []
    for d, meta in zip(dirs, metas, strict=True):
        T = a.T or eval_T(meta, world)
        scripts = eval_scripts(world, "test", T)
        if len(scripts) < n:
            raise SystemExit(f"only {len(scripts)} test scripts at T={T}, asked for {n}")
        scripts = scripts[:n]                 # identical scripts for every method (H2)
        pol, _ = load_policy(d, world, bm)
        mh = bool(meta.get("mask_history"))
        print(f"[live] {run_id(d)}: {n} sessions, T={T}", flush=True)
        stats = await run_sessions(scripts, lambda s, p=pol, m=mh: policy_chooser(p, m), llm, world,
                                   mode=meta["mode"], reward=load_reward(meta["reward"]),
                                   out_dir=out_root / run_id(d), concurrency=cfg.concurrency, mask_history=mh,
                                   run_tags={"run": run_id(d), "label": label_of(meta)}, every=50)
        print(f"[live] {run_id(d)}: {stats}", flush=True)
        logs = read_logs(out_root / run_id(d))
        logs = logs[logs.session_id.isin({s.id for s in scripts})]
        tab = live_session_table(logs, world)
        for k, v in tags_of(meta, d, T).items():
            tab[k] = v
        tables.append(tab)
    u = llm.usage
    print(f"[live] total this run: {u.calls} calls, {u.cost_cny:.2f} CNY")
    return pd.concat(tables, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["sim", "live"], required=True)
    ap.add_argument("--runs", nargs="+", default=None, help="sweep directories (sim); default: all of runs/")
    ap.add_argument("--select", default="configs/exp/e6_live.yaml", help="live: which runs to evaluate")
    ap.add_argument("--best-seed", action="store_true", help="sim: keep only the best seed per group")
    ap.add_argument("--world", default=None, help="override the world in run.json")
    ap.add_argument("--sim", default=None, help="override the behaviour model in run.json")
    ap.add_argument("--T", type=int, default=None, help="evaluate at this T instead of the trained one")
    ap.add_argument("--n-sessions", type=int, default=None)
    ap.add_argument("--llm-config", default="configs/llm.yaml")
    ap.add_argument("--live-dir", default="results/live_logs")
    ap.add_argument("--fake", action="store_true", help="live backend with the offline FakeLLM")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.backend == "sim":
        roots = [resolve(r) for r in (a.runs or ["runs"])]
        dirs = sorted({d for r in roots for d in run_dirs(r)})
        if a.best_seed:
            dirs = best_seed(dirs)
        if not dirs:
            raise SystemExit("no run.json found under " + ", ".join(map(str, roots)))
        print(f"[eval] {len(dirs)} runs on the simulator")
        df = run_sim(a, dirs)
        out = resolve(a.out or "results/sim_eval.parquet")
    else:
        sel_cfg = load_yaml(resolve(a.select))
        roots = [resolve(r) for r in (a.runs or sel_cfg["runs"])]
        dirs_all = sorted({d for r in roots for d in run_dirs(r)})
        dirs = []
        for sel in sel_cfg["select"]:
            sel = {**{k: sel_cfg[k] for k in ("mode", "reward") if k in sel_cfg}, **sel}
            hit = [d for d in dirs_all if matches(json.loads((d / "run.json").read_text()), sel)]
            if not hit:
                raise SystemExit(f"no trained run matches {sel}")
            dirs += [d for d in best_seed(hit) if d not in dirs]
        print(f"[eval] live on {len(dirs)} runs: {[d.name for d in dirs]}")
        df = asyncio.run(run_live(a, dirs, sel_cfg))
        df["backend"] = "fake" if a.fake else "live"
        out = resolve(a.out or "results/live_eval.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    save_run_metadata(out.parent, name=f"{out.stem}_meta.json", args=vars(a), n_rows=len(df))
    summ = df.groupby(["mode", "reward", "label"])["return"].agg(["mean", "count"]).round(3)
    print(summ.to_string())
    print(f"[eval] wrote {out}")


if __name__ == "__main__":
    main()
