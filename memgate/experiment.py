"""Glue shared by the scripts: environments, run directories, policy loading, simulator evaluation."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from memgate.config import ROOT, load_reward
from memgate.env.backends import SimBackend
from memgate.env.gate_env import MemGateEnv
from memgate.policies.heuristics import AlwaysPolicy, NeverPolicy, ThresholdPolicy
from memgate.policies.linucb import LinUCB
from memgate.policies.model_based import ModelGreedy
from memgate.sim.behavior_model import BehaviorModel
from memgate.world.generator import SessionFactory, World, fixed_scripts, load_scripts

TRAINED = {"dqn", "ppo", "linucb", "linucb_risk"}
EVAL_SEED = 12345


def resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def load_world(path) -> World:
    return World.load(resolve(path))


def load_sim(path) -> BehaviorModel:
    p = resolve(path)
    return BehaviorModel.load(p / "model.pkl" if p.is_dir() else p)


def eval_scripts(world: World, split: str, T: int | None = None, per_user: int | None = None):
    """The fixed evaluation set. Default T uses the saved file; other T are generated deterministically."""
    T = T or world.cfg.T_default
    per_user = per_user or (world.cfg.scripts_per_test_user if split == "test" else world.cfg.scripts_per_val_user)
    saved = world.path / f"scripts_{split}.jsonl" if world.path else None
    if T == world.cfg.T_default and saved is not None and saved.exists():
        return load_scripts(saved)
    return fixed_scripts(world, split, per_user, T, EVAL_SEED + T)


def make_sim_env(world: World, bm: BehaviorModel, *, split: str, mode: str, reward: str, T_choices: list[int],
                 mask_history: bool, seed: int, scripts=None) -> MemGateEnv:
    backend = SimBackend(world, bm, seed=seed + 1)
    factory = None if scripts is not None else SessionFactory(world, split, seed)
    return MemGateEnv(world, backend, mode=mode, reward=load_reward(reward), scripts=scripts, factory=factory,
                      T_choices=T_choices, mask_history=mask_history, seed=seed)


# ---- sweeps ------------------------------------------------------------------------------------

def expand_sweep(cfg: dict) -> list[dict]:
    defaults = cfg.get("defaults", {})
    grid = cfg.get("grid", {})
    keys = list(grid)
    runs = []
    for combo in itertools.product(*(grid[k] for k in keys)) if keys else [()]:
        r = {**defaults, **dict(zip(keys, combo, strict=True))}
        r.update({k: cfg[k] for k in ("world", "sim") if k in cfg})
        r["sweep"] = cfg["name"]
        runs.append(r)
    for extra in cfg.get("extra", []):
        runs.append({**defaults, **extra, "world": cfg.get("world"), "sim": cfg.get("sim"), "sweep": cfg["name"]})
    return runs


def run_name(r: dict) -> str:
    T = "-".join(str(t) for t in r.get("T_choices", [12]))
    g = f"_g{r['gamma']}" if r["method"] in ("dqn", "ppo") else ""
    mh = "_nohist" if r.get("mask_history") else ""
    return f"{r['method']}{g}_{r['mode']}_{r['reward']}{mh}_T{T}_s{r.get('seed', 0)}"


def label_of(r: dict) -> str:
    lab = r["method"]
    if r["method"] in ("dqn", "ppo"):
        lab += f" g={r['gamma']}"
    if r.get("mask_history"):
        lab += " no-hist"
    return lab


def load_policy(run_dir: Path, world: World, bm: BehaviorModel):
    meta = json.loads((run_dir / "run.json").read_text())
    m = meta["method"]
    if m in ("dqn", "ppo"):
        from memgate.policies.sb3 import SB3Policy
        return SB3Policy(run_dir / "policy.zip", m, label_of(meta)), meta
    if m in ("linucb", "linucb_risk"):
        return LinUCB.load(run_dir / "policy.pkl"), meta
    if m == "threshold":
        return ThresholdPolicy(meta["tau"]), meta
    if m == "always":
        return AlwaysPolicy(), meta
    if m == "never":
        return NeverPolicy(), meta
    if m in ("model_greedy", "amortized_greedy"):
        backend = SimBackend(world, bm)
        return ModelGreedy(bm, backend, load_reward(meta["reward"]), amortize=m == "amortized_greedy"), meta
    raise ValueError(m)


def run_dirs(root: Path) -> list[Path]:
    return sorted(p.parent for p in Path(root).glob("**/run.json"))


# ---- evaluation on the simulator --------------------------------------------------------------

def evaluate_sim(policy, world: World, bm: BehaviorModel, scripts, *, mode: str, reward: str, mask_history: bool,
                 seed: int = EVAL_SEED, keep_turns: bool = False) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    env = make_sim_env(world, bm, split="test", mode=mode, reward=reward, T_choices=[world.cfg.T_default],
                       mask_history=mask_history, seed=seed, scripts=scripts)
    env.backend.reseed(seed)
    rows, turns = [], []
    for s in scripts:
        obs, info = env.reset(options={"script": s})
        tot = np.zeros(4)
        n_q = n_c = n_s = n_i = n_h = 0
        mem_tok = 0.0
        while True:
            a = policy.act(obs, info)
            obs, r, term, trunc, info = env.step(a)
            out = info["outcome"]
            tot += [out.r_quality, out.r_stale, out.r_intrusion, out.r_tokens]
            if info["turn"].kind == "query":
                n_q += 1
                n_c += out.answer == "correct"
                n_s += out.answer == "stale"
            n_i += len(out.intrusion_ids)
            n_h += out.hedged
            recs = env.st.by_id()
            mem_tok += sum(recs[i].tokens for i in (set(env.st.in_context) if mode == "persistent"
                                                     else set(info["chosen"])) if i in recs)
            if keep_turns:
                turns.append({"session_id": s.id, "t": info["turn"].idx, "kind": info["turn"].kind,
                              "injected": ", ".join(info["chosen"]), "reward": r})
            if term or trunc:
                break
        rows.append({"session_id": s.id, "return": tot.sum(), "quality": tot[0], "stale": tot[1],
                     "intrusion": tot[2], "tokens": tot[3], "acc": n_c / n_q if n_q else np.nan,
                     "stale_rate": n_s / n_q if n_q else np.nan, "n_intrusions": n_i,
                     "mem_tokens": mem_tok / s.T, "hedged_rate": n_h / n_q if n_q else np.nan})
    return pd.DataFrame(rows), (pd.DataFrame(turns) if keep_turns else None)
