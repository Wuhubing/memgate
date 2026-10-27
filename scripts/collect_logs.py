"""R3: real-LLM logs under the mixed behaviour policy, for fitting the simulator (plan §3.10)."""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _common import make_llm  # noqa: E402

from memgate.config import load_reward, save_run_metadata  # noqa: E402
from memgate.experiment import load_world, resolve  # noqa: E402
from memgate.harness.runner import policy_chooser, run_sessions  # noqa: E402
from memgate.policies.heuristics import MixedBehaviorPolicy  # noqa: E402
from memgate.retrieval.retriever import Retriever  # noqa: E402
from memgate.world.generator import load_scripts  # noqa: E402


def median_fused(world, scripts, n: int = 200) -> float:
    """Default threshold for the 30% threshold-policy sessions: the median fused score of top-3 candidates."""
    ret = Retriever(world.emb)
    vals = []
    for s in scripts[:n]:
        store = world.user(s.user_id).initial_store
        for t in s.turns:
            vals += [c.fused for c in ret.topk(t.text, store) if c is not None]
    return float(np.median(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True)
    ap.add_argument("--llm", default="configs/llm.yaml")
    ap.add_argument("--split", choices=["train", "val"], default="train")
    ap.add_argument("--mode", choices=["persistent", "ephemeral"], required=True)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--p-uniform", type=float, default=0.7)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fake", action="store_true")
    a = ap.parse_args()
    world = load_world(a.world)
    llm, cfg = make_llm(a.llm, world, a.fake)
    name = "scripts_log.jsonl" if a.split == "train" else "scripts_log_val.jsonl"
    scripts = load_scripts(world.path / name)
    # each mode uses its own half of the fixed log scripts so the two modes are independent samples
    half = len(scripts) // 2 if a.split == "train" else len(scripts)
    scripts = (scripts[:half] if a.mode == "persistent" or a.split == "val" else scripts[half:])[: a.n]
    tau = a.tau if a.tau is not None else median_fused(world, scripts)
    out = resolve(a.out) / a.split / a.mode
    seed_of = {s.id: i for i, s in enumerate(scripts)}
    pols = {s.id: MixedBehaviorPolicy(tau, a.p_uniform, seed_of[s.id]) for s in scripts}
    stats = asyncio.run(run_sessions(
        scripts, lambda s: policy_chooser(pols[s.id]), llm, world, mode=a.mode, reward=load_reward(),
        out_dir=out, prompt_version=cfg.system_prompt_version, concurrency=cfg.concurrency,
        tags_for=lambda s: {"behavior_policy": pols[s.id].name, "tau": tau, "split": a.split}))
    save_run_metadata(out, llm=cfg, args=vars(a), tau=tau)
    u = llm.usage
    print(f"{stats}; tau={tau:.4f}; usage {u.calls} calls, {u.prompt_tokens} in, {u.completion_tokens} out, "
          f"{u.cost_cny:.2f} CNY")


if __name__ == "__main__":
    main()
