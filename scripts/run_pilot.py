"""R2 / E0: does an injected memory keep affecting later turns? (plan §3.9)"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _common import make_llm  # noqa: E402

from memgate.analysis.plots import fig1_pilot  # noqa: E402
from memgate.analysis.stats import bootstrap_ci  # noqa: E402
from memgate.config import load_reward, save_run_metadata  # noqa: E402
from memgate.experiment import load_world, resolve  # noqa: E402
from memgate.harness.runner import read_logs, run_sessions  # noqa: E402
from memgate.schema import SessionScript  # noqa: E402
from memgate.world.generator import make_pilot_session, stale_records  # noqa: E402

CONDITIONS = ("none", "current", "stale", "irrelevant")


def build(world, n: int, seed: int):
    rng = np.random.default_rng(seed)
    users = [u for u in world.split_users("train") if any(stale_records(u.initial_store, k) for k in u.changing_keys)]
    base = []
    for j in range(n):
        u = users[int(rng.integers(len(users)))]
        s, K, cur_id, R_id = make_pilot_session(u, rng, f"pilot-{u.id}-{j:03d}")
        olds = sorted(stale_records(u.initial_store, K), key=lambda r: -r.created_at)
        base.append({"script": s, "K": K, "cur": cur_id, "stale": olds[0].id, "R": R_id, "base": s.id})
    return base


def chooser_for(world, item, cond):
    rec = world.records
    inject = {"none": [], "current": [rec[item["cur"]]], "stale": [rec[item["stale"]]],
              "irrelevant": [rec[item["R"]]]}[cond]

    def choose(st, turn, cands, obs, info):
        return list(inject) if turn.idx == 0 else []
    return choose


def summarize(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    rows = []
    df = df.copy()
    df["R_mention"] = [meta[s]["R"] in json.loads(m) for s, m in zip(df.session_id, df.mentioned_ids, strict=True)]
    for (cond, mode, t), g in df.groupby(["condition", "mode", "t"]):
        for metric in ("R_mention", "stale_used", "correct"):
            if metric != "R_mention" and g.kind.iloc[0] != "query":
                continue
            m, lo, hi = bootstrap_ci(g[metric].astype(float))
            ref = df[(df.condition == "none") & (df["mode"] == mode) & (df.t == t)].set_index("base")[metric]
            cur = g.set_index("base")[metric]
            common = cur.index.intersection(ref.index)
            d, dlo, dhi = bootstrap_ci(cur.loc[common].astype(float) - ref.loc[common].astype(float))
            rows.append({"condition": cond, "mode": mode, "t": t, "kind": g.kind.iloc[0], "metric": metric,
                         "mean": m, "lo": lo, "hi": hi, "diff_vs_none": d, "diff_lo": dlo, "diff_hi": dhi,
                         "n": len(g)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True)
    ap.add_argument("--llm", default="configs/llm.yaml")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--n-ephemeral", type=int, default=20)
    ap.add_argument("--prompt", choices=["v1", "v2"], default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fake", action="store_true", help="offline FakeLLM instead of DeepSeek")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    world = load_world(a.world)
    llm, cfg = make_llm(a.llm, world, a.fake)
    prompt = a.prompt or cfg.system_prompt_version
    out = resolve(a.out)
    reward = load_reward()
    base = build(world, a.n, a.seed)
    meta = {}
    for mode, n in (("persistent", a.n), ("ephemeral", a.n_ephemeral)):
        for cond in CONDITIONS:
            items = base[:n]
            scripts = []
            for it in items:
                sid = f"{it['base']}-{cond}-{mode[:4]}"
                scripts.append(SessionScript(sid, it["script"].user_id, it["script"].turns))
                meta[sid] = {**it, "condition": cond}
            by_id = {s.id: meta[s.id] for s in scripts}
            stats = asyncio.run(run_sessions(
                scripts, lambda s, b=by_id: chooser_for(world, b[s.id], b[s.id]["condition"]), llm, world,
                mode=mode, reward=reward, out_dir=out / "logs" / f"{mode}-{cond}", prompt_version=prompt,
                concurrency=cfg.concurrency,
                tags_for=lambda s, b=by_id: {"condition": b[s.id]["condition"], "base": b[s.id]["base"]}))
            print(f"[E0] {mode}/{cond}: {stats}")
    df = read_logs(out / "logs")
    df.to_parquet(out / "turns.parquet", index=False)
    summ = summarize(df, meta)
    summ.to_csv(out / "summary.csv", index=False)
    fig1_pilot(summ[summ["mode"] == "persistent"], out / "fig1.png")
    save_run_metadata(out, llm=cfg, reward=reward, args=vars(a))

    p = summ[summ["mode"] == "persistent"]
    r_rate = p[(p.condition == "irrelevant") & (p.metric == "R_mention") & (p.t.isin([1, 2, 4, 6]))]
    s_rate = p[(p.condition == "stale") & (p.metric == "stale_used") & (p.t.isin([5, 7]))]
    r_m, s_m = r_rate["mean"].mean(), s_rate["mean"].mean()
    print("\nG0 (persistent mode; control = 'none', plan §3.9):")
    print(f"  distractor R mentioned again in turns 1-7: {r_m:.1%} (diff vs none {r_rate.diff_vs_none.mean():+.1%})")
    print(f"  old value used after the correction (turns 5,7): {s_m:.1%} "
          f"(diff vs none {s_rate.diff_vs_none.mean():+.1%})")
    c = p[(p.condition == "current") & (p.metric == "stale_used") & (p.t.isin([5, 7]))]["mean"].mean()
    print(f"  'correct when injected, stale later' (current condition, turns 5,7): {c:.1%}")
    ok = r_m >= 0.10 or s_m >= 0.10
    print(f"  G0 {'PASSED' if ok else 'NOT passed: see plan J1'}")
    (out / "g0.json").write_text(json.dumps({"R_mention": r_m, "R_mention_diff": r_rate.diff_vs_none.mean(),
                                             "stale_used": s_m, "stale_used_diff": s_rate.diff_vs_none.mean(),
                                             "current_then_stale": c, "passed": bool(ok)}, default=float))
    u = llm.usage
    print(f"usage: {u.calls} calls, {u.prompt_tokens} in, {u.completion_tokens} out, {u.cost_cny:.2f} CNY")


if __name__ == "__main__":
    main()
