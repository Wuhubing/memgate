"""R5: fit the behaviour model and run the three simulator checks (plan §3.10; 原文档 模块 6)."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from memgate.config import git_commit, save_run_metadata
from memgate.experiment import load_world, make_sim_env, resolve
from memgate.harness.runner import read_logs
from memgate.schema import script_from_json  # noqa: F401  (kept for interactive use)
from memgate.sim.behavior_model import BehaviorModel
from memgate.sim.dataset import build_datasets
from memgate.sim.outcome_features import ANSWER_CLASSES
from memgate.text import sha


def log_loss_multi(model: BehaviorModel, X, y) -> float:
    if len(y) == 0:
        return float("nan")
    p = model.answer_probs_x(X)
    idx = np.array([ANSWER_CLASSES.index(c) for c in y])
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), idx], 1e-9, 1))))


def log_loss_bin(p, y) -> float:
    if len(y) == 0:
        return float("nan")
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration(p, y, bins=10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = [{"bin": b, "p_mean": p[idx == b].mean(), "freq": y[idx == b].mean(), "n": int((idx == b).sum())}
            for b in range(bins) if (idx == b).any()]
    return pd.DataFrame(rows)


def gbt_compare(tr, te) -> dict:
    """Gradient boosting reference (F4): if it beats the logistic model clearly, features are missing."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    out = {}
    if len(set(tr.ya)) > 1 and len(te.ya):
        g = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(tr.Xa, tr.ya)
        p = g.predict_proba(te.Xa)
        cls = list(g.classes_)
        ok = np.array([c in cls for c in te.ya])
        idx = np.array([cls.index(c) for c in te.ya[ok]])
        out["answer_gbt_logloss"] = float(-np.mean(np.log(np.clip(p[ok][np.arange(ok.sum()), idx], 1e-9, 1))))
    if len(set(tr.ym)) > 1 and len(te.ym):
        g = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(tr.Xm, tr.ym)
        out["mention_gbt_logloss"] = log_loss_bin(g.predict_proba(te.Xm)[:, 1], te.ym)
    return out


def replay_in_sim(df: pd.DataFrame, world, model: BehaviorModel, reward: str = "R_full", seed: int = 0) -> pd.DataFrame:
    """Re-run the logged sessions in the simulator with the *logged actions* (same candidates, since the
    context only depends on actions), and compare reward components with the real logs."""
    from memgate.schema import SessionScript, turn_from_dict
    rows = []
    for (sid, mode), g in df.groupby(["session_id", "mode"]):
        g = g.sort_values("t")
        script = SessionScript(sid, g.user_id.iloc[0], tuple(turn_from_dict(json.loads(t)) for t in g.turn_json))
        env = make_sim_env(world, model, split="train", mode=mode, reward=reward, T_choices=[script.T],
                           mask_history=False, seed=seed, scripts=[script])
        obs, info = env.reset(options={"script": script})
        tot = np.zeros(4)
        for a in g.action:
            obs, r, term, _, info = env.step(int(a))
            o = info["outcome"]
            tot += [o.r_quality, o.r_stale, o.r_intrusion, o.r_tokens]
            if term:
                break
        real = g[["r_quality", "r_stale", "r_intrusion", "r_tokens"]].sum().to_numpy()
        rows.append({"session_id": sid, "mode": mode, "behavior_policy": g.get("behavior_policy", pd.Series([""])).iloc[0],
                     **{f"sim_{k}": v for k, v in zip(("quality", "stale", "intrusion", "tokens"), tot, strict=True)},
                     **{f"real_{k}": v for k, v in zip(("quality", "stale", "intrusion", "tokens"), real, strict=True)}})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True, help="directory with train/<mode>/part-*.parquet (and optionally val/)")
    ap.add_argument("--world", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--C", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    ap.add_argument("--replay", type=int, default=300, help="sessions to replay for the sim-vs-real check")
    a = ap.parse_args()
    world = load_world(a.world)
    out = resolve(a.out)
    out.mkdir(parents=True, exist_ok=True)
    logs_root = resolve(a.logs)
    df = read_logs(logs_root / "train")
    commits = sorted(df.git_commit.unique()) if "git_commit" in df else []
    print(f"{len(df)} turns from {df.session_id.nunique()} sessions; log commits {commits}; now {git_commit()[:8]}")
    if "llm_model" in df:
        print("models in logs:", df.llm_model.value_counts().to_dict())   # A1
    ds = build_datasets(df, world)
    sessions = sorted(df.session_id.unique())
    held = {s for s in sessions if int(sha(s, 8), 16) % 5 == 0}           # 20% of sessions held out
    tr, te = ds.split(held)

    best, best_ll = None, np.inf
    report = {"n_answer": len(ds.ya), "n_mention": len(ds.ym), "answer_class_counts":
              pd.Series(ds.ya).value_counts().to_dict(), "mention_rate": float(ds.ym.mean()) if len(ds.ym) else 0}
    for C in a.C:
        m = BehaviorModel.fit(tr.Xa, tr.ya, tr.Xm, tr.ym, tr.Xl, tr.yl, C=C)
        ll = log_loss_multi(m, te.Xa, te.ya) + log_loss_bin(m.mention_probs_x(te.Xm), te.ym)
        report[f"heldout_logloss_C{C}"] = ll
        if ll < best_ll:
            best, best_ll, best_C = m, ll, C
    report["C"] = best_C
    report["answer_logloss"] = log_loss_multi(best, te.Xa, te.ya)
    report["mention_logloss"] = log_loss_bin(best.mention_probs_x(te.Xm), te.ym)
    report.update(gbt_compare(tr, te))
    final = BehaviorModel.fit(ds.Xa, ds.ya, ds.Xm, ds.ym, ds.Xl, ds.yl, C=best_C)   # refit on all logs
    final.meta.update(report)
    final.save(out / "model.pkl")

    # check 1: calibration on the held-out sessions
    cal = []
    pa = best.answer_probs_x(te.Xa) if len(te.ya) else np.zeros((0, 3))
    for j, c in enumerate(ANSWER_CLASSES):
        if len(te.ya):
            k = calibration(pa[:, j], (te.ya == c).astype(float))
            k["target"] = f"answer={c}"
            cal.append(k)
    if len(te.ym):
        k = calibration(best.mention_probs_x(te.Xm), te.ym.astype(float))
        k["target"] = "mention"
        cal.append(k)
    cal = pd.concat(cal, ignore_index=True) if cal else pd.DataFrame()
    cal.to_csv(out / "calibration.csv", index=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4, 4))
        for tgt, g in cal.groupby("target"):
            ax.plot(g.p_mean, g.freq, marker="o", label=tgt)
        ax.plot([0, 1], [0, 1], "k:", lw=1)
        ax.set(xlabel="predicted", ylabel="observed", title="held-out calibration")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "calibration.png", dpi=150)
        plt.close(fig)
    except Exception as e:  # plotting never blocks the fit
        print("calibration plot skipped:", e)

    # check 2: replay the behaviour policy in the simulator and compare reward components
    sample = sorted(df.session_id.unique())[: a.replay]
    rep = replay_in_sim(df[df.session_id.isin(sample)], world, final)
    rep.to_csv(out / "sim_vs_real.csv", index=False)
    comp = {k: {"real": rep[f"real_{k}"].mean(), "sim": rep[f"sim_{k}"].mean()}
            for k in ("quality", "stale", "intrusion", "tokens")}
    report["sim_vs_real"] = comp

    # F7: held-out users (validation-user logs), if collected
    if (logs_root / "val").exists():
        dv = build_datasets(read_logs(logs_root / "val"), world)
        report["val_users_answer_logloss"] = log_loss_multi(final, dv.Xa, dv.ya)
        report["val_users_mention_logloss"] = log_loss_bin(final.mention_probs_x(dv.Xm), dv.ym)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    save_run_metadata(out, args=vars(a))
    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
