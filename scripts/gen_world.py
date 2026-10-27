"""R1: generate the world, the fixed evaluation scripts and the log-collection scripts (plan §3.2)."""
from __future__ import annotations

import argparse

from memgate.config import load_world, save_run_metadata
from memgate.experiment import resolve
from memgate.world.generator import fixed_scripts, generate_world, log_scripts, save_scripts, validate_script


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/world.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p-stale", type=float, default=None, help="override for E3 worlds")
    ap.add_argument("--embedder", choices=["minilm", "hash"], default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    over = {"seed": a.seed}
    if a.p_stale is not None:
        over["p_stale"] = a.p_stale
    if a.embedder:
        over["embedder"] = a.embedder
    cfg = load_world(resolve(a.config), **over)
    out = resolve(a.out)
    world = generate_world(cfg)
    world.save(out)
    val = fixed_scripts(world, "val", cfg.scripts_per_val_user, cfg.T_default, 1000 + a.seed)
    test = fixed_scripts(world, "test", cfg.scripts_per_test_user, cfg.T_default, 2000 + a.seed)
    logs = log_scripts(world, cfg.log_scripts, cfg.log_T_choices, 3000 + a.seed)
    # extra validation-user sessions, only for checking the simulator on unseen users (F7)
    logs_val = fixed_scripts(world, "val", 10, cfg.T_default, 4000 + a.seed)
    for s in val + test + logs + logs_val:
        validate_script(world, s)
    save_scripts(out / "scripts_val.jsonl", val)
    save_scripts(out / "scripts_test.jsonl", test)
    save_scripts(out / "scripts_log.jsonl", logs)
    save_scripts(out / "scripts_log_val.jsonl", logs_val)
    save_run_metadata(out, world=cfg)
    n_rec = sum(len(u.initial_store) for u in world.users.values())
    print(f"world -> {out}: {len(world.users)} users, {n_rec} initial records, "
          f"{len(val)} val / {len(test)} test / {len(logs)} log scripts, embedder={cfg.embedder}")


if __name__ == "__main__":
    main()
