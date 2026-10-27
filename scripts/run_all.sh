#!/usr/bin/env bash
# The whole experiment after E0, in order (plan §6.2, R2-R10). Safe to re-run: every finished step
# leaves a marker in results/.done/, and each script also resumes inside a step (cached LLM replies,
# finished sessions and trained runs are skipped). Progress: tail -f results/pipeline.log
#
#   WORKERS=6 bash scripts/run_all.sh        # WORKERS = parallel training processes (default 6)
#   bash scripts/run_all.sh --from r7        # forget the markers from r7 on and redo those steps
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
WORKERS=${WORKERS:-6}
WORLD=data/world_s0
mkdir -p results/.done
exec > >(tee -a results/pipeline.log) 2>&1

STEPS=(r2 r3 r4 r6 r7 r8 r9 r10)
if [[ "${1:-}" == "--from" ]]; then
  drop=0
  for s in "${STEPS[@]}"; do [[ $s == "$2" ]] && drop=1; [[ $drop == 1 ]] && rm -f "results/.done/$s"; done
fi

step() {   # step <id> <description> <command...>
  local id=$1 desc=$2; shift 2
  if [[ -f results/.done/$id ]]; then echo "[run_all] $id already done: $desc"; return; fi
  echo; echo "[run_all] $(date '+%F %T') START $id: $desc"
  "$@"
  touch "results/.done/$id"
  echo "[run_all] $(date '+%F %T') DONE  $id"
}

[[ -f .env ]] || { echo ".env with DEEPSEEK_API_KEY is missing"; exit 1; }
[[ -d $WORLD ]] || $PY scripts/gen_world.py --out $WORLD

step r2 "E0 pilot (re-runs only failed sessions; finished ones come from the cache)" \
  $PY scripts/run_pilot.py --world $WORLD --out results/e0

r3() {
  $PY scripts/collect_logs.py --world $WORLD --mode persistent --out data/logs
  $PY scripts/collect_logs.py --world $WORLD --mode ephemeral --out data/logs
  $PY scripts/collect_logs.py --world $WORLD --mode persistent --split val --out data/logs
}
step r3 "real-LLM logs, 1500 sessions per mode + validation users" r3

step r4 "fit the behaviour model (simulator)" \
  $PY scripts/fit_sim.py --logs data/logs --world $WORLD --out runs/sim

step r6 "exact DP on the small instance (Delta, Delta_amort, behaviour-only Delta)" \
  $PY scripts/solve_dp.py --sim runs/sim --out runs/e3

step r7 "train / tune every method on the simulator" \
  $PY scripts/train.py --workers "$WORKERS" --sweep configs/exp/e1_main.yaml configs/exp/e2_gamma.yaml \
    configs/exp/e_behavior.yaml configs/exp/e4_ablation.yaml configs/exp/e5_ephemeral.yaml configs/exp/e3_full.yaml

step r8 "evaluate every run on the test users (simulator)" \
  $PY scripts/evaluate.py --backend sim --workers "$WORKERS"

step r9 "E6: real-LLM check of the method ranking" \
  $PY scripts/evaluate.py --backend live --select configs/exp/e6_live.yaml

step r10 "figures and tables" $PY scripts/make_figures.py

echo "[run_all] $(date '+%F %T') ALL DONE -> figures/summary.json"
