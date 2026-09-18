#!/usr/bin/env bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# The verdict for the engineering-design instance. Two surfaces, TWO ARMS EACH,
# neither of which ever entered selection.
#
#   surface 1  EngDesign v1 -- the same 61 tasks with nine grading exploits
#              closed (anti-reward-hacking probe).
#   surface 2  Frontier-Eng -- a different benchmark's verifiers, MINUS its own
#              EngDesign domain (those seven tasks are all in the evolve set).
#
# Both arms run in the same session from their own git worktrees:
#   arm base  = H_0            (runs/eng/frontier.json trajectory[0].commit)
#   arm champ = the incumbent  (runs/eng/frontier.json incumbent.commit)
# The paired difference is the whole measurement.
#
# Usage (from anywhere):
#   bash domains/eng/scripts/final_eval.sh v1        # EngDesign hardened
#   bash domains/eng/scripts/final_eval.sh frontier  # Frontier-Eng OOD
set -uo pipefail
DOM="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"          # domains/eng (main checkout)
REPO="$(cd "$DOM/../.." && pwd)"                                  # rrsi root
RUNS="${RRSI_RUNS_DIR:-$REPO/runs/eng}"
PY="${PYBIN:-${RRSI_AGENT_PYTHON:?set RRSI_AGENT_PYTHON}}"
FR_ROOT="${FROOT:-$DOM/Frontier-Engineering}"
MANIFEST="${FRONTIER_MANIFEST:-$DOM/data/frontier_manifest.json}"
SURFACE="${1:?surface: v1 | frontier}"
K="${K:-4}"

BASE_COMMIT="$(python3 -c "import json;print(json.load(open('$RUNS/frontier.json'))['trajectory'][0]['commit'])")"
CHAMP_COMMIT="$(python3 -c "import json;print(json.load(open('$RUNS/frontier.json'))['incumbent']['commit'])")"
echo "== arms: base=$BASE_COMMIT  champ=$CHAMP_COMMIT =="

if pgrep -f "[r]loop.py --domain eng" >/dev/null; then
  echo "FATAL: an eng evolution process is still running (touch $RUNS/STOP and wait)."; exit 1
fi
LOCK="$RUNS/.final_eval.lock"; mkdir -p "$RUNS"
exec 9>"$LOCK"
if ! flock -n 9; then echo "FATAL: another final_eval is already running (lock: $LOCK)."; exit 1; fi
echo "$SURFACE $$ $(date -Is)" >&9

export RRSI_RUNS_DIR="$RUNS" RRSI_SPLIT_PATH="$DOM/data/split_engd.json"
export GRADING_PYTHON="${GRADING_PYTHON:-$(readlink -f "$DOM/.venvs")/engdesign/bin/python}"
export WORKSPACE_BASE="$RUNS/workspaces" GATEWAY_PORT="${GATEWAY_PORT:-8994}" AGENT_PYTHON="$PY"
mkdir -p "$RUNS/logs" "$WORKSPACE_BASE"
bash "$DOM/scripts/gateway.sh" start || exit 1

arm_wt() {  # $1=arm  $2=commit  -> prints worktree path
  local wt="$RUNS/wt/final_$1"
  git -C "$REPO" worktree remove --force "$wt" >/dev/null 2>&1; rm -rf "$wt"
  git -C "$REPO" worktree prune
  git -C "$REPO" worktree add --detach "$wt" "$2" >/dev/null || { echo "worktree failed"; exit 1; }
  echo "$wt"
}

run_arm() {  # $1=arm name  $2=commit  $3=bench root  $4=job suffix
  local arm="$1" commit="$2" broot="$3" suffix="$4"
  local wt; wt="$(arm_wt "$arm" "$commit")"
  echo "-- arm $arm ($commit) on $(basename "$broot") from $wt"
  BENCH_ROOT="$broot" "$PY" "$wt/domains/eng/bench/run_tasks.py" \
      --job "final_${suffix}_${arm}" --n "$K" >> "$RUNS/logs/final_${suffix}_${arm}.log" 2>&1
  BENCH_ROOT="$broot" python3 "$wt/domains/eng/bench/verify.py" --job "final_${suffix}_${arm}" \
      >> "$RUNS/logs/final_${suffix}_${arm}.log" 2>&1
  BENCH_ROOT="$broot" python3 "$DOM/bench/score.py" --job "final_${suffix}_${arm}" --n "$K"
}

case "$SURFACE" in
  v1)
    [ -d "$DOM/engdesign_bench_v1" ] || (cd "$DOM" && bash scripts/build_testsurfaces.sh)
    run_arm base  "$BASE_COMMIT"  "$DOM/engdesign_bench_v1" v1
    run_arm champ "$CHAMP_COMMIT" "$DOM/engdesign_bench_v1" v1
    echo "== paired comparison (pass rate is the paper's metric; combined_score for resolution) =="
    BENCH_ROOT="$DOM/engdesign_bench_v1" python3 - "$K" <<'PY'
import sys, os
sys.path.insert(0, os.path.join(os.environ["RRSI_SPLIT_PATH"].rsplit("/data/", 1)[0], "bench"))
sys.path.insert(0, os.environ["RRSI_SPLIT_PATH"].rsplit("/data/", 1)[0])
import score as S
k = int(sys.argv[1]); ids = S.task_ids("evolve")
b = S.score_job("final_v1_base", ids, k); c = S.score_job("final_v1_champ", ids, k)
p = S.paired(b["per_task"], c["per_task"], ids)
print(f"  base : {b['total_passes']:3d}/{b['n_expected_trials']} passes ({b['pass_rate']:.4f})  score {b['mean_score']:.4f}  valid {b['valid_rate']:.3f}")
print(f"  champ: {c['total_passes']:3d}/{c['n_expected_trials']} passes ({c['pass_rate']:.4f})  score {c['mean_score']:.4f}  valid {c['valid_rate']:.3f}")
print(f"  pass-rate delta {c['pass_rate']-b['pass_rate']:+.4f}; paired combined_score delta {p['delta']:+.4f} t={p['t']:+.2f}")
print("  READ THIS AGAINST THE v0 SURFACE: if the champion's advantage is smaller here, part of what it learned was the closed exploits.")
PY
    ;;
  frontier)
    [ -f "$MANIFEST" ] || { echo "FATAL: no Frontier-Eng manifest at $MANIFEST (FRONTIER_MANIFEST); see domains/eng/README.md"; exit 1; }
    echo "NOTE: tasks whose family venv was never built are excluded from BOTH arms (paired); the count is reported."
    for arm in base champ; do
      commit=$([ "$arm" = base ] && echo "$BASE_COMMIT" || echo "$CHAMP_COMMIT")
      wt="$(arm_wt "$arm" "$commit")"
      "$PY" "$wt/domains/eng/bench/run_tasks_frontier.py" --job "final_fr_${arm}" --n 1 \
        --frontier-root "$FR_ROOT" --manifest "$MANIFEST" >> "$RUNS/logs/final_fr_${arm}.log" 2>&1
      python3 "$wt/domains/eng/bench/verify_frontier.py" --frontier-root "$FR_ROOT" --manifest "$MANIFEST" \
        --job "final_fr_${arm}" --out "$RUNS/final_fr_${arm}.json" >> "$RUNS/logs/final_fr_${arm}.log" 2>&1
    done
    python3 - "$RUNS" <<'PY'
import json, sys
R = sys.argv[1]
b = json.load(open(f"{R}/final_fr_base.json"))
c = {r["task"]: r for r in json.load(open(f"{R}/final_fr_champ.json"))}
w = l = t = skip = 0
for r in b:
    o = c.get(r["task"])
    if not o or r["combined_score"] is None or o["combined_score"] is None:
        skip += 1; continue
    rb, rc = (r["valid"] == 1.0, r["combined_score"]), (o["valid"] == 1.0, o["combined_score"])
    if rc > rb: w += 1
    elif rc < rb: l += 1
    else: t += 1
n = w + l + t
print(f"  champion vs base, within-task: {w} win / {t} tie / {l} loss  (n={n}, {skip} ungradeable in one or both arms)")
print(f"  win rate over decided tasks: {w/max(1, w+l):.3f}")
PY
    ;;
  *) echo "unknown surface: $SURFACE"; exit 2;;
esac
