#!/usr/bin/env bash
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# H_0 and the incumbent on SWE-bench Verified, each from its own worktree, one
# attempt per instance; then the resolve rates over the full 500-instance
# denominator. Usage (from anywhere):
#   bash domains/coding/scripts/swe_eval.sh [n_concurrent]
set -euo pipefail
DOM="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; REPO="$(cd "$DOM/../.." && pwd)"
RUNS="${RRSI_RUNS_DIR:-$REPO/runs/coding}"; N="${1:-8}"
DATASET="${SWE_DATASET:-swe-bench/swe-bench-verified}"
BASE="$(python3 -c "import json;print(json.load(open('$RUNS/frontier.json'))['trajectory'][0]['commit'])")"
CHAMP="$(python3 -c "import json;print(json.load(open('$RUNS/frontier.json'))['incumbent']['commit'])")"
export RRSI_CODING_VENV="${RRSI_CODING_VENV:-$DOM/.venv}"
export MODEL="${MODEL:-$(python3 -c "import json;print(json.load(open('$DOM/rrsi.json')).get('policy_model','vertex_ai/claude-opus-4-8'))")}"
for arm in "base:$BASE" "best:$CHAMP"; do
  name="${arm%%:*}"; commit="${arm#*:}"; wt="$RUNS/worktrees/swe_$name"
  git -C "$REPO" worktree remove --force "$wt" 2>/dev/null || true
  git -C "$REPO" worktree add --detach "$wt" "$commit" >/dev/null
  echo "[swe_eval] arm $name = $commit"
  (cd "$wt/domains/coding" && scripts/run_eval.sh "swe_$name" "$DATASET" 1 "$N" --jobs-dir "$RUNS/jobs")
done
python3 "$DOM/scripts/swe_summary.py" "$RUNS/jobs/swe_base" "$RUNS/jobs/swe_best"
