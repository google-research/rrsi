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
# Run the forked AgentHarness on a harbor dataset.
# Usage: run_eval.sh <job-name> [dataset] [n_attempts] [n_concurrent] [extra harbor flags...]
#   dataset default: terminal-bench/terminal-bench-2-1
#   extra flags: --jobs-dir <dir> (default <this checkout>/runs/jobs), -i <task> ...
# Model/infra invariants are pinned here (frozen; not evolvable).
set -euo pipefail
REPO="${RRSI_CODING_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"

JOB_NAME="${1:?job name required}"
DATASET="${2:-terminal-bench/terminal-bench-2-1}"
N_ATTEMPTS="${3:-1}"
N_CONCURRENT="${4:-4}"
shift $(( $# > 4 ? 4 : $# ))

JOBS_DIR="$REPO/runs/jobs"
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --jobs-dir) JOBS_DIR="$2"; shift 2;;
    *) EXTRA+=("$1"); shift;;
  esac
done

export PATH="$REPO/bin:$PATH"            # docker -> sudo -E docker shim
export PYTHONPATH="$(cd "$REPO/../.." && pwd)/third_party"   # so --agent harbor_terminus2:AgentHarness resolves
export VERTEXAI_PROJECT="${VERTEXAI_PROJECT:?set VERTEXAI_PROJECT (GCP project for Vertex AI)}"
export VERTEXAI_LOCATION="${VERTEXAI_LOCATION:-global}"
VENV="${RRSI_CODING_VENV:-$REPO/.venv}"

exec "$VENV/bin/harbor" run \
  -d "$DATASET" \
  --agent harbor_terminus2:AgentHarness \
  -m "${MODEL:-vertex_ai/gemini-3.5-flash}" \
  --agent-kwarg 'llm_kwargs={"num_retries": 10}' \
  -k "$N_ATTEMPTS" \
  -n "$N_CONCURRENT" \
  --jobs-dir "$JOBS_DIR" \
  --job-name "$JOB_NAME" \
  "${EXTRA[@]}"
