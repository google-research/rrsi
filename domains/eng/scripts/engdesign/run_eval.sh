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
# Eval entrypoint for one EngDesign task, called by bench/verify.py as
#   run_eval.sh {python} {benchmark_dir} {candidate}
# Writes metrics.json (combined_score, valid) in the benchmark dir. Always exits
# 0: validity is reported in metrics.json.
set -uo pipefail

PYTHON_CMD="${1:?missing python}"
BENCHMARK_DIR="${2:?missing benchmark dir}"
CANDIDATE_PATH="${3:-}"

if [[ "${BENCHMARK_DIR}" != /* ]]; then
  BENCHMARK_DIR="$(cd "${BENCHMARK_DIR}" && pwd -P)"
fi
if [[ -n "${CANDIDATE_PATH}" && "${CANDIDATE_PATH}" != /* && ! -f "${CANDIDATE_PATH}" ]]; then
  CANDIDATE_PATH="${BENCHMARK_DIR}/${CANDIDATE_PATH}"
fi

TASK_TIMEOUT_S="${ENGDESIGN_TASK_TIMEOUT_S:-180}"
METRICS_JSON="${BENCHMARK_DIR}/metrics.json"

"${PYTHON_CMD}" "${BENCHMARK_DIR}/frontier_eval/evaluate_submission.py" \
  --candidate "${CANDIDATE_PATH}" \
  --benchmark-dir "${BENCHMARK_DIR}" \
  --metrics-out "${METRICS_JSON}" \
  --artifacts-out "${BENCHMARK_DIR}/artifacts.json" \
  --task-timeout-s "${TASK_TIMEOUT_S}" \
  > "${BENCHMARK_DIR}/eval.stdout.txt" 2> "${BENCHMARK_DIR}/eval.stderr.txt"

if [[ ! -f "${METRICS_JSON}" ]]; then
  printf '{"combined_score": 0.0, "valid": 0.0}\n' > "${METRICS_JSON}"
fi
exit 0
