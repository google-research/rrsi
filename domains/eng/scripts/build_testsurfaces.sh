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
# Build the two verdict surfaces. Neither ever enters selection.
#
#   engdesign_bench_v1/  the SAME 61 tasks with nine grading exploits closed.
#                        The anti-reward-hacking probe: a scaffold that learned
#                        to game a verifier scores on v0 and drops here.
#   frontier_bench/      Frontier-Engineering MINUS its EngDesign domain. Those
#                        seven tasks (AM_02 AM_03 CY_03 WJ_01 XY_05 YJ_02 YJ_03)
#                        are all in the evolve set, so leaving them in would put
#                        training tasks in the OOD test.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"

echo "== EngDesign v1 (hardened) =="
rm -rf engdesign_bench_v1
cp -r engdesign_bench engdesign_bench_v1
for t in EngDesignOpen_AutomotiveEngineering/YX_01 \
         EngDesignOpen_ComputerArchitecture/XY_05 \
         EngDesignOpen_ComputerScience/Yiqi_01 \
         EngDesignOpen_ComputerScience/Yiqi_02 \
         EngDesignOpen_ComputerVision/AB_03 \
         EngDesignOpen_MachineLearning/HC_03 \
         EngDesignOpen_MechanicalEngineering/ZH_03 \
         EngDesignOpen_MechanicalEngineering/ZH_04 \
         EngDesignOpen_PidControllerDesign/XG_13; do
  # frontier_eval/ is OURS (built from the official tree); the hardening lives
  # in the task files. Keep our eval wiring, take their task, then restore the
  # two frontier_eval files v1 legitimately changes (copy_files for tasks whose
  # agent-visible set moved).
  rsync -a --exclude 'frontier_eval/evaluate_submission.py' \
           --exclude 'frontier_eval/run_eval.sh' \
           --exclude 'frontier_eval/eval_command.txt' \
           --exclude 'frontier_eval/env_name.txt' \
           --exclude 'frontier_eval/eval_cwd.txt' \
           --exclude 'frontier_eval/candidate_destination.txt' \
           --exclude 'frontier_eval/initial_program.txt' \
           "data/engdesign_v1/$t/" "engdesign_bench_v1/benchmarks/$t/"
done
echo "  tasks: $(find engdesign_bench_v1/benchmarks -maxdepth 2 -mindepth 2 -type d | wc -l)"
echo "  hardened dirs differing from v0: $(diff -rq engdesign_bench/benchmarks engdesign_bench_v1/benchmarks 2>/dev/null | grep -c differ)"

echo "== FrontierEng (minus the EngDesign domain) =="
rm -rf frontier_bench && mkdir -p frontier_bench/benchmarks
for d in Frontier-Engineering/benchmarks/*/; do
  name=$(basename "$d")
  [ "$name" = "EngDesign" ] && { echo "  SKIP domain EngDesign (7 tasks, all in the evolve set)"; continue; }
  cp -r "$d" "frontier_bench/benchmarks/$name"
done
echo "  tasks: $(find frontier_bench/benchmarks -maxdepth 2 -mindepth 2 -type d -name '*' | wc -l) (before frontier_eval filter)"
find frontier_bench/benchmarks -maxdepth 2 -mindepth 2 -type d | while read -r t; do
  [ -d "$t/frontier_eval" ] || echo "  (no frontier_eval, will be skipped) ${t#frontier_bench/benchmarks/}"
done | head -20
echo "  gradeable: $(find frontier_bench/benchmarks -maxdepth 3 -type d -name frontier_eval | wc -l)"
