# Copyright 2026 The rrsi Authors.
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

#!/usr/bin/env python3
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
"""Judge production runs with Harvey LAB's own scoring (per-criterion rubric).

For each runs/<label>/<task>/ with output/, calls harvey-labs
evaluation.scoring.score_rubric verbatim and writes scores.json next to the
output. Idempotent: skips tasks that already have scores.json.

Run under the harvey-labs venv:
  GOOGLE_GENAI_USE_VERTEXAI=true GOOGLE_CLOUD_PROJECT=<project> \
  GOOGLE_CLOUD_LOCATION=global \
  harvey-labs/.venv/bin/python workspace_judge.py <label> [--judge-model gemini-3.5-flash]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HARVEY_LAB_ROOT = os.environ.get("HARVEY_LAB_ROOT", "harvey-labs")
sys.path.insert(0, HARVEY_LAB_ROOT)

from evaluation.judge import Judge            # noqa: E402
from evaluation.scoring import score_rubric   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--judge-model", default="gemini-3.5-flash")
    ap.add_argument("--parallel", type=int, default=8)
    args = ap.parse_args()

    runs = os.path.join(os.environ.get("RRSI_JOBS_DIR", os.path.join(HERE, "runs")), args.label)
    judge = Judge(model=args.judge_model)
    names = sorted(os.listdir(runs))
    done = passed_tasks = 0
    for name in names:
        run_dir = os.path.join(runs, name)
        out = os.path.join(run_dir, "output")
        sc_path = os.path.join(run_dir, "scores.json")
        if not os.path.isdir(out):
            continue
        if os.path.exists(sc_path):
            done += 1
            passed_tasks += json.load(open(sc_path)).get("all_pass", 0)
            continue
        task_dir = os.path.join(HARVEY_LAB_ROOT, "tasks", name.replace("__", "/"))
        tj = json.load(open(os.path.join(task_dir, "task.json")))
        criteria = tj.get("criteria") or []
        try:
            res = score_rubric(criteria, run_dir, judge,
                               task_desc=tj.get("title", name),
                               parallel=args.parallel)
            payload = {
                "task": name,
                "all_pass": 1 if res.score == 1.0 else 0,
                "score": res.score,
                "passed": sum(1 for c in res.criteria_results if c.get("verdict") == "pass"),
                "total": len(res.criteria_results),
                "criteria": res.criteria_results,
            }
        except Exception as e:  # noqa: BLE001
            print(f"[judge-error] {name}: {e!r}", flush=True)
            continue
        json.dump(payload, open(sc_path, "w"), indent=1)
        done += 1
        passed_tasks += payload["all_pass"]
        print(f"[judged] {name}: {payload['passed']}/{payload['total']} "
              f"all_pass={payload['all_pass']}", flush=True)
    # aggregate
    aggs = []
    for name in names:
        p = os.path.join(runs, name, "scores.json")
        if os.path.exists(p):
            aggs.append(json.load(open(p)))
    n = len(aggs)
    crit_pass = sum(a["passed"] for a in aggs)
    crit_tot = sum(a["total"] for a in aggs)
    summary = {
        "label": args.label, "judge": args.judge_model,
        "n_judged": n,
        "all_pass_tasks": sum(a["all_pass"] for a in aggs),
        "all_pass_rate": round(sum(a["all_pass"] for a in aggs) / max(1, n), 4),
        "criterion_pass_rate": round(crit_pass / max(1, crit_tot), 4),
        "criteria_passed": crit_pass, "criteria_total": crit_tot,
    }
    json.dump(summary, open(os.path.join(runs, "SUMMARY.json"), "w"), indent=1)
    print(f"[judge] SUMMARY {json.dumps(summary)}", flush=True)


if __name__ == "__main__":
    main()
