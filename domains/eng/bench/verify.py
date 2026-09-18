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
"""Grade a job's trials with each task's FROZEN verifier.

For every runs/jobs/<job>/<task>/t<k>/payload.py this copies the pristine task
dir to a throwaway location, drops the payload in at the task's own
`candidate_destination`, runs `frontier_eval/run_eval.sh` under the grading venv,
and writes the resulting metrics as verdict.json next to the payload.

Three properties this file exists to guarantee:

  * The benchmark tree is NEVER written to. Verifiers write metrics.json,
    artifacts.json and scratch files into their working directory; running them
    in place would mutate the pristine tree and make later trials incomparable.
  * A trial that produced no payload is still GRADED, and scores zero. Dropping
    it would pay a candidate for the trials it destroyed.
  * The grader always leaves a verdict behind. A verifier that crashes, hangs or
    writes no metrics yields an explicit zero with a machine-readable reason
    rather than an absent file, because an absent file is indistinguishable from
    "not graded yet" and would silently shrink the denominator.

Idempotent: a trial that already has a verdict.json is skipped.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

JOBS = Path(config.RUNS) / "jobs"
BENCH = Path(config.BENCH_TASKS)


def task_dirs() -> dict[str, Path]:
    return {td.name: td for td in BENCH.glob("*/*") if (td / "frontier_eval").is_dir()}


def grade_one(task_dir: Path, payload: Path, timeout: int) -> dict:
    t0 = time.time()
    verdict = {"combined_score": 0.0, "valid": 0.0, "passed": False,
               "status": "ok", "error": None, "details": None}
    with tempfile.TemporaryDirectory(prefix="engd_grade_") as tmp:
        work = Path(tmp) / task_dir.name
        try:
            shutil.copytree(task_dir, work, symlinks=True)
        except Exception as e:  # noqa: BLE001
            verdict.update(status="ERR_COPY", error=repr(e)[:300])
            return verdict
        dest = work / (work / "frontier_eval" / "candidate_destination.txt").read_text().strip()
        dest.parent.mkdir(parents=True, exist_ok=True)
        if payload.is_file():
            shutil.copy(payload, dest)
        else:
            dest.write_text("PAYLOAD = {}\n")
        cmd = ["bash", str(work / "frontier_eval" / "run_eval.sh"),
               config.GRADING_PYTHON, str(work), str(dest)]
        env = {**os.environ, "ENGDESIGN_TASK_TIMEOUT_S": str(max(30, timeout - 30))}
        try:
            subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            verdict.update(status="ERR_TIMEOUT",
                           error=f"verifier exceeded {timeout}s")
            verdict["wall_s"] = round(time.time() - t0, 1)
            return verdict
        except Exception as e:  # noqa: BLE001
            verdict.update(status="ERR_EXEC", error=repr(e)[:300])
            verdict["wall_s"] = round(time.time() - t0, 1)
            return verdict
        arts = work / "artifacts.json"
        details = None
        if arts.is_file():
            try:
                details = (json.loads(arts.read_text()) or {}).get("details")
            except Exception:  # noqa: BLE001
                details = None
        m = work / "metrics.json"
        if not m.is_file():
            err = work / "eval.stderr.txt"
            verdict.update(status="ERR_NO_METRICS",
                           error=(err.read_text()[-400:] if err.is_file() else ""))
            verdict["wall_s"] = round(time.time() - t0, 1)
            return verdict
        try:
            metrics = json.loads(m.read_text())
        except Exception as e:  # noqa: BLE001
            verdict.update(status="ERR_PARSE", error=repr(e)[:300])
            verdict["wall_s"] = round(time.time() - t0, 1)
            return verdict

    score = metrics.get("combined_score")
    verdict.update(
        combined_score=float(score) if isinstance(score, (int, float)) else 0.0,
        valid=float(metrics.get("valid") or 0.0),
        passed=bool(metrics.get("passed")),
        raw_score=metrics.get("raw_score"),
        score_max=metrics.get("score_max"),
        error=metrics.get("error"),
        # The verifier's own per-sub-criterion breakdown ("iou_score 0.83 vs
        # pass_threshold 0.8", "phase margin correct, settling time wrong").
        # This is the single most informative artefact for the analyst: it says
        # WHICH part of the design was wrong, which a scalar score cannot.
        details=details,
    )
    # Clamp: a verifier that reports outside [0,1] (a sentinel, a ratio that
    # beat the reference) must not distort the mean across tasks.
    verdict["combined_score"] = max(0.0, min(1.0, verdict["combined_score"]))
    verdict["wall_s"] = round(time.time() - t0, 1)
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--jobs", type=int, default=config.VERIFY_JOBS)
    ap.add_argument("--timeout", type=int, default=config.VERIFY_TIMEOUT)
    ap.add_argument("--force", action="store_true", help="re-grade existing verdicts")
    a = ap.parse_args()

    tds = task_dirs()
    todo = []
    jdir = JOBS / a.job
    for tdir in sorted(jdir.glob("*/t*")):
        tid = tdir.parent.name
        if tid not in tds:
            continue
        if not (tdir / "meta.json").is_file():
            continue                      # trial never ran; score_job counts it 0
        if (tdir / "verdict.json").is_file() and not a.force:
            continue
        todo.append((tds[tid], tdir))
    print(f"[verify] job={a.job} to grade: {len(todo)}", flush=True)
    if not todo:
        return

    def work(item):
        td, tdir = item
        v = grade_one(td, tdir / "payload.py", a.timeout)
        (tdir / "verdict.json").write_text(json.dumps(v, indent=1))
        return tdir.parent.name, v

    bad = 0
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for tid, v in ex.map(work, todo):
            if v["status"] != "ok":
                bad += 1
                print(f"  [{v['status']}] {tid}: {str(v['error'])[:120]}", flush=True)
    print(f"[verify] done; non-ok verifier runs: {bad}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
