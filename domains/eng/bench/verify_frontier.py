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
"""Grade a job's trials with a Frontier-Engineering task's own frozen verifier.

Kept SEPARATE from bench/verify.py on purpose: that file is called by the
running evolution loop, and a bug introduced there would silently corrupt an
in-flight candidate's measurement. This one only ever runs for the final
out-of-distribution eval.

The contract is the same shape as EngDesign's, because EngDesign was built into
the Frontier layout in the first place: every task carries a `frontier_eval/`
folder naming its candidate destination, its eval command, its working
directory, and the venv the command must run under. The differences are that the
command is per-task rather than a shared wrapper, and that the venv is per
FAMILY (`frontier-v1-main`, `frontier-eval-driver`, ...) rather than one shared
grading environment.

  metrics.json -> {combined_score, valid, ...}

TWO THINGS THE CALLER MUST KNOW.

`combined_score` here is in each task's NATIVE UNITS (m/s, Mbps, /100, ...) and
upstream is explicit that it must never be averaged across tasks. So this file
reports per-task scores and leaves aggregation to the caller, which must compare
WITHIN a task (win / tie / loss against a fixed reference) rather than across
them. A mean over these numbers is meaningless.

A task whose family venv was never built fails with a clear marker rather than a
zero, because "the environment is missing" and "the agent's program was bad" are
different facts and averaging them together would understate the scaffold.
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

# The seven tasks of Frontier's own EngDesign domain are all in our evolve set,
# so they are not out-of-distribution and must never enter this eval.
EXCLUDED_DOMAINS = {"EngDesign"}


def _read(p: Path, default: str = "") -> str:
    return p.read_text(encoding="utf-8").strip() if p.is_file() else default


def load_manifest(manifest: Path) -> dict[str, dict]:
    """Per-task {task_rel, env_name, eval_command, eval_cwd, candidate_destination}.

    Read from the OFFICIAL converter's output rather than re-derived here. Two
    of the four fields cannot be recovered from the task directory at all:
    `env_name` lives in `frontier_eval/conf/batch/v1.yaml` (most tasks have no
    `env_name.txt`), and `eval_command` carries a `{repo_root}` placeholder that
    points outside the task. Re-implementing that resolution would be a second
    source of truth for which venv grades which task -- and a wrong venv does
    not error, it silently scores the task badly.
    """
    out = {}
    for t in json.loads(manifest.read_text()):
        fr = t.get("frontier") or {}
        rel = fr.get("task_rel") or ""
        if not rel or rel.split("/")[0] in EXCLUDED_DOMAINS:
            continue
        out[rel.replace("/", "__")] = fr
    return out


def venv_python(froot: Path, env_name: str) -> Path:
    return froot / ".venvs" / env_name / "bin" / "python"


def grade_one(froot: Path, fr: dict, candidate: Path, timeout: int) -> dict:
    t0 = time.time()
    rel = fr["task_rel"]
    task_dir = froot / "benchmarks" / rel
    env_name = fr.get("env_name") or "frontier-eval-driver"
    py = venv_python(froot, env_name)
    v = {"task": rel, "env_name": env_name,
         "combined_score": None, "valid": None, "status": "ok", "error": None}
    if not py.is_file():
        v.update(status="ERR_NO_VENV",
                 error=f"family venv not built: {py} (run setup_v1_task_envs.sh)")
        return v

    with tempfile.TemporaryDirectory(prefix="fr_grade_") as tmp:
        work = Path(tmp) / task_dir.name
        try:
            shutil.copytree(task_dir, work, symlinks=True)
        except Exception as e:  # noqa: BLE001
            v.update(status="ERR_COPY", error=repr(e)[:300])
            return v
        dest_rel = fr.get("candidate_destination") or ""
        if not dest_rel:
            v.update(status="ERR_NO_DEST", error="no candidate_destination in manifest")
            return v
        dest = work / dest_rel
        if candidate is not None and candidate.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(candidate, dest)
        # else: the shipped initial_program stays in place, which is exactly the
        # reference-replay case (grade the baseline the benchmark ships).

        cmd_tmpl = fr.get("eval_command") or ""
        if not cmd_tmpl:
            v.update(status="ERR_NO_CMD", error="no eval_command in manifest")
            return v
        # {repo_root} points at SHARED evaluator code outside the task dir (a
        # whole domain's tasks often share one evaluate_unified.py), so it must
        # resolve to the real checkout, not to our throwaway copy.
        cmd = (cmd_tmpl.replace("{python}", str(py))
                       .replace("{repo_root}", str(froot))
                       .replace("{benchmark}", str(work))
                       .replace("{candidate}", str(dest)))
        cwd_rel = fr.get("eval_cwd") or "."
        cwd = (work / cwd_rel).resolve()
        try:
            r = subprocess.run(cmd, shell=True, cwd=str(cwd), capture_output=True,
                               text=True, timeout=timeout,
                               env={**os.environ, "PYTHONNOUSERSITE": "1"})
        except subprocess.TimeoutExpired:
            v.update(status="ERR_TIMEOUT", error=f"verifier exceeded {timeout}s")
            v["wall_s"] = round(time.time() - t0, 1)
            return v
        except Exception as e:  # noqa: BLE001
            v.update(status="ERR_EXEC", error=repr(e)[:300])
            v["wall_s"] = round(time.time() - t0, 1)
            return v

        m = None
        for cand in (cwd / "metrics.json", work / "metrics.json"):
            if cand.is_file():
                m = cand
                break
        if m is None:
            v.update(status="ERR_NO_METRICS",
                     error=(r.stderr or r.stdout)[-500:], returncode=r.returncode)
            v["wall_s"] = round(time.time() - t0, 1)
            return v
        try:
            metrics = json.loads(m.read_text())
        except Exception as e:  # noqa: BLE001
            v.update(status="ERR_PARSE", error=repr(e)[:300])
            v["wall_s"] = round(time.time() - t0, 1)
            return v

    v.update(combined_score=metrics.get("combined_score"),
             valid=metrics.get("valid"),
             metrics={k: metrics[k] for k in list(metrics)[:20]},
             returncode=r.returncode)
    v["wall_s"] = round(time.time() - t0, 1)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontier-root", required=True)
    ap.add_argument("--manifest", default=os.environ.get("FRONTIER_MANIFEST", "data/frontier_manifest.json"),
                    help="output of scripts/frontier_eng/frontier_to_apex.py")
    ap.add_argument("--job", default="", help="grade a job's payloads; omit to "
                                              "replay each task's shipped baseline")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--tasks", default="", help="comma-separated subset")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    froot = Path(a.frontier_root).resolve()
    tasks = load_manifest(Path(a.manifest))
    if a.tasks:
        want = {t.strip() for t in a.tasks.split(",") if t.strip()}
        tasks = {k: v for k, v in tasks.items()
                 if k in want or k.split("__")[-1] in want}
    print(f"[fverify] {len(tasks)} tasks (EngDesign domain excluded), "
          f"root={froot}", flush=True)

    def work(item):
        key, td = item
        cand = None
        if a.job:
            p = JOBS / a.job / key / "t0" / "payload.py"
            cand = p if p.is_file() else None
        return grade_one(froot, td, cand, a.timeout)

    rows = []
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for v in ex.map(work, sorted(tasks.items())):
            rows.append(v)
            print(f"  {v['task']:52s} score={str(v['combined_score']):>12s} "
                  f"valid={str(v['valid']):>5s} {v['status']}"
                  + (f" :: {str(v['error'])[:70]}" if v["error"] else ""),
                  flush=True)

    scored = [r for r in rows if r["combined_score"] is not None]
    novenv = [r for r in rows if r["status"] == "ERR_NO_VENV"]
    print(f"\n[fverify] produced a score: {len(scored)}/{len(rows)} | "
          f"missing family venv: {len(novenv)} | "
          f"other failures: {len(rows) - len(scored) - len(novenv)}")
    print("[fverify] scores are per-task NATIVE UNITS -- do not average them; "
          "compare within a task against a fixed reference.")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1, default=str))
        print(f"[fverify] wrote {a.out}")


if __name__ == "__main__":
    main()
