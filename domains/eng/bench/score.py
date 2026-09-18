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
"""Scoring / aggregation for a job dir, and the statistics the accept rule uses.

THREE AXES, all from the frozen verifier, each doing a different job:

  combined_score  continuous in [0,1] (raw_score / score_max). The PRIMARY
                  statistic. It is what the paired test runs on, because a
                  design that goes from 0.42 to 0.71 has visibly improved even
                  though `passed` never flipped -- and on a 61-task suite that
                  extra resolution is the difference between seeing an effect
                  and not.
  passed          binary, at the task's own gate. The HEADLINE and the hard
                  correctness floor: a candidate is
                  never accepted if it costs passes, no matter what the
                  continuous mean says.
  valid           the candidate met the task's hard constraints. The DAMAGE
                  axis. A scaffold can raise the average by pushing designs into
                  aggressive territory that more often violates a constraint
                  outright; the continuous mean alone will not show that, and it
                  is exactly what fails on the hardened test surface.

MISSING TRIALS COUNT AS ZERO, never as absent. A crashed, timed-out or
empty-payload trial is a failure of the scaffold, and dropping it from the
denominator would pay a candidate for the trials it destroyed.

Acceptance is a PAIRED comparison: per-task differences d_i on the SAME tasks,
summarised as mean(d) and se = sd(d)/sqrt(n). Pairing removes between-task
difficulty variance, which here runs from tasks solved every time to tasks no
model has ever solved.
"""

import json
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

JOBS = Path(config.RUNS) / "jobs"
BENCH = Path(config.BENCH_TASKS)

METRIC_PATH = Path(config.REPO) / "data" / "metric.json"
METRIC = json.loads(METRIC_PATH.read_text()) if METRIC_PATH.is_file() else {}
HARD_DROP_DELTA = METRIC.get("hard_drop_delta", 0.50)


def all_task_ids() -> list[str]:
    return sorted(td.name for td in BENCH.glob("*/*")
                  if (td / "frontier_eval").is_dir())


def split() -> dict:
    p = Path(config.SPLIT_PATH)
    if p.is_file():
        return json.loads(p.read_text())
    return {"evolve": [{"id": i} for i in all_task_ids()]}


def task_ids(set_name: str = "evolve") -> list[str]:
    return [t["id"] for t in split()[set_name]]


def trial_records(job: str, task_id: str) -> list[dict]:
    """One record per trial DIRECTORY that exists. A trial with a meta.json but
    no verdict has not been graded yet and is reported separately by score_job."""
    out = []
    tdir = JOBS / job / task_id
    if not tdir.is_dir():
        return out
    for trial in sorted(tdir.glob("t*")):
        if not (trial / "meta.json").is_file():
            continue
        v = trial / "verdict.json"
        if not v.is_file():
            out.append({"graded": False})
            continue
        try:
            d = json.loads(v.read_text())
        except Exception:  # noqa: BLE001
            d = {}
        meta = {}
        try:
            meta = json.loads((trial / "meta.json").read_text())
        except Exception:  # noqa: BLE001
            pass
        out.append({"graded": True,
                    "score": float(d.get("combined_score") or 0.0),
                    "passed": bool(d.get("passed")),
                    "valid": float(d.get("valid") or 0.0),
                    "vstatus": d.get("status"),
                    "no_payload": bool(meta.get("no_payload")),
                    "tokens": meta.get("total_tokens"),
                    "wall_s": meta.get("wall_s"),
                    "status": meta.get("status")})
    return out


def score_job(job: str, ids: list[str], n: int) -> dict:
    """Aggregate one job over a task list at n trials/task.

    Every (task, trial) slot in ids x n is accounted for: a slot with no graded
    record contributes a zero on all three axes.
    """
    per_task: dict[str, dict] = {}
    missing = ungraded = 0
    toks, walls = [], []
    for tid in ids:
        recs = trial_records(job, tid)
        graded = [r for r in recs if r.get("graded")]
        ungraded += len(recs) - len(graded)
        scores = [r["score"] for r in graded][:n]
        passes = [1.0 if r["passed"] else 0.0 for r in graded][:n]
        valids = [r["valid"] for r in graded][:n]
        nopay = [1.0 if r["no_payload"] else 0.0 for r in graded][:n]
        hole = n - len(scores)
        missing += hole
        scores += [0.0] * hole
        passes += [0.0] * hole
        valids += [0.0] * hole
        nopay += [1.0] * hole
        for r in graded:
            if isinstance(r.get("tokens"), int):
                toks.append(r["tokens"])
            if isinstance(r.get("wall_s"), (int, float)):
                walls.append(r["wall_s"])
        per_task[tid] = {
            "mean": sum(scores) / n,
            "scores": scores,
            "pass_rate": sum(passes) / n,
            "passes": int(sum(passes)),
            "valid_rate": sum(valids) / n,
            "no_payload_rate": sum(nopay) / n,
        }
    n_exp = len(ids) * n
    return {
        "job": job,
        "n_tasks": len(ids),
        "n_attempts": n,
        "n_expected_trials": n_exp,
        "missing_trials": missing,
        "ungraded_trials": ungraded,
        "mean_score": (sum(t["mean"] for t in per_task.values()) / len(ids)) if ids else 0.0,
        "total_passes": sum(t["passes"] for t in per_task.values()),
        "pass_rate": (sum(t["pass_rate"] for t in per_task.values()) / len(ids)) if ids else 0.0,
        "valid_rate": (sum(t["valid_rate"] for t in per_task.values()) / len(ids)) if ids else 0.0,
        "no_payload_rate": (sum(t["no_payload_rate"] for t in per_task.values()) / len(ids)) if ids else 0.0,
        "mean_tokens_per_trial": (sum(toks) / len(toks)) if toks else None,
        "mean_wall_s": (sum(walls) / len(walls)) if walls else None,
        "per_task": per_task,
    }


def paired(champ: dict, cand: dict, ids: list[str], key: str = "mean") -> dict:
    """Paired per-task difference statistics on the same tasks."""
    d = [cand[i][key] - champ[i][key] for i in ids if i in champ and i in cand]
    n = len(d)
    if n < 2:
        return {"n": n, "delta": 0.0, "se": 0.0, "t": 0.0,
                "hard_drops": [], "big_gains": [], "improved": [], "worsened": []}
    delta = sum(d) / n
    se = st.stdev(d) / math.sqrt(n)
    used = [i for i in ids if i in champ and i in cand]
    return {
        "n": n,
        "delta": delta,
        "se": se,
        "t": (delta / se) if se > 1e-12 else 0.0,
        "hard_drops": [i for i in used
                       if cand[i][key] - champ[i][key] <= -HARD_DROP_DELTA],
        "big_gains": [i for i in used
                      if cand[i][key] - champ[i][key] >= HARD_DROP_DELTA],
        "improved": [i for i in used if cand[i][key] > champ[i][key]],
        "worsened": [i for i in used if cand[i][key] < champ[i][key]],
    }


def screen_subset(ids: list[str], seed: int, k: int, champ_per_task: dict) -> list[str]:
    """Rotating STRATIFIED subsample for the cheap screen.

    An unbiased draw would spend most of its trials on tasks that cannot move.
    The strata are taken from the champion's own per-task continuous mean:
    half the budget on the movable middle (where a candidate's effect lives), a
    quarter on tasks the champion nearly always passes (a regression tripwire),
    and the rest on the floor (where a real breakthrough would show first).
    The window rotates with the iteration so the screen is not the same 30 tasks
    every round, which would quietly turn into a second training set.
    """
    def band(i):
        m = (champ_per_task.get(i) or {}).get("mean", 0.0)
        return "floor" if m <= 0.05 else ("solid" if m >= 0.95 else "movable")

    groups = {"movable": [], "solid": [], "floor": []}
    for i in ids:
        groups[band(i)].append(i)
    want = {"movable": k // 2, "solid": k // 4, "floor": k - k // 2 - k // 4}
    out: list[str] = []
    for g, need in want.items():
        pool = groups[g]
        if not pool:
            continue
        off = (seed * 7) % len(pool)
        out += [pool[(off + j) % len(pool)] for j in range(min(need, len(pool)))]
    # Backfill from whatever is left if a stratum was short.
    if len(out) < min(k, len(ids)):
        rest = [i for i in ids if i not in out]
        off = (seed * 13) % max(1, len(rest))
        out += [rest[(off + j) % len(rest)] for j in range(min(k - len(out), len(rest)))]
    return sorted(set(out))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--n", type=int, default=config.N_ATTEMPTS)
    ap.add_argument("--set", default="evolve")
    a = ap.parse_args()
    res = score_job(a.job, task_ids(a.set), a.n)
    print(json.dumps({k: v for k, v in res.items() if k != "per_task"}, indent=1))
