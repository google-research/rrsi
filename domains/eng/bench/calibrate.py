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
"""Calibrate the metric constants from a completed baseline job.

Nothing about the accept rule is guessed. The noise scale of this benchmark is
an empirical fact, and it decides whether an iteration's move is signal or a
coin flip. Getting it wrong fails in one of two directions: accepting on a
statistic the null itself clears ratchets the training score upward on noise,
while a hard-drop limit borrowed from a different metric makes acceptance
structurally impossible (most candidates identical to the champion would be
vetoed).

Four questions:

1. HEADROOM. What fraction of tasks is pinned at 0.0 or 1.0 across all trials?
   Those cannot move and contribute nothing to a paired comparison. Measured on
   BOTH axes, because they disagree here: a task can be immovable on `passed`
   while its continuous score wanders freely, and that wandering is exactly the
   resolution this benchmark was chosen for.

2. CONTINUOUS NOISE. What does a paired comparison between two runs of the SAME
   harness look like? Measured by HALF-SPLITTING the baseline's own trials, so
   the true difference is zero by construction and everything observed is noise.

3. PASS NOISE. The same half-split on the binary axis, which sets the pass-count
   noise band the hard correctness floor uses.

4. DAMAGE. The null distribution of hard drops, big gains, and their NET, by
   resampling each task from its own baseline rate.

Requires a baseline with n >= 4 (to split 2-vs-2).

Writes data/metric.json (consumed by bench/score.py and loop.py).

Usage: python3 bench/calibrate.py --job baseline --n 4 [--write]
"""

import argparse
import json
import math
import random
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))

import config  # noqa: E402
import score as S  # noqa: E402


def _axis_values(job: str, tid: str, n: int, axis: str) -> list[float]:
    recs = [r for r in S.trial_records(job, tid) if r.get("graded")]
    vals = [(r["score"] if axis == "score" else (1.0 if r["passed"] else 0.0))
            for r in recs][:n]
    return vals + [0.0] * max(0, n - len(vals))


def half_split_null(job: str, ids: list[str], n: int, axis: str) -> dict:
    """Paired delta between two disjoint halves of the same job's trials.

    Also returns the distribution of the ABSOLUTE per-task movement, which is
    what hard_drop_delta must clear: a fall only counts as "this task was
    destroyed" if it is larger than what an unchanged harness produces on its
    own.
    """
    half = n // 2
    d = []
    for tid in ids:
        v = _axis_values(job, tid, n, axis)
        a, b = v[:half], v[half:half * 2]
        if not a or not b:
            continue
        d.append(st.mean(a) - st.mean(b))
    if len(d) < 2:
        return {"n": len(d)}
    se = st.stdev(d) / math.sqrt(len(d))
    absd = sorted(abs(x) for x in d)

    def q(f):
        return absd[min(len(absd) - 1, int(f * len(absd)))]

    return {"axis": axis, "n": len(d), "delta": round(st.mean(d), 5),
            "se": round(se, 5), "t": round(st.mean(d) / se, 3) if se else 0.0,
            "half_trials_each": half,
            "abs_move_p90": round(q(0.90), 4), "abs_move_p95": round(q(0.95), 4),
            "abs_move_p99": round(q(0.99), 4), "abs_move_max": round(absd[-1], 4)}


def pass_count_null(job: str, ids: list[str], n: int, reps: int = 2000,
                    seed: int = 7) -> dict:
    """How far the TOTAL pass count of an unchanged harness wanders.

    The correctness floor is stated in passes, so it needs the spread of the
    pass count itself, not of a per-task mean. Each task is resampled from its
    own baseline pass rate, which makes the true effect zero by construction.
    """
    rng = random.Random(seed)
    rates = []
    for tid in ids:
        v = _axis_values(job, tid, n, "passed")
        rates.append(sum(v) / n)
    base = sum(r * n for r in rates)
    diffs = []
    for _ in range(reps):
        tot = sum(1 for r in rates for _ in range(n) if rng.random() < r)
        diffs.append(tot - base)
    diffs.sort()

    def q(f):
        return diffs[min(len(diffs) - 1, int(f * len(diffs)))]

    return {"sd": round(st.stdev(diffs), 2), "p05": round(q(0.05), 1),
            "p01": round(q(0.01), 1), "reps": reps}


def null_hard_drops(per: dict, n: int, hard_drop: float,
                    reps: int = 400, seed: int = 11) -> dict:
    """Distribution of hard DROPS, big GAINS, and their NET, when nothing changed.

    Why the NET matters and counting drops alone does not: with few trials the
    per-task score is coarse, so any candidate with a real effect moves tasks in
    BOTH directions, while a threshold derived from a no-op null counts only the
    downside. Such a limit penalises EFFECT SIZE rather than asymmetric damage.
    The net isolates "destroyed more than it fixed", which is what the guard was
    always meant to catch.
    """
    rng = random.Random(seed)
    drops, gains, nets = [], [], []
    for _ in range(reps):
        hd = bg = 0
        for v in per.values():
            scores = v["scores"]
            cand = st.mean([rng.choice(scores) for _ in range(n)])
            d = cand - v["mean"]
            if d <= -hard_drop:
                hd += 1
            elif d >= hard_drop:
                bg += 1
        drops.append(hd)
        gains.append(bg)
        nets.append(hd - bg)
    drops.sort()
    gains.sort()
    nets.sort()

    def q(arr, f):
        return arr[min(len(arr) - 1, int(f * len(arr)))]

    return {"drops_mean": round(st.mean(drops), 1),
            "drops_p95": q(drops, 0.95), "drops_p99": q(drops, 0.99),
            "gains_mean": round(st.mean(gains), 1),
            "net_mean": round(st.mean(nets), 1),
            "net_p95": q(nets, 0.95), "net_p99": q(nets, 0.99),
            "reps": reps, "hard_drop_delta": hard_drop}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--set", default="evolve")
    ap.add_argument("--n", type=int, default=config.N_ATTEMPTS)
    ap.add_argument("--write", action="store_true",
                    help="write data/metric.json (otherwise report only)")
    args = ap.parse_args()

    ids = S.task_ids(args.set)
    res = S.score_job(args.job, ids, args.n)
    per = res["per_task"]
    n_tasks = len(per)

    always_s = sum(1 for v in per.values() if v["mean"] >= 0.999)
    never_s = sum(1 for v in per.values() if v["mean"] <= 0.001)
    always_p = sum(1 for v in per.values() if v["pass_rate"] >= 0.999)
    never_p = sum(1 for v in per.values() if v["pass_rate"] <= 0.001)

    within = [st.stdev(v["scores"]) for v in per.values() if len(v["scores"]) > 1]
    sd_trial = st.median(within) if within else 0.0

    null_score = half_split_null(args.job, ids, args.n, "score")
    null_pass = half_split_null(args.job, ids, args.n, "passed")
    pcn = pass_count_null(args.job, ids, args.n)

    # A per-task fall only counts as "broke this task" if it exceeds what the
    # null ACTUALLY produces. Anchored on the measured p99 of the half-split's
    # absolute per-task movement rather than on a guessed constant or on the
    # within-task sd (which is 0 for the median task here, and would collapse
    # the threshold to the trial step -- far coarser than the metric warrants).
    hard_drop = max(0.25, round(null_score.get("abs_move_p99", 0.25) + 0.03, 2))
    nhd = null_hard_drops(per, args.n, hard_drop)

    # FLOOR THE DAMAGE GUARD AT 1, deliberately, and record why. The resampling
    # null draws from each task's four OBSERVED scores, so it cannot produce an
    # outcome the baseline never showed and therefore understates the true
    # trial-to-trial spread -- it returns net p99 = 0 here, i.e. "a candidate may
    # never break even one task more than it fixes". That is the exact shape of
    # the guard that made acceptance structurally impossible on the previous
    # project. One net destroyed task is slack for a single unlucky draw; two is
    # the asymmetric damage this guard exists to catch.
    max_net = max(1, nhd["net_p99"])

    out = {
        "source_job": args.job, "set": args.set, "n_attempts": args.n,
        "n_tasks": n_tasks,
        "missing_trials": res["missing_trials"],
        "mean_score": round(res["mean_score"], 4),
        "total_passes": res["total_passes"],
        "pass_rate": round(res["pass_rate"], 4),
        "valid_rate": round(res["valid_rate"], 4),
        "no_payload_rate": round(res["no_payload_rate"], 4),
        # headroom, on both axes
        "score_always": round(always_s / n_tasks, 4) if n_tasks else 0,
        "score_never": round(never_s / n_tasks, 4) if n_tasks else 0,
        "score_movable_fraction": round((n_tasks - always_s - never_s) / n_tasks, 4)
                                  if n_tasks else 0,
        "pass_always": round(always_p / n_tasks, 4) if n_tasks else 0,
        "pass_never": round(never_p / n_tasks, 4) if n_tasks else 0,
        "pass_movable_fraction": round((n_tasks - always_p - never_p) / n_tasks, 4)
                                 if n_tasks else 0,
        "median_within_task_sd": round(sd_trial, 4),
        # noise
        "null_paired_score": null_score,
        "null_paired_se": null_score.get("se"),
        "detectable_delta_at_t2": (round(2 * null_score["se"], 5)
                                   if null_score.get("se") else None),
        "null_paired_pass": null_pass,
        "null_pass_count": pcn,
        # The pass-count floor: how far below the best-ever pass count a
        # candidate may sit and still be considered. Set at the null's 1st
        # percentile so an unchanged harness clears it ~99% of the time.
        "noise_band_passes": max(2, int(round(abs(pcn["p01"])))),
        # damage
        "hard_drop_delta": hard_drop,
        "null_hard_drops": nhd,
        "max_net_hard_drops": max_net,
        "_max_net_hard_drops_note": (
            f"measured null net p99 = {nhd['net_p99']}, floored to {max_net}; "
            "the resampling null draws only from observed trial values and so "
            "understates the true spread"),
        "max_hard_drops": max(2, nhd["drops_p99"]),
    }
    print(json.dumps(out, indent=1))

    if null_score.get("se"):
        print(f"\n[calibrate] a t=2.0 accept needs delta >= "
              f"{2 * null_score['se']:.4f} combined_score "
              f"({2 * null_score['se'] * 100:.2f}pp)", flush=True)
    print(f"[calibrate] pass-count null: sd {pcn['sd']}, p01 {pcn['p01']} "
          f"-> noise_band_passes = {out['noise_band_passes']}", flush=True)
    print(f"[calibrate] null drops mean {nhd['drops_mean']} p99 {nhd['drops_p99']} | "
          f"gains mean {nhd['gains_mean']} | NET p99 {nhd['net_p99']} "
          f"-> max_net_hard_drops = {max_net} (floored)", flush=True)
    if out["score_movable_fraction"] < 0.5:
        print(f"[calibrate] WARNING only {out['score_movable_fraction']:.0%} of "
              f"tasks can move on the continuous axis "
              f"({out['score_never']:.0%} never, {out['score_always']:.0%} "
              f"always) -- limited room for a scaffold effect", flush=True)

    if args.write:
        (ROOT / "data" / "metric.json").write_text(json.dumps(out, indent=1))
        print("wrote data/metric.json")


if __name__ == "__main__":
    main()
