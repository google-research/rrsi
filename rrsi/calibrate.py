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
"""Estimate the noise band delta by repeatedly evaluating the unchanged base
harness (Sec. stability-aware acceptance).

delta bounds |S_hat(H) - S_hat(H)| between two independent evaluations of the
SAME harness. With R >= 2 independent base evaluations the null difference is
observed directly; with a single k-trial evaluation the standard error of S_hat
is bootstrapped over trials within each task, and the null difference of two
independent evaluations has standard deviation sqrt(2) * se. Either way

    delta = z * sd(null Delta S),   z = cfg.delta_z (2.0 by default),

so an unchanged harness clears the floor S* - delta about 97.5% of the time
and a gain must exceed the band before the L1 cost rule treats it as real.
"""

from __future__ import annotations

import json
import math
import random
import statistics as st
from pathlib import Path

from .evaluate import EvalResult


def bootstrap_se(ev: EvalResult, reps: int = 2000, seed: int = 7) -> float:
    """se(S_hat) by resampling trials within each task (weights respected)."""
    rng = random.Random(seed)
    tasks = [tr for tr in ev.per_task.values() if tr.rewards]
    vals = []
    for _ in range(reps):
        num = den = 0.0
        for tr in tasks:
            n = len(tr.rewards)
            for _j in range(n):
                i = rng.randrange(n)
                num += tr.rewards[i] * tr.weights[i]
                den += tr.weights[i]
        vals.append(num / den if den else 0.0)
    return st.pstdev(vals) if len(vals) > 1 else 0.0


def pooled(evals: list[EvalResult]) -> EvalResult:
    """Concatenate the trials of several evaluations of the same harness."""
    from .evaluate import TaskResult, aggregate
    per: dict = {}
    for ev in evals:
        for t, tr in ev.per_task.items():
            p = per.setdefault(t, TaskResult(rewards=[], weights=[], tokens=[]))
            p.rewards += tr.rewards
            p.weights += tr.weights
            p.tokens += tr.tokens
            p.missing += tr.missing
    return aggregate("pooled", sum(e.k for e in evals), per)


def calibrate(evals: list[EvalResult], z: float = 2.0, reps: int = 2000) -> dict:
    """delta from one or more evaluations of the base harness."""
    if not evals:
        raise ValueError("no base evaluations")
    if len(evals) >= 2:
        scores = [e.S for e in evals]
        diffs = [abs(a - b) for i, a in enumerate(scores) for b in scores[i + 1:]]
        sd_null = st.stdev(scores) * math.sqrt(2) if len(scores) > 1 else 0.0
        method = "repeated base evaluations"
        observed = {"S_per_eval": scores, "max_abs_diff": max(diffs)}
    else:
        sd_null = 0.0
        method = "bootstrap over trials of one base evaluation"
        observed = {}
    pooled_ev = pooled(evals)
    se = bootstrap_se(pooled_ev, reps=reps)
    sd_boot = math.sqrt(2) * se * math.sqrt(pooled_ev.k / evals[0].k)
    # With repeated evals prefer the direct observation unless it is degenerate.
    sd_use = sd_null if (len(evals) >= 2 and sd_null > 0) else sd_boot
    return {"delta": round(z * sd_use, 6), "z": z, "sd_null": round(sd_use, 6),
            "sd_null_bootstrap": round(sd_boot, 6), "se_bootstrap": round(se, 6),
            "method": method, "n_evals": len(evals), "k": evals[0].k,
            "n_tasks": len(pooled_ev.per_task), "S_base": round(evals[0].S, 6),
            "C_base": evals[0].C, **observed}


def write(path: Path | str, cal: dict) -> None:
    Path(path).write_text(json.dumps(cal, indent=1))


def read_delta(path: Path | str) -> float | None:
    p = Path(path)
    if not p.exists():
        return None
    return float(json.loads(p.read_text())["delta"])
