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
"""Evaluate(H', D_evolve, k): the empirical score and cost of Eq. (estimate).

    S_hat(H) = 1/(k|D|) sum_x sum_j r(x, tau_x^(j))
    C_hat(H) = 1/(k|D|) sum_x sum_j c(tau_x^(j))

with r in [0,1] and c the policy tokens of a trajectory. A missing trial (crash,
timeout, infrastructure) contributes r = 0 with the full denominator, never an
absent slot, so a candidate cannot look better by destroying the trials it
finds hard.

Rewards may carry weights. Every coding / engineering trial has weight 1; a
Harvey LAB trial has reward = criteria passed / criteria total and weight =
criteria total, so S_hat is the fraction of criteria passed over all tasks, the
benchmark's own metric (Appendix, Harvey LAB).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class TaskResult:
    rewards: list[float]                  # one per trial, missing trials -> 0.0
    weights: list[float] = field(default_factory=list)   # default 1.0 each
    tokens: list = field(default_factory=list)           # int or None per trial
    missing: int = 0
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.weights:
            self.weights = [1.0] * len(self.rewards)
        if not self.tokens:
            self.tokens = [None] * len(self.rewards)

    @property
    def mean(self) -> float:
        w = sum(self.weights)
        return (sum(r * x for r, x in zip(self.rewards, self.weights)) / w) if w else 0.0


@dataclass
class EvalResult:
    job: str
    k: int
    per_task: dict                        # task_id -> TaskResult
    S: float                              # S_hat
    C: float | None                       # C_hat (None if no token counts)
    n_expected: int
    missing: int
    extra: dict = field(default_factory=dict)   # domain aggregates (pass counts, ...)

    def to_json(self) -> dict:
        d = {"job": self.job, "k": self.k, "S": self.S, "C": self.C,
             "n_expected": self.n_expected, "missing": self.missing,
             "extra": self.extra,
             "per_task": {t: asdict(r) for t, r in self.per_task.items()}}
        return d

    @classmethod
    def from_json(cls, d: dict) -> "EvalResult":
        per = {t: TaskResult(**r) for t, r in d["per_task"].items()}
        return cls(job=d["job"], k=d["k"], per_task=per, S=d["S"], C=d["C"],
                   n_expected=d["n_expected"], missing=d["missing"],
                   extra=d.get("extra") or {})

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=1))

    @classmethod
    def load(cls, path: Path | str) -> "EvalResult":
        return cls.from_json(json.loads(Path(path).read_text()))


def aggregate(job: str, k: int, per_task: dict, extra: dict | None = None) -> EvalResult:
    """Fold per-task trial records into S_hat and C_hat."""
    num = den = 0.0
    toks = []
    missing = 0
    for tr in per_task.values():
        for r, w in zip(tr.rewards, tr.weights):
            num += r * w
            den += w
        toks += [x for x in tr.tokens if isinstance(x, (int, float)) and x > 0]
        missing += tr.missing
    return EvalResult(job=job, k=k, per_task=per_task,
                      S=(num / den) if den else 0.0,
                      C=(sum(toks) / len(toks)) if toks else None,
                      n_expected=len(per_task) * k, missing=missing,
                      extra=dict(extra or {}))


def evaluate(domain, root: Path, runs_dir: Path, job: str, ids: list[str],
             k: int, log_prefix: str = "") -> EvalResult:
    """Run H' (the harness checked out under `root`) on `ids` with k trials and
    score it. Resume-safe: the domain runner fills only missing trials."""
    domain.run(root, runs_dir, job, ids, k, log_prefix=log_prefix)
    per_task, extra = domain.score(runs_dir, job, ids, k)
    return aggregate(job, k, per_task, extra)


def relative_cost_change(C_cand: float | None, C_inc: float | None) -> float:
    """Delta C = (C' - C_t) / C_t; 0 when either side has no token count."""
    if not C_cand or not C_inc:
        return 0.0
    return (C_cand - C_inc) / C_inc
