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
"""Hyperparameters of RRSI (paper Appendix "Hyperparameters") plus the
engineering knobs a run needs that are not part of the method.

Symbols follow the paper:

  T          number of rounds, t = 0..T-1
  k          trials per task inside Evaluate (Eq. estimate)
  m          candidate harnesses drawn per round, |C_t| before screening
  b_min/b_max  bounds of the annealed L0 edit budget b_t (Eq. anneal)
  w          stall window of the exploration flag sigma_t (Eq. explore)
  m_draft    candidate slots reserved for exploratory edits when stalled
  delta      empirical noise band (Sec. stability-aware acceptance)
  beta0/beta1  L1 cost rule for gaining candidates (Eq. tokenbudget)
  w_s/w_c/w_n  shaped rule for candidates inside the noise band (Alg. 2 l.5)
  n_prune    window of the recent-yield summary g_t (Eq. yield / prune)

Units: scores S are fractions in [0,1] (Eq. estimate), cost C is mean policy
tokens per trial, Delta C is RELATIVE (Eq. tokenbudget), so beta0 = 0.10 means
"10% more tokens for free" and beta1 = 40 means "each +1pp of S buys +40%".
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class RRSIConfig:
    # ---- horizon and estimator -------------------------------------------
    T: int = 20
    k: int = 2
    m: int = 2
    # ---- proposal side (Algorithm 1) -------------------------------------
    b_min: int = 1
    b_max: int = 4
    w: int = 3
    m_draft: int = 1
    # ---- selection side (Algorithm 2) ------------------------------------
    delta: float | None = None      # None -> runs/<domain>/calibration.json
    delta_z: float = 2.0            # calibration: delta = z * sd(null Delta S)
    beta0: float = 0.10
    beta1: float = 40.0
    w_s: float = 100.0
    w_c: float = 15.0
    w_n: float = 0.5
    n_prune: int = 4
    # ---- engineering knobs (not part of the method) ----------------------
    repair_rounds: int = 5          # critic -> proposer repair attempts
    invalid_missing_frac: float = 0.15
    n_fail_traces: int = 22
    n_success_traces: int = 6
    eval_parallel: int = 1          # candidates evaluated concurrently
    proposer_model: str = "claude-opus-4-8"
    analyst_model: str = "claude-opus-4-8"
    critic_model: str = "claude-opus-4-8"
    notes: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | str, **overrides) -> "RRSIConfig":
        raw = json.loads(Path(path).read_text())
        known = {f.name for f in fields(cls)}
        kw = {k: v for k, v in raw.items() if k in known}
        kw["notes"] = {k: v for k, v in raw.items() if k not in known}
        kw.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**kw)

    def dump(self) -> dict:
        return asdict(self)
