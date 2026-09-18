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
"""Annealed L0 update budget, Eq. (anneal) of the paper:

    b_t = ceil( b_min + (b_max - b_min) * 1/2 (1 + cos(pi t / T)) ),  t = 0..T-1

Early rounds may bundle several coordinated edits in one candidate; late rounds
become sparse and attributable. The constraint bounds ||z_t||_0, the number of
independent edits active in one proposal, and nothing else: the set of
mechanisms the harness may eventually contain is not restricted.
"""

from __future__ import annotations

import math


def edit_budget(t: int, T: int, b_min: int, b_max: int) -> int:
    """b_t for round t (0-indexed) of a T-round run."""
    if T <= 0:
        return int(b_max)
    t = max(0, min(int(t), int(T)))
    v = b_min + (b_max - b_min) * 0.5 * (1.0 + math.cos(math.pi * t / T))
    # Guard against 1.0000000002 -> 2 from floating error at t = T.
    return int(math.ceil(round(v, 9)))


def budget_table(T: int, b_min: int, b_max: int) -> list[int]:
    return [edit_budget(t, T, b_min, b_max) for t in range(T)]
