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
"""LAB training split: evolve-120 + heldout-40 from the eligible pool.

Eligibility: has criteria & documents, criteria <= 100 (all-pass on 100+ is
a guaranteed 0 and a judge-cost bomb), doc bundle <= 20MB (context bombs).

Stratification: proportional per practice area; within an area tasks are
ordered by criteria count (proxy for deliverable complexity; no difficulty
prior exists for LAB) and picked by even stride, heldout first (so heldout
spans the complexity range), evolve from the remainder. Deterministic.

APEX-Agents (Law subset) is the pristine cross-benchmark test: it never enters
selection and is only touched by the final evaluation.

Output: data/split_workspace.json
"""
import json
import os
from collections import defaultdict

LAB = os.path.join(os.environ.get("HARVEY_LAB_ROOT", "harvey-labs"), "tasks")
HERE = os.path.dirname(os.path.abspath(__file__))
N_EVOLVE, N_HELDOUT = 120, 40
MAX_CRIT = 100
MAX_DOC_BYTES = 20 * 1024 * 1024


def stride_pick(items, n):
    if n >= len(items):
        return list(items), []
    idx = {round(i * (len(items) - 1) / max(1, n - 1)) for i in range(n)}
    while len(idx) < n:
        idx.add(min(i for i in range(len(items)) if i not in idx))
    picked = [items[i] for i in sorted(idx)]
    rest = [x for i, x in enumerate(items) if i not in idx]
    return picked, rest


def main():
    cands = defaultdict(list)
    for dirpath, _, filenames in os.walk(LAB):
        if "task.json" not in filenames or not os.path.isdir(os.path.join(dirpath, "documents")):
            continue
        rel = os.path.relpath(dirpath, LAB)
        try:
            tj = json.load(open(os.path.join(dirpath, "task.json")))
        except Exception:
            continue
        crit = tj.get("criteria") or []
        if not crit or len(crit) > MAX_CRIT:
            continue
        docs = os.path.join(dirpath, "documents")
        size = sum(os.path.getsize(os.path.join(r, f))
                   for r, _, fs in os.walk(docs) for f in fs)
        if size > MAX_DOC_BYTES:
            continue
        cands[rel.split("/")[0]].append((rel, len(crit)))

    total = sum(len(v) for v in cands.values())
    split = {"evolve": [], "heldout": []}
    report = []
    for area in sorted(cands):
        pool = sorted(cands[area], key=lambda x: (x[1], x[0]))
        ids = [t for t, _ in pool]
        q_ho = max(0, round(N_HELDOUT * len(pool) / total))
        q_ev = max(1, round(N_EVOLVE * len(pool) / total))
        ho, rest = stride_pick(ids, q_ho)
        ev, _ = stride_pick(rest, q_ev)
        split["heldout"] += ho
        split["evolve"] += ev
        report.append(f"{area}: ev {len(ev)} ho {len(ho)} / {len(pool)}")

    # trim/pad to exact sizes deterministically (drop from largest areas last)
    split["evolve"] = sorted(split["evolve"])[:N_EVOLVE]
    split["heldout"] = sorted(split["heldout"])[:N_HELDOUT]
    smoke = split["evolve"][:2]   # cheap pipe checks come from evolve
    out = {
        "mode": "workspace_train_v1",
        "tasks": split,
        "smoke": smoke,
        "meta": {"n": {k: len(v) for k, v in split.items()},
                 "note": "LAB evolve-120 fixed training set + LAB heldout-40 "
                         "(final only). APEX Law 160 = pristine cross-benchmark "
                         "test, never touched during evolution."},
    }
    allids = split["evolve"] + split["heldout"]
    assert len(allids) == len(set(allids)), "overlap between evolve and heldout"
    json.dump(out, open(os.path.join(HERE, "data", "split_workspace.json"), "w"), indent=1)
    print(json.dumps(out["meta"], indent=1))
    print("; ".join(report))


if __name__ == "__main__":
    main()
