#!/usr/bin/env python3
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
"""Evolve / held-out split of Harvey LAB: 120 evolve tasks + 40 held-out tasks.

The split is not distributed with this repository. It is generated from a
Harvey LAB checkout (https://github.com/harveyai/harvey-labs) the first time
the workspace domain needs it and written to data/split_workspace.json. The
generator is deterministic, so one checkout always yields one split; the
checkout the paper's split was generated from is pinned in rrsi.json as
`harvey_lab_commit`, and a checkout at a different commit is reported as a
warning because later upstream edits to a task's criteria or documents can
move it across the eligibility or stratification boundaries.

Eligibility: the task has criteria and documents, at most 100 criteria (an
all-pass score on 100+ criteria is a guaranteed zero and a judge-cost bomb)
and a document bundle of at most 20MB.

Stratification: proportional per practice area; within an area tasks are
ordered by criteria count (a proxy for deliverable complexity) and picked by
even stride, held-out first (so the held-out set spans the complexity range),
evolve from the remainder.

Usage: HARVEY_LAB_ROOT=/path/to/harvey-labs python3 split_workspace.py
"""
import json
import os
import subprocess
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "split_workspace.json")
N_EVOLVE, N_HELDOUT = 120, 40
MAX_CRIT = 100
MAX_DOC_BYTES = 20 * 1024 * 1024


def _cfg() -> dict:
    return json.load(open(os.path.join(HERE, "rrsi.json")))


def lab_root() -> str:
    return os.environ.get("HARVEY_LAB_ROOT", _cfg().get("harvey_lab_root") or "harvey-labs")


def checkout_commit(root: str) -> str:
    try:
        return subprocess.run(["git", "-C", root, "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return ""


def stride_pick(items, n):
    if n >= len(items):
        return list(items), []
    idx = {round(i * (len(items) - 1) / max(1, n - 1)) for i in range(n)}
    while len(idx) < n:
        idx.add(min(i for i in range(len(items)) if i not in idx))
    picked = [items[i] for i in sorted(idx)]
    rest = [x for i, x in enumerate(items) if i not in idx]
    return picked, rest


def build_split(root: str) -> tuple:
    """Return (split dict, per-area report) for the Harvey LAB checkout at root."""
    tasks_dir = os.path.join(root, "tasks")
    cands = defaultdict(list)
    for dirpath, _, filenames in os.walk(tasks_dir):
        if "task.json" not in filenames or not os.path.isdir(os.path.join(dirpath, "documents")):
            continue
        rel = os.path.relpath(dirpath, tasks_dir)
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
    if not total:
        raise SystemExit(f"no eligible Harvey LAB tasks under {tasks_dir}")
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

    # trim to exact sizes deterministically
    split["evolve"] = sorted(split["evolve"])[:N_EVOLVE]
    split["heldout"] = sorted(split["heldout"])[:N_HELDOUT]
    allids = split["evolve"] + split["heldout"]
    assert len(allids) == len(set(allids)), "overlap between evolve and heldout"
    out = {
        "mode": "workspace_train_v1",
        "tasks": split,
        "smoke": split["evolve"][:2],   # cheap pipe checks come from the evolve set
        "meta": {"n": {k: len(v) for k, v in split.items()},
                 "harvey_lab_commit": checkout_commit(root)},
    }
    return out, report


def write_split(path: str = OUT, root=None) -> dict:
    """Generate the split from the checkout at root (default: HARVEY_LAB_ROOT) and write it."""
    root = root or lab_root()
    if not os.path.isdir(os.path.join(root, "tasks")):
        raise SystemExit(f"no Harvey LAB checkout at {root} (set HARVEY_LAB_ROOT)")
    pin, head = _cfg().get("harvey_lab_commit") or "", checkout_commit(root)
    if pin and head and head != pin:
        print(f"[split_workspace] warning: checkout is at {head[:9]}; the split reported in "
              f"the paper was generated at {pin[:9]}", file=sys.stderr)
    out, report = build_split(root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out["meta"], indent=1))
    print("; ".join(report))
    return out


if __name__ == "__main__":
    write_split()
