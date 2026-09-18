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
"""The edit history L_t and the three summaries the paper computes from it.

    L_t = {(t_i, l_i, h_i, d_i, Delta S_i, Delta C_i, a_i) : i <= n_t}    Eq. (history)
    T_t = {l_i : i <= n_t}                                                 (tried components)
    g_t(l) = max{Delta S_i : l_i = l, t - t_i <= n_prune},  max(empty) = -inf   Eq. (yield)
    U_t = K \\ T_t,  sigma_t = 1[S_t - S_{t-w} <= delta]                    Eq. (explore)
    B_t = {l in T_t : g_t(l) <= 0}                                          Eq. (prune)

One JSONL record per EDIT. A candidate harness that bundles n edits (n <= b_t)
receives one measurement (Delta S, Delta C, a) and every edit in the bundle
carries it; as b_t anneals to 1 the record becomes evidence about a single
component. Candidates dropped before measurement (critic reject, smoke fail,
invalid evaluation) are recorded with delta_S = None and do not enter T_t or g_t.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from .components import K

MEASURED_OUTCOMES = ("ACCEPTED", "REJECTED", "LOST")


class History:
    def __init__(self, path: Path | str):
        self.path = Path(path)

    # ------------------------------------------------------------ storage --
    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    def append(self, rec: dict) -> None:
        rec = dict(rec)
        rec.setdefault("ts", time.strftime("%Y-%m-%d %H:%M:%S"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def append_candidate(self, t: int, variant: str, edits: list[dict],
                         outcome: str, delta_S: float | None,
                         delta_C: float | None, accepted: bool,
                         S: float | None, C: float | None,
                         diff: str | None, detail: str = "") -> None:
        """Write the per-edit records of one candidate harness H'."""
        for e in edits or [{"id": "C1", "component": None, "hypothesis": None}]:
            self.append({
                "t": t, "variant": variant, "edit_id": e.get("id"),
                "component": e.get("component"),
                "hypothesis": e.get("hypothesis") or e.get("mechanism"),
                "targets_mode": e.get("targets_mode"),
                "predicted_affected": e.get("predicted_affected"),
                "diff": diff,
                "delta_S": None if delta_S is None else round(delta_S, 6),
                "delta_C": None if delta_C is None else round(delta_C, 6),
                "accepted": bool(accepted),
                "outcome": outcome,
                "S": None if S is None else round(S, 6),
                "C": None if C is None else round(C, 1),
                "bundle": len(edits or []),
                "detail": detail[:600] if detail else "",
            })

    def replace_round(self, t: int, keep=lambda r: False) -> None:
        """Drop the per-edit records of round t (a re-adjudication rewrites
        them); records satisfying `keep` survive."""
        recs = [r for r in self.records()
                if not (r.get("t") == t and r.get("edit_id")) or keep(r)]
        self.path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))

    def has(self, t: int, variant: str) -> bool:
        return any(r.get("t") == t and r.get("variant") == variant
                   for r in self.records())

    # ---------------------------------------------------------- summaries --
    def measured(self) -> list[dict]:
        return [r for r in self.records() if r.get("delta_S") is not None
                and r.get("component") in K]

    def tried(self) -> set[str]:
        """T_t: components with at least one measured edit."""
        return {r["component"] for r in self.measured()}

    def attempted(self) -> dict[str, int]:
        """Edits per component that were shipped at all, measured or not."""
        counts = {c: 0 for c in K}
        for r in self.records():
            if r.get("component") in counts and r.get("edit_id"):
                counts[r["component"]] += 1
        return counts

    def accepted_edits(self) -> dict[str, list[dict]]:
        """Machinery currently in the incumbent, per component (accepted edits)."""
        out: dict[str, list[dict]] = {c: [] for c in K}
        for r in self.records():
            if r.get("accepted") and r.get("component") in out:
                out[r["component"]].append(
                    {"t": r["t"], "edit_id": r.get("edit_id"),
                     "hypothesis": r.get("hypothesis"),
                     "delta_S": r.get("delta_S")})
        return out

    def incumbent_component_counts(self) -> dict[str, int]:
        return {c: len(v) for c, v in self.accepted_edits().items()}

    def yield_g(self, t: int, n_prune: int) -> dict[str, float]:
        """g_t(l) for every tried component; -inf when nothing recent."""
        g = {c: -math.inf for c in self.tried()}
        for r in self.measured():
            if t - int(r["t"]) <= n_prune:
                c = r["component"]
                g[c] = max(g.get(c, -math.inf), float(r["delta_S"]))
        return g

    def prune_set(self, t: int, n_prune: int) -> list[dict]:
        """B_t with, for each component, the accepted machinery to remove."""
        g = self.yield_g(t, n_prune)
        acc = self.accepted_edits()
        out = []
        for c in sorted(g):
            if g[c] <= 0:
                out.append({"component": c,
                            "recent_best_gain": (None if g[c] == -math.inf
                                                 else round(g[c], 5)),
                            "accepted_edits_in_incumbent": acc.get(c, [])})
        return out

    # ------------------------------------------------------- for prompts --
    def render(self, n: int = 40) -> list[dict]:
        """Compact view of the most recent records for the proposer's context.
        Measured outcomes dominate: gate failures without a measurement are
        kept only for their most recent few, because a wall of aborts is a
        feedback loop, not evidence."""
        recs = self.records()
        kept, unmeasured = [], 0
        for r in reversed(recs):
            if r.get("delta_S") is None:
                unmeasured += 1
                if unmeasured > 4:
                    continue
            kept.append(r)
            if len(kept) >= n:
                break
        keys = ("t", "variant", "edit_id", "component", "hypothesis",
                "targets_mode", "delta_S", "delta_C", "accepted", "outcome",
                "bundle", "detail")
        return [{k: r.get(k) for k in keys if r.get(k) is not None}
                for r in reversed(kept)]


# --------------------------------------------------------------- explore --
def stall_flag(trajectory: list[float], t: int, w: int, delta: float) -> int:
    """sigma_t = 1[S_t - S_{t-w} <= delta]; 0 while fewer than w rounds exist."""
    if t < w or t >= len(trajectory) or t - w < 0:
        return 0
    return int(trajectory[t] - trajectory[t - w] <= delta)


def exploration(t: int, stall: int, tried: set[str], m_draft: int) -> dict:
    """E_t = (sigma_t, U_t, m_draft) plus the text handed to the proposer."""
    untried = [c for c in K if c not in tried]
    if stall and untried:
        text = (f"STALL: the incumbent has not moved by more than the noise band "
                f"over the last rounds (sigma_t = 1). {m_draft} candidate slot(s) "
                f"this round are RESERVED for exploratory edits on components the "
                f"run has never exercised: {untried}. A variant holding a reserved "
                f"slot must put at least one edit on one of those components.")
    elif untried:
        text = (f"Components not yet exercised in this run: {untried}. Not "
                f"mandatory this round (sigma_t = 0), but evidence about them is "
                f"still missing.")
    else:
        text = "Every component in K has been exercised at least once."
    return {"sigma": stall, "untried": untried, "m_draft": m_draft, "text": text}
