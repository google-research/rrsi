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
"""What the score cannot see.

The accept rule reads three numbers. This module reads the behaviour behind
them, so an edit-history entry can distinguish a mechanism that never fired from one
that fired hard and still did nothing. The first invites a better implementation
of the same idea; the second rules out the whole family. Without this
distinction the proposer re-ships the same hypothesis in a new costume every
few rounds and burns the budget.

The axes here are the ones a scaffold can actually move on this benchmark:
how much the agent COMPUTES (code_exec volume and error rate), whether it ever
checks its own submission before finalising, how often it ends with nothing
submitted, and the shape of its losses (malformed / invalid / under target).

Nothing here participates in grading.
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

JOBS = Path(config.RUNS) / "jobs"

_MALFORMED = re.compile(r"validation|literal|PAYLOAD|schema|pydantic|field",
                        re.I)


def _tool_calls(traj: dict):
    for m in (traj or {}).get("messages") or []:
        if m.get("role") != "assistant":
            continue
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            yield fn.get("name") or "", (fn.get("arguments") or "")


def trial_stats(traj: dict, meta: dict | None = None) -> dict:
    meta = meta or {}
    msgs = (traj or {}).get("messages") or []
    names = []
    n_payload_writes = 0
    n_reads_payload = 0
    for name, args in _tool_calls(traj):
        names.append(name)
        if name in ("write_file", "code_exec") and config.CANDIDATE_REL in args:
            n_payload_writes += 1
        if name in ("read_text_file", "code_exec") and "payload.py" in args:
            n_reads_payload += 1
    # A failed code_exec is visible only in the tool RESULT, not the call.
    n_failed = 0
    for m in msgs:
        if m.get("role") == "tool" and m.get("name") == "code_exec":
            c = m.get("content")
            if isinstance(c, str) and ("'success': False" in c
                                       or '"success": false' in c.lower()
                                       or "Traceback" in c):
                n_failed += 1
    counts: dict[str, int] = {}
    for n in names:
        counts[n] = counts.get(n, 0) + 1
    return {
        "n_steps": sum(1 for m in msgs if m.get("role") == "assistant"),
        "n_tool_calls": len(names),
        "n_code_exec": counts.get("code_exec", 0),
        "n_code_exec_failed": n_failed,
        "n_write_file": counts.get("write_file", 0),
        "n_read": counts.get("read_text_file", 0) + counts.get("read_pdf", 0),
        "n_payload_writes": n_payload_writes,
        "n_payload_readbacks": n_reads_payload,
        "n_todo_write": counts.get("todo_write", 0),
        "tool_calls": counts,
        "no_payload": bool(meta.get("no_payload")),
        "total_tokens": meta.get("total_tokens"),
        "wall_s": meta.get("wall_s"),
        "status": meta.get("status"),
    }


def analyse(job: str, ids: list[str] | None = None) -> dict:
    """Aggregate behaviour over a whole job."""
    jdir = JOBS / job
    rows = []
    for tdir in sorted(jdir.glob("*/t*")):
        if ids is not None and tdir.parent.name not in ids:
            continue
        mp, vp, tp = tdir / "meta.json", tdir / "verdict.json", tdir / "traj.json"
        if not mp.is_file():
            continue
        meta = json.loads(mp.read_text())
        verdict = json.loads(vp.read_text()) if vp.is_file() else {}
        traj = json.loads(tp.read_text()) if tp.is_file() else {}
        s = trial_stats(traj, meta)
        err = str(verdict.get("error") or "")
        s["passed"] = bool(verdict.get("passed"))
        s["valid"] = float(verdict.get("valid") or 0.0)
        s["score"] = float(verdict.get("combined_score") or 0.0)
        s["malformed"] = bool(err) and bool(_MALFORMED.search(err))
        rows.append(s)
    if not rows:
        return {"job": job, "n_trials": 0}

    def mean(key, default=0.0):
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else default

    def rate(pred):
        return round(sum(1 for r in rows if pred(r)) / len(rows), 4)

    tool_totals: dict[str, float] = {}
    for r in rows:
        for k, v in (r.get("tool_calls") or {}).items():
            tool_totals[k] = tool_totals.get(k, 0) + v
    n = len(rows)
    return {
        "job": job,
        "n_trials": n,
        "mean_steps": mean("n_steps"),
        "mean_code_exec": mean("n_code_exec"),
        "code_exec_error_rate": (round(sum(r["n_code_exec_failed"] for r in rows)
                                       / max(1, sum(r["n_code_exec"] for r in rows)), 4)),
        "mean_payload_writes": mean("n_payload_writes"),
        "payload_readback_rate": rate(lambda r: r["n_payload_readbacks"] > 0),
        "no_payload_rate": rate(lambda r: r["no_payload"]),
        # The loss decomposition: these three partition every non-pass and want
        # opposite mechanisms (see evolve/trace_render.render_grading).
        "malformed_rate": rate(lambda r: r["malformed"] and not r["no_payload"]),
        "invalid_rate": rate(lambda r: (not r["malformed"]) and (not r["no_payload"])
                             and r["valid"] < 1.0),
        "under_target_rate": rate(lambda r: r["valid"] >= 1.0 and not r["passed"]),
        "pass_rate": rate(lambda r: r["passed"]),
        "mean_score": mean("score"),
        "mean_tokens": mean("total_tokens", None),
        "mean_wall_s": mean("wall_s", None),
        "timeout_rate": rate(lambda r: r.get("status") == "timeout"),
        "tool_calls_per_run": {k: round(v / n, 3) for k, v in
                               sorted(tool_totals.items(), key=lambda kv: -kv[1])},
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    a = ap.parse_args()
    print(json.dumps(analyse(a.job), indent=1))
