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
"""Summarise SWE-bench Verified arms run through harbor: resolve rate over the
full 500-instance denominator (a missing or errored trial counts as unresolved),
infrastructure failures, token cost, and the paired per-instance comparison.

  python3 domains/coding/scripts/swe_summary.py runs/coding/jobs/swe_base runs/coding/jobs/swe_best
"""
import json, sys, glob, collections
from pathlib import Path

N_TOTAL = 500


def load(job_dir):
    out = {}
    for f in glob.glob(str(Path(job_dir) / "*__*" / "result.json")):
        r = json.load(open(f))
        name = str(r.get("task_name", "")).split("/")[-1]
        rew = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        e = r.get("exception_info") or {}
        ar = r.get("agent_result") or {}
        out[name] = {"resolved": rew == 1.0,
                     "exc": (e.get("exception_type") if isinstance(e, dict) else str(e)) if e else None,
                     "tokens": (ar.get("n_input_tokens") or 0) + (ar.get("n_output_tokens") or 0)}
    return out


def report(label, d):
    n = len(d); res = sum(1 for v in d.values() if v["resolved"])
    exc = collections.Counter(v["exc"] for v in d.values() if v["exc"])
    toks = [v["tokens"] for v in d.values() if v["tokens"]]
    print(f"{label:9s} trials={n}/{N_TOTAL} resolved={res}  rate={res / N_TOTAL:.4f} "
          f"(over {N_TOTAL}; {res / max(1, n):.4f} over ran)  exceptions={dict(exc)}  "
          f"tokens/trial={sum(toks) / max(1, len(toks)) / 1000:.0f}k")
    return res


if __name__ == "__main__":
    arms = {Path(p).name: load(p) for p in sys.argv[1:]}
    for k, d in arms.items():
        report(k, d)
    if len(arms) == 2:
        (a, da), (b, db) = arms.items()
        common = set(da) & set(db)
        up = sum(1 for t in common if db[t]["resolved"] and not da[t]["resolved"])
        down = sum(1 for t in common if da[t]["resolved"] and not db[t]["resolved"])
        print(f"paired on {len(common)} common instances: {b} gains {up}, loses {down} (net {up - down:+d})")
