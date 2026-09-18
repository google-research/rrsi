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
"""RRSI command line.

  python3 rrsi.py --domain eng baseline                # evaluate H_0, seed the frontier
  python3 rrsi.py --domain eng calibrate [--jobs base,base2]   # noise band delta
  python3 rrsi.py --domain eng round --t 3 [--dry-run] # one round (Alg. 1 + Alg. 2)
  python3 rrsi.py --domain eng run [--start 0]         # driver: rounds until T
  python3 rrsi.py --domain eng readjudicate --t 3      # re-apply Alg. 2 to round 3's stored measurements
  python3 rrsi.py --domain eng reevaluate --t 3        # re-measure round 3's candidates (infra failure), then re-adjudicate
  python3 rrsi.py --domain workspace heldout --label champ [--ref <commit>] [--set heldout]
  python3 rrsi.py --domain eng smoke                   # liveness check of the incumbent
  python3 rrsi.py --domain eng status                  # frontier + history summary

Hyperparameters come from domains/<domain>/rrsi.json; any of them can be
overridden on the command line (--T, --k, --m, --b-min, --b-max, --w,
--m-draft, --delta, --beta0, --beta1, --w-s, --w-c, --w-n, --n-prune).
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from rrsi import gitops as G                 # noqa: E402
from rrsi.config import RRSIConfig          # noqa: E402
from rrsi.domain import load_domain          # noqa: E402
from rrsi.driver import drive                # noqa: E402
from rrsi.loop import Run                    # noqa: E402
from rrsi.schedule import budget_table       # noqa: E402

OVERRIDES = ["T", "k", "m", "b_min", "b_max", "w", "m_draft", "delta", "delta_z",
             "beta0", "beta1", "w_s", "w_c", "w_n", "n_prune", "eval_parallel"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--runs", default=str(ROOT / "runs"))
    for name in OVERRIDES:
        typ = int if name in ("T", "k", "m", "b_min", "b_max", "w", "m_draft",
                              "n_prune", "eval_parallel") else float
        ap.add_argument("--" + name.replace("_", "-"), dest=name, type=typ, default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("baseline").add_argument("--job", default="base")
    sub.add_parser("calibrate").add_argument("--jobs", default="base")
    p = sub.add_parser("round")
    p.add_argument("--t", type=int, required=True)
    p.add_argument("--dry-run", action="store_true")
    sub.add_parser("run").add_argument("--start", type=int, default=0)
    sub.add_parser("readjudicate", help="re-run Algorithm 2 on a stored round after a "
                   "delta / weight change (no new evaluation)").add_argument("--t", type=int, required=True)
    p = sub.add_parser("reevaluate", help="re-measure a round's committed candidates after an "
                       "infrastructure failure, then re-adjudicate")
    p.add_argument("--t", type=int, required=True)
    p.add_argument("--variants", default="", help="e.g. A,B (default: all)")
    p = sub.add_parser("heldout")
    p.add_argument("--label", required=True)
    p.add_argument("--ref", default=None, help="commit/branch to evaluate (default incumbent)")
    p.add_argument("--set", default="heldout", choices=["heldout", "evolve"])
    sub.add_parser("smoke")
    sub.add_parser("status")
    args = ap.parse_args()

    domain = load_domain(args.domain)
    cfg = RRSIConfig.load(domain.root / "rrsi.json",
                           **{k: getattr(args, k) for k in OVERRIDES})
    run = Run(domain, cfg, ROOT, Path(args.runs))

    if args.cmd == "baseline":
        run.baseline(args.job)
        run.calibrate([args.job])
    elif args.cmd == "calibrate":
        run.calibrate([j.strip() for j in args.jobs.split(",") if j.strip()])
    elif args.cmd == "round":
        run.round(args.t, dry_run=args.dry_run)
    elif args.cmd == "readjudicate":
        run.readjudicate(args.t)
    elif args.cmd == "reevaluate":
        run.reevaluate(args.t, [v for v in args.variants.split(",") if v] or None)
    elif args.cmd == "run":
        drive(Path(__file__), args.domain, Path(args.runs), cfg.T, args.start,
              extra_args=[a for k in OVERRIDES if getattr(args, k) is not None
                          for a in ("--" + k.replace("_", "-"), str(getattr(args, k)))])
    elif args.cmd == "heldout":
        ids = domain.heldout_ids() if args.set == "heldout" else domain.evolve_ids()
        if not ids:
            sys.exit(f"domain {domain.name} has no {args.set} split")
        run.heldout(args.label, ids, ref=args.ref)
    elif args.cmd == "smoke":
        run.ensure_branch()
        wt = run.checkout("smoke", run.branch)
        ok, detail = domain.smoke(wt, run.runs, "smoke", domain.smoke_ids())
        print(json.dumps({"ok": ok, **detail}, indent=1))
        sys.exit(0 if ok else 1)
    elif args.cmd == "status":
        fr = run.frontier()
        print(json.dumps({k: v for k, v in fr.items() if k != "config"}, indent=1))
        print("b_t schedule:", budget_table(cfg.T, cfg.b_min, cfg.b_max))
        print(f"delta = {run.delta():.5f}")
        for r in run.history.render(60):
            print(json.dumps(r, ensure_ascii=False))
        print("evolve branch:", G.rev(ROOT, run.branch), "tree", run.harness_tree(run.branch))


if __name__ == "__main__":
    main()
