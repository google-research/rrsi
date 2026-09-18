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
"""Unit tests for the method core: no API, no benchmark.

    python3 -m pytest -q tests/        or        python3 tests/test_core.py
"""

import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rrsi.calibrate import calibrate                       # noqa: E402
from rrsi.components import normalize, novelty              # noqa: E402
from rrsi.config import RRSIConfig                         # noqa: E402
from rrsi.evaluate import EvalResult, TaskResult, aggregate  # noqa: E402
from rrsi.history import History, exploration, stall_flag   # noqa: E402
from rrsi.schedule import budget_table, edit_budget         # noqa: E402
from rrsi.selection import Candidate, cost_rule, select_round  # noqa: E402


def _ev(job, rewards_by_task, k, tokens=1000.0):
    per = {t: TaskResult(rewards=list(r), tokens=[tokens] * len(r))
           for t, r in rewards_by_task.items()}
    return aggregate(job, k, per)


def test_schedule_matches_eq_anneal():
    T, bmin, bmax = 20, 1, 4
    for t in range(T):
        expect = math.ceil(bmin + (bmax - bmin) * 0.5 * (1 + math.cos(math.pi * t / T)))
        assert edit_budget(t, T, bmin, bmax) == expect
    tab = budget_table(T, bmin, bmax)
    assert tab[0] == bmax and tab == sorted(tab, reverse=True)
    assert edit_budget(T, T, bmin, bmax) == bmin


def test_estimator_and_weights():
    ev = _ev("j", {"a": [1, 0], "b": [1, 1]}, 2)
    assert abs(ev.S - 0.75) < 1e-9 and ev.n_expected == 4 and ev.missing == 0
    per = {"a": TaskResult(rewards=[0.5, 1.0], weights=[10, 10]),
           "b": TaskResult(rewards=[0.0, 0.0], weights=[90, 90])}
    ev = aggregate("w", 2, per)
    assert abs(ev.S - 15 / 200) < 1e-9          # criteria-weighted fraction


def test_calibration_bootstrap_and_repeats():
    rng_rewards = {f"t{i}": [i % 2, (i + 1) % 2] for i in range(40)}
    ev = _ev("base", rng_rewards, 2)
    cal = calibrate([ev], z=2.0, reps=300)
    assert cal["delta"] > 0 and cal["method"].startswith("bootstrap")
    ev2 = _ev("base2", {f"t{i}": [1, 1 if i % 3 else 0] for i in range(40)}, 2)
    cal2 = calibrate([ev, ev2], z=2.0, reps=100)
    assert cal2["n_evals"] == 2 and cal2["delta"] >= 0


def test_cost_rule_both_branches():
    cfg = RRSIConfig(beta0=0.1, beta1=40.0, w_s=100.0, w_c=15.0, w_n=0.5)
    ok, _ = cost_rule(0.05, 0.10 + 40 * 0.05 - 0.01, 0, 0.02, cfg)      # within budget
    assert ok
    ok, _ = cost_rule(0.05, 0.10 + 40 * 0.05 + 0.01, 0, 0.02, cfg)      # over budget
    assert not ok
    ok, _ = cost_rule(0.0, -0.10, 0, 0.02, cfg)                          # neutral, cheaper
    assert ok
    ok, _ = cost_rule(0.0, 0.10, 0, 0.02, cfg)                           # neutral, costlier
    assert not ok
    ok, _ = cost_rule(0.0, 0.0, 1, 0.02, cfg)                            # neutral, new structural
    assert ok


def test_selection_floor_argmax_and_sstar():
    cfg = RRSIConfig(beta0=0.1, beta1=40.0, w_s=100.0, w_c=15.0, w_n=0.5)
    inc = _ev("inc", {f"t{i}": [1, 1] if i < 5 else [0, 0] for i in range(10)}, 2)   # S=0.5
    a = Candidate("A", [{"id": "C1", "component": "prompt"}],
                  ev=_ev("A", {f"t{i}": [1, 1] if i < 7 else [0, 0] for i in range(10)}, 2))  # 0.7
    b = Candidate("B", [{"id": "C1", "component": "skill"}],
                  ev=_ev("B", {f"t{i}": [1, 1] if i < 6 else [0, 0] for i in range(10)}, 2))  # 0.6
    c = Candidate("C", [{"id": "C1", "component": "config"}],
                  ev=_ev("C", {f"t{i}": [1, 1] if i < 3 else [0, 0] for i in range(10)}, 2))  # 0.3 floor
    d = Candidate("D", [], gate_failure="critic_reject")
    S_star, delta = 0.55, 0.05                      # S* above the incumbent (earlier peak)
    win, decs = select_round([a, b, c, d], inc, S_star, delta, cfg, {}, guard_fn=None)
    assert win is a
    by = {x.variant: x for x in decs}
    assert by["A"].admissible and by["B"].admissible
    assert not by["C"].admissible and "floor" in by["C"].reason
    assert not by["D"].admissible and by["D"].reason == "critic_reject"
    # a gaining candidate that is too expensive is blocked by the L1 rule
    exp = Candidate("E", [{"id": "C1", "component": "prompt"}],
                    ev=_ev("E", {f"t{i}": [1, 1] if i < 7 else [0, 0] for i in range(10)}, 2,
                           tokens=1000.0 * (1 + 0.1 + 40 * 0.2 + 0.5)))
    win2, decs2 = select_round([exp], inc, S_star, delta, cfg, {})
    assert win2 is None and "cost rule" in decs2[0].reason
    # a domain guard is non-compensatory
    win3, decs3 = select_round([a], inc, S_star, delta, cfg, {},
                               guard_fn=lambda i, c: ["valid rate fell"])
    assert win3 is None and "guard" in decs3[0].reason


def test_history_summaries_prune_and_explore():
    with tempfile.TemporaryDirectory() as td:
        h = History(Path(td) / "history.jsonl")
        h.append_candidate(0, "A", [{"id": "C1", "component": "prompt", "hypothesis": "h1"}],
                           "ACCEPTED", 0.03, 0.05, True, 0.53, 1000, None)
        h.append_candidate(1, "A", [{"id": "C1", "component": "prompt", "hypothesis": "h2"}],
                           "REJECTED", -0.01, 0.0, False, 0.52, 1000, None)
        h.append_candidate(1, "B", [{"id": "C1", "component": "skill", "hypothesis": "h3"},
                                    {"id": "C2", "component": "memory", "hypothesis": "h4"}],
                           "REJECTED", -0.02, 0.3, False, 0.51, 1300, None)
        h.append_candidate(2, "A", [{"id": "C1", "component": "config", "hypothesis": "h5"}],
                           "critic_reject", None, None, False, None, None, None, "leak")
        assert h.tried() == {"prompt", "skill", "memory"}       # critic-rejected config not tried
        g = h.yield_g(t=3, n_prune=4)
        assert g["prompt"] == 0.03 and g["skill"] == -0.02
        # window: at t=5 with n_prune=4 the accepted prompt edit (t=0) falls out
        g5 = h.yield_g(t=5, n_prune=4)
        assert g5["prompt"] == -0.01
        assert h.yield_g(t=6, n_prune=4)["prompt"] == -math.inf    # nothing recent at all
        prune = {p["component"]: p for p in h.prune_set(t=5, n_prune=4)}
        assert set(prune) == {"prompt", "skill", "memory"}
        assert prune["prompt"]["accepted_edits_in_incumbent"][0]["hypothesis"] == "h1"
        assert h.incumbent_component_counts()["prompt"] == 1
        assert novelty(["client_tool", "prompt"], h.incumbent_component_counts()) == 1
        assert novelty(["skill"], {"skill": 2}) == 0
        assert h.has(1, "B") and not h.has(5, "A")
    traj = [0.50, 0.53, 0.53, 0.535, 0.60]
    assert stall_flag(traj, 3, 3, 0.02) == 0       # 0.535 - 0.50 = 0.035 > 0.02
    assert stall_flag(traj, 3, 2, 0.02) == 1       # 0.535 - 0.53 = 0.005 <= 0.02
    assert stall_flag(traj, 4, 2, 0.02) == 0       # 0.60 - 0.53 = 0.07
    assert stall_flag(traj, 1, 3, 0.02) == 0       # not enough history
    e = exploration(3, 1, {"prompt"}, 1)
    assert e["sigma"] == 1 and "prompt" not in e["untried"] and "RESERVED" in e["text"]


def test_component_normalization():
    assert normalize("Skill", "", []) == "prompt"          # no evidence in an empty diff
    assert normalize("Skill", "+++ b/x/skills/y/SKILL.md", []) == "skill"
    assert normalize("bogus", "+++ b/harness/skills/x/SKILL.md", []) == "skill"
    assert normalize(None, "+++ b/harness/resum.py\n+ keep_last = 5",
                     [("context_mgmt", [r"resum\.py"])]) == "context_mgmt"
    assert normalize(None, "nothing structural", []) == "prompt"


def test_config_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "rrsi.json"
        p.write_text(json.dumps({"T": 7, "k": 3, "beta1": 12.5, "custom": "x"}))
        cfg = RRSIConfig.load(p, T=None, k=5)
        assert cfg.T == 7 and cfg.k == 5 and cfg.beta1 == 12.5 and cfg.notes["custom"] == "x"
        ev = EvalResult.from_json(aggregate("j", 1, {"a": TaskResult(rewards=[1.0])}).to_json())
        assert ev.S == 1.0


if __name__ == "__main__":
    import inspect
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and inspect.isfunction(fn):
            try:
                fn()
                print(f"ok   {name}")
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f"FAIL {name}: {e!r}")
    sys.exit(1 if fails else 0)
