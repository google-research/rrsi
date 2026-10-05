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
"""Regressions for failed candidates retaining cached evaluation results."""

from dataclasses import replace
from unittest import mock

import pytest
from rrsi.config import RRSIConfig
from rrsi.evaluate import EvalResult, TaskResult, aggregate
from rrsi.selection import Candidate, judge, select_round


def evaluation(score, job="test"):
    return aggregate(job, 1, {"task": TaskResult([score], tokens=[100])})


@pytest.mark.parametrize(
    "failure", ["critic_reject", "smoke_fail", "eval_invalid", "no_proposal"]
)
@pytest.mark.parametrize("cached", [False, True])
def test_gate_failure_is_authoritative_even_with_cached_metrics(failure, cached):
    incumbent = evaluation(0.5)
    candidate = Candidate(
        "failed", [], ev=evaluation(0.9) if cached else None, gate_failure=failure
    )
    result = judge(candidate, incumbent, 0.5, 0.02, RRSIConfig(), {})
    assert not result.admissible
    assert result.reason == failure


@pytest.mark.parametrize("failed_first", [False, True])
def test_failed_high_score_cannot_displace_a_valid_candidate(failed_first):
    failed = Candidate("failed", [], ev=evaluation(0.99), gate_failure="eval_invalid")
    valid = Candidate("valid", [], ev=evaluation(0.7))
    candidates = [failed, valid] if failed_first else [valid, failed]
    winner, decisions = select_round(
        candidates, evaluation(0.5), 0.5, 0.02, RRSIConfig(), {}
    )
    assert winner is valid
    assert [decision.variant for decision in decisions] == [
        candidate.variant for candidate in candidates
    ]
    assert [decision.admissible for decision in decisions] == [
        candidate is valid for candidate in candidates
    ]


def test_domain_guard_does_not_run_on_rejected_evaluations():
    failed_eval = evaluation(0.99)
    valid_eval = evaluation(0.7)
    failed = Candidate("failed", [], ev=failed_eval, gate_failure="smoke_fail")
    valid = Candidate("valid", [], ev=valid_eval)

    def guard(incumbent, candidate):
        assert candidate is not failed_eval
        return []

    guard_spy = mock.Mock(side_effect=guard)
    incumbent = evaluation(0.5)
    winner, _ = select_round(
        [failed, valid], incumbent, 0.5, 0.02, RRSIConfig(), {}, guard_fn=guard_spy
    )
    assert winner is valid
    guard_spy.assert_called_once_with(incumbent, valid_eval)


def test_failed_candidates_stay_rejected_after_evaluation_roundtrip(tmp_path):
    path = tmp_path / "eval.json"
    evaluation(0.99).save(path)
    candidate = Candidate(
        "restored", [], ev=EvalResult.load(path), gate_failure="critic_reject"
    )
    winner, decisions = select_round(
        [candidate], evaluation(0.5), 0.5, 0.02, RRSIConfig(), {}
    )
    assert winner is None
    assert decisions[0].reason == "critic_reject"


def test_clear_gate_allows_normal_scoring_and_domain_guards():
    candidate = Candidate("valid", [], ev=evaluation(0.8))
    incumbent = evaluation(0.5)
    assert judge(candidate, incumbent, 0.5, 0.02, RRSIConfig(), {}).admissible
    assert not judge(
        candidate, incumbent, 0.5, 0.02, RRSIConfig(), {}, ["regression"]
    ).admissible
    not_evaluated = replace(candidate, ev=None)
    assert (
        judge(not_evaluated, incumbent, 0.5, 0.02, RRSIConfig(), {}).reason
        == "not evaluated"
    )
