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
from rrsi.components import K_STR, classify_diff, normalize, novelty, text_only
from rrsi.config import RRSIConfig
from rrsi.evaluate import TaskResult, aggregate
from rrsi.selection import Candidate, select_round
import pytest

SIGNALS = {
    "skill": "SkillRegistry",
    "memory": "memory.remember()",
    "client_tool": "ToolRegistry",
    "subagent": "subcall()",
}


def prompt_diff(text, comment=False):
    line = "# " + text if comment else repr(text)
    return (
        "diff --git a/prompts.py b/prompts.py\n"
        "--- a/prompts.py\n+++ b/prompts.py\n@@ -1 +1 @@\n"
        "-# Old instructions\n+" + line + "\n"
    )


@pytest.mark.parametrize("component", K_STR)
@pytest.mark.parametrize("comment", [False, True])
def test_declared_structural_tags_do_not_override_text_only_changes(component, comment):
    diff = prompt_diff(SIGNALS[component], comment)
    assert text_only(diff)
    assert classify_diff(diff) == "prompt"
    assert normalize(component, diff) == "prompt"
    assert normalize("  " + component.upper() + "  ", diff) == "prompt"
    assert novelty([normalize(component, diff)], {}) == 0


def test_domain_signals_do_not_override_text_only_changes():
    diff = prompt_diff("ToolRegistry behavior")
    signals = [("config", [r"prompts\.py"])]
    assert normalize("config", diff, signals) == "prompt"


def test_executable_structural_changes_keep_their_tags():
    for component, signal in SIGNALS.items():
        diff = "--- a/tools.py\n+++ b/tools.py\n@@ -0,0 +1 @@\n+instance = " + signal
        assert not text_only(diff)
        assert normalize(component, diff) == component
    assert normalize("skill", "+++ b/skills/example/SKILL.md") == "skill"


def test_text_only_candidate_receives_no_structural_acceptance_bonus():
    evaluation = aggregate("same-score", 1, {"task": TaskResult([0.5], tokens=[100])})
    diff = prompt_diff("Consult the SkillRegistry before answering")
    component = normalize("skill", diff)
    candidate = Candidate("text", [{"component": component}], ev=evaluation)
    winner, decisions = select_round(
        [candidate], evaluation, 0.5, 0.02, RRSIConfig(), {}
    )
    assert winner is None
    assert not decisions[0].admissible
    assert decisions[0].novelty == 0
    assert "cost rule failed" in decisions[0].reason
