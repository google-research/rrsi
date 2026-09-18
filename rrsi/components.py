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
"""The editable component set K and its structural subset K_str.

The paper tags every candidate edit with the harness component l' it modifies
(Sec. evidence-aware credit assignment). K is the fixed vocabulary of those
tags; K_str are the components that add machinery (a tool, a skill file, a
memory store, an extra policy call) rather than changing text or constants.

A proposer declares the component of each edit; the declaration is validated
against K and, when it is missing or invalid, recovered from the diff using the
domain's file-name signals so a mislabelled prompt tweak cannot pass as a skill.
"""

from __future__ import annotations

import re

K = ["prompt", "control_flow", "config", "output_plumbing", "context_mgmt",
     "client_tool", "skill", "memory", "subagent"]
K_STR = ["client_tool", "skill", "memory", "subagent"]

# Signals shared by every domain (structural substrate names in mechanisms.py).
GENERIC_SIGNALS = [
    ("memory",      [r"\bMemory\(", r"\.remember\(", r"\.recall\(", r"_STATE_DIR"]),
    ("skill",       [r"skills/", r"SkillRegistry", r"skill_use", r"skill_catalog",
                     r"upload_skills"]),
    ("client_tool", [r"ToolRegistry", r"register_tool", r"tool_spec", r"CLIENT_TOOLS"]),
    ("subagent",    [r"\bsubcall\(", r"sub_agent", r"subagent"]),
]


_STRING_LINE = re.compile(r'^[+-]\s*(?:[frb]?["\']|""")')


def text_only(diff: str) -> bool:
    """True when every changed line is a string literal or a comment: a
    model-facing text edit, i.e. `prompt`, whatever words the prose contains."""
    changed = [l for l in diff.splitlines()
               if (l.startswith("+") or l.startswith("-"))
               and not l.startswith("+++") and not l.startswith("---") and l[1:].strip()]
    if not changed:
        return False
    return all(_STRING_LINE.match(l) or l[1:].lstrip().startswith("#") for l in changed)


def classify_diff(diff: str, domain_signals: list | None = None) -> str:
    """First matching component in (domain signals, then generic); default prompt."""
    if text_only(diff):
        return "prompt"
    for component, pats in (domain_signals or []) + GENERIC_SIGNALS:
        if any(re.search(p, diff) for p in pats):
            return component
    return "prompt"


def has_evidence(component: str, diff: str, domain_signals: list | None = None) -> bool:
    """Does the diff contain any signal of `component`? A declared tag is kept
    only when the diff carries evidence for it: a proposer can name a
    skill/memory/tool/subagent edit without shipping one, or relabel a
    control-flow change as an untried component to satisfy a reserved
    exploration slot, and an unverified tag corrupts T_t, U_t and the novelty
    term."""
    for comp, pats in (domain_signals or []) + GENERIC_SIGNALS:
        if comp == component and any(re.search(p, diff) for p in pats):
            return True
    return False


def normalize(declared: str | None, diff: str,
              domain_signals: list | None = None) -> str:
    d = (declared or "").strip().lower()
    if d in K and has_evidence(d, diff, domain_signals):
        return d
    return classify_diff(diff, domain_signals)


def novelty(edit_components: list[str], incumbent_counts: dict) -> int:
    """nu(l'): number of STRUCTURAL components the candidate touches that the
    incumbent has never had an accepted edit on. Only ever tie-breaks a
    candidate inside the noise band (Alg. 2 line 5)."""
    return sum(1 for c in set(edit_components)
               if c in K_STR and incumbent_counts.get(c, 0) == 0)
