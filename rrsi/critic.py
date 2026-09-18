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
"""Critic(H_t, H') -> {0,1}: the leakage screen (Algorithm 1, last line).

Reads the candidate diff BEFORE any evaluation is spent and rejects edits
that encode task names, entity names, task-specific values, answers or other
suite-specific logic, as well as edits that add inert machinery. Two layers:

1. Deterministic pre-checks: a generic denylist (credentials, infra names)
   plus the domain's own patterns (grader paths, task ids, judge references).
2. LLM review (Claude Opus 4.8) of intent and content, not style.

A rejection sends the objections back to the proposer for a bounded number of
repair rounds; a candidate that cannot be repaired is dropped and recorded in
the history without a measurement.
"""

from __future__ import annotations

import json
import re

from .llm import generate

GENERIC_PATTERNS = [
    (r"AIza[0-9A-Za-z_-]{35}|sk-[A-Za-z0-9]{20,}|api_key\s*=\s*[\"\'][^\"\']{8,}", "credential in diff"),
]

SYSTEM_TMPL = """You are a strict reviewer of harness (agent scaffold) code changes
in an evolution loop. The harness is evolved against the very tasks it is
measured on, so anti-overfitting review is critical. The change was made by
another model in response to aggregated failure modes. Your job is to catch
changes that would be cheating, degenerate, or self-destructive.

{domain_brief}

REJECT if ANY of:

1. LEAKAGE / TASK-SPECIALIZATION: the diff hard-codes knowledge of specific
   evolve-set tasks: task names or ids, entity/company/case names, task-specific
   file names, expected outputs or answers, magic constants that only make
   sense for one task, per-task templates, a list of the suite's domains, or
   branching on task-identifying features ("if the task mentions X"). General
   professional practice and general procedure are fine. Litmus test: would
   this change still make sense, and still help, on an unfamiliar task from a
   different suite in the same kind of work?
2. DEGENERATE: the diff is effectively a no-op (dead code, unused flag,
   comment-only) while claiming a mechanism, or it deletes/disables an
   existing safety mechanism (context compression, output truncation, error
   handling, termination guards) without a working replacement.
3. GRADER GAMING: the change tries to read, detect, reconstruct or influence
   the verifier / judge / rubric at runtime, hard-codes a threshold or reference
   value that could only have come from a grader, or games the completion or
   output format for the grader rather than improving actual task performance.
4. UNDECLARED BUNDLING: multiple DECLARED edits per candidate are allowed and
   normal (the declared edit list is in the payload). Reject only if the diff
   contains independent changes not covered by ANY declared edit, or one
   declared edit secretly bundles several independent mechanisms.
5. RUNTIME MEMORY / SKILL LEAKAGE: if an edit adds a memory/state or skill
   mechanism, inspect what it PERSISTS or INJECTS at runtime. Entity-free
   general procedures are fine. REJECT if the code path can persist or inject
   task-specific runtime data across trials (captured file contents, tool
   outputs, computed answers, task/entity/file names, per-task solutions keyed
   by task identity): the harness is evaluated on the tasks it evolves on, so
   that is memorization, not a general improvement.
6. UNBOUNDED WORK: an added check, retry or "keep improving" loop with no
   give-up path, or anything that implies finishing should wait for a
   confirmation that may never come. A bounded pass with an explicit exit is
   fine.

Otherwise ACCEPT. You are reviewing intent and content, not style.
Runtime correctness (undefined names, crashes, syntax) is NOT your job:
deterministic compile/constructor/smoke checks handle that after you, and you
only see the diff (not the full files), so never speculate that something is
undefined or would crash.
Return STRICT JSON:
{{"verdict": "accept" | "reject", "reasons": ["..."], "risk_notes": ["..."]}}"""


def precheck(diff: str, domain_patterns: list) -> list[str]:
    hits = []
    for pat, why in GENERIC_PATTERNS + list(domain_patterns or []):
        if re.search(pat, diff):
            hits.append(why)
    return hits


def review(domain, diff: str, summary: str, targets_mode: str,
           edits: list | None = None, state_files: str = "",
           model: str | None = None) -> dict:
    hard = precheck(diff, domain.critic_patterns)
    if hard:
        return {"verdict": "reject", "reasons": [f"precheck: {h}" for h in hard],
                "risk_notes": []}
    if not diff.strip():
        return {"verdict": "reject", "reasons": ["empty diff"], "risk_notes": []}
    system = SYSTEM_TMPL.format(domain_brief=domain.briefs["critic"])
    payload = (
        f"CANDIDATE SUMMARY: {summary}\nTARGETS: {targets_mode}\n\n"
        f"=== DECLARED EDITS (independent changes in this candidate) ===\n"
        f"{json.dumps(edits or [], ensure_ascii=False, indent=1)[:20_000]}\n\n"
        f"=== DIFF ===\n{diff[:120_000]}\n\n"
        f"=== STATE FILES (if a mechanism persists state) ===\n{state_files[:20_000]}"
    )
    last = ""
    for _ in range(3):
        out = generate(payload, system=system, json_only=True, model=model)
        last = out
        try:
            v = json.loads(out)
            if isinstance(v, dict) and v.get("verdict") in ("accept", "reject"):
                return v
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", out, re.S)
            if m:
                try:
                    return json.loads(m.group(0))
                except json.JSONDecodeError:
                    pass
    return {"verdict": "reject",
            "reasons": [f"critic output unparseable after 3 attempts: {last[:200]}"],
            "risk_notes": []}
