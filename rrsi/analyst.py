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
"""Analyze(H_t, D_evolve) -> F_t  (Algorithm 1, line 1).

The batch analyst never reads raw traces. It dispatches read-only digesters
over the incumbent's own evaluation and aggregates their digests into a
three-lens report:

  failure_modes    what blockers lost points, clustered across tasks
  capability_gaps  what the agent tried to do but could not
  success_habits   what let passing tasks finish cleanly (kept across rounds
                   as a regression guard for the proposer)
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .digester import digest_task
from .llm import generate

MAX_TURNS = 30
DIGEST_PARALLELISM = 6

SYSTEM_TMPL = """You are the batch analyst in a harness-evolution loop. A frozen
policy LLM, driven by an evolvable scaffold, ran the evolve set of a benchmark;
some trials scored well, some did not. Your job: produce a three-lens analysis
report that a harness engineer will act on.

{domain_brief}

You do NOT read traces yourself. You dispatch read-only digester subagents
that investigate one trace each and return structured digests. Budget their
use: digest FAILED / low-scoring tasks first (failure lens), prioritising
coverage of every suspected failure cluster over digesting every failure;
digest 3-5 representative SUCCESSFUL tasks (success lens, prefer ones that
resemble a big failure cluster); use the capability_gap lens or follow-up
questions where a failure digest hints the agent was blocked by the
environment or the scaffold's plumbing rather than by its own judgment.

Actions (STRICT JSON, one per turn):
  {{"action": "digest_many", "requests": [
      {{"task_id": "<id exactly as it appears in the task table>",
       "lens": "failure|capability_gap|success",
       "questions": ["optional targeted questions"]}}, ...]}}
      (up to 8 per call, they run in parallel; each returns a digest)
  {{"action": "report",
   "failure_modes": [
     {{"mode": "snake_case_label", "n_tasks": 0, "affected_tasks": [...],
      "description": "generalized mechanism, entity-free",
      "needed_instead": "...",
      "representative_evidence": [{{"task_id": "...", "where": "...", "quote": "..."}}]}}],
   "capability_gaps": [
     {{"gap": "snake_case_label", "n_tasks": 0, "affected_tasks": [...],
      "description": "what the agent could not do and why, entity-free",
      "representative_evidence": [...]}}],
   "success_habits": [
     {{"habit": "snake_case_label", "n_tasks": 0,
      "description": "the reusable behavior, entity-free",
      "risk_if_broken": "what regresses if a harness change disrupts it"}}]}}

Aggregation rules:
1. MERGE digests describing the same underlying mechanism even if worded
   differently; SPLIT a label that covers two distinct mechanisms.
2. RANK failure_modes by total impact (number of tasks weighted by how much
   score the mode costs on each).
3. Descriptions must be entity-free and task-agnostic (no task names, no
   subject-matter facts, no task-specific values); evidence quotes may
   contain them.
4. Keep prior mode names when the same mechanism recurs (stable naming);
   prior names are provided in the context.
5. Do NOT prescribe code changes and do NOT attribute blame to model vs
   scaffold.
6. If digests contradict or a cluster is unclear, dispatch follow-up digests
   with targeted questions before reporting. Report once, at the end."""


def prepare_rendered(domain, traces: dict, rendered_dir: Path) -> None:
    rendered_dir.mkdir(parents=True, exist_ok=True)
    for tid, rec in traces.items():
        out = rendered_dir / f"{tid}.txt"
        if not out.exists():
            out.write_text(domain.render_trace(rec, detail=True))


def task_table(domain, traces: dict, per_task: dict) -> str:
    rows = []
    for tid, rec in traces.items():
        tr = per_task.get(tid)
        rows.append((tr.mean if tr else 0.0, domain.task_row(tid, rec, tr)))
    return "\n".join(r for _, r in sorted(rows, key=lambda x: x[0]))


def analyze(domain, traces: dict, per_task: dict, round_dir: Path,
            prior_modes: list | None = None, prior_habits: list | None = None,
            model: str | None = None) -> dict:
    rendered_dir = round_dir / "analysis" / "rendered"
    prepare_rendered(domain, traces, rendered_dir)
    digests_dir = round_dir / "analysis" / "digests"
    digests_dir.mkdir(parents=True, exist_ok=True)
    system = SYSTEM_TMPL.format(domain_brief=domain.briefs["analyst"])
    context = "\n\n".join([
        "=== TASK TABLE (one rendered trace per task; lowest scores first) ===",
        task_table(domain, traces, per_task),
        "=== PRIOR FAILURE-MODE NAMES (for stable naming) ===",
        json.dumps([{"mode": m.get("mode"), "description": m.get("description")}
                    for m in (prior_modes or [])], ensure_ascii=False, indent=1),
        "=== PRIOR SUCCESS-HABIT NAMES (for stable naming) ===",
        json.dumps([{"habit": h.get("habit"), "description": h.get("description")}
                    for h in (prior_habits or [])], ensure_ascii=False, indent=1),
        "=== TASK ===",
        "Dispatch digesters, then produce the report. First action:",
    ])
    transcript = ""
    n_digests = 0
    for _ in range(MAX_TURNS):
        prompt = (context + "\n\n=== INTERACTION LOG ===\n" + transcript
                  + "\nReply with exactly one JSON action object.")
        raw = generate(prompt, system=system, json_only=True, model=model)
        try:
            act = json.loads(raw)
        except json.JSONDecodeError:
            transcript += f"\n[you] {raw[:300]}\n[result] ERROR: invalid JSON"
            continue
        if isinstance(act, list):
            act = next((x for x in act if isinstance(x, dict)), None)
        if not isinstance(act, dict):
            transcript += ("\n[you] (non-object)\n[result] ERROR: reply with "
                           "EXACTLY ONE JSON action object")
            continue
        a = act.get("action")
        if a == "report":
            report = {k: act.get(k) or [] for k in
                      ("failure_modes", "capability_gaps", "success_habits")}
            report["failure_modes"].sort(key=lambda m: -(m.get("n_tasks") or 0))
            report["n_digests"] = n_digests
            return report
        if a == "digest_many":
            reqs = (act.get("requests") or [])[:8]
            for r in reqs:
                if isinstance(r, dict) and r.get("task_id") is not None:
                    r["task_id"] = str(r["task_id"])
            valid = [r for r in reqs if isinstance(r, dict) and r.get("task_id") in traces]
            if reqs and not valid:
                print(f"[analyst] WARNING all {len(reqs)} requests had unknown "
                      f"task_ids; known ids look like {next(iter(traces))!r}",
                      flush=True)

            def run(r):
                d = digest_task(rendered_dir, r["task_id"], r.get("lens", "failure"),
                                domain.briefs["digester"], r.get("questions"),
                                model=model)
                (digests_dir / f"{r['task_id']}_{r.get('lens', 'failure')}.json"
                 ).write_text(json.dumps(d, ensure_ascii=False, indent=1))
                return d
            with ThreadPoolExecutor(max_workers=DIGEST_PARALLELISM) as ex:
                digests = list(ex.map(run, valid))
            n_digests += len(digests)
            result = json.dumps(digests, ensure_ascii=False)
            if len(reqs) - len(valid):
                result += f"\n[{len(reqs) - len(valid)} requests skipped: unknown task_id]"
            print(f"[analyst] {len(digests)} digests done (total {n_digests})",
                  flush=True)
            transcript += (f"\n[you] digest_many ({len(valid)} requests)\n"
                           f"[result] {result}")
        else:
            transcript += (f"\n[you] {json.dumps(act)[:300]}\n"
                           f"[result] ERROR: unknown action {a}")
    return {"failure_modes": [], "capability_gaps": [], "success_habits": [],
            "error": "analyst hit max turns", "n_digests": n_digests}


def load_digests(round_dir: Path) -> list:
    dg = round_dir / "analysis" / "digests"
    if dg.exists():
        return [json.loads(p.read_text()) for p in sorted(dg.glob("*.json"))]
    return []
