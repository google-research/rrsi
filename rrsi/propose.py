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
"""The regularized proposer P_reg(. | H_t, F_t, L_t, b_t, E_t) of Algorithm 1.

A Claude Opus 4.8 coding agent that edits the harness IN PLACE inside a
candidate's worktree through a strict-JSON action protocol and finishes with
done(edits=[...]). Every edit is tagged with (component l', hypothesis h');
the diff d' is taken from git afterwards. The agent is conditioned on:

  F_t   the analyst's three-lens report, the per-task digests and read-only
        access to the rendered trajectories
  L_t   the edit history (component, hypothesis, Delta S, Delta C, accepted)
  b_t   the annealed edit budget: at most b_t independent edits (||z_t||_0 <= b_t)
  E_t   exploration directives (stall flag, untried components, reserved slots)
  B_t   components to prune, with the accepted machinery to remove

Budget, reserved exploration slots and tagging are enforced in done(): an
over-budget or untagged submission is bounced back, never accepted.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .components import K
from .llm import generate

MAX_TURNS = 40
MAX_EDITS = 80
TRACE_READ_CAP = 60_000

SYSTEM_TMPL = """You are a harness engineer agent. You directly modify the source
code of an LLM-agent scaffold (the "harness") to fix recurring failure modes
observed on an evolve set of tasks. The policy LLM is frozen and is a DIFFERENT
model from you: do not assume it shares your capabilities, habits or judgment.
Improve the harness from ITS perspective, using the trajectories as evidence of
how it actually behaves. The ONLY thing you can change is the scaffold code in
your working directory; the tool environment, the grader and the task set are
frozen.

{domain_brief}

You interact through a STRICT JSON protocol: reply with EXACTLY ONE action
object per turn, nothing else. Actions:
  {{"action": "list_files"}}
  {{"action": "read_file", "path": "relative/path"}}
  {{"action": "list_traces"}}
      (this round's traces: task_id, score, status, steps, modes hit)
  {{"action": "read_trace", "task_id": "<id from the task list>", "from_step": 30,
   "to_step": 60, "detail": true}}
      (READ-ONLY rendered trajectory segment: the raw evidence behind the
      failure modes. Step range optional. "detail": true expands per-message
      caps so you can see what tool results actually contained. The grading
      verdict is always included.)
  {{"action": "edit_file", "path": "p", "old": "exact substring", "new": "replacement"}}
      (old must occur EXACTLY ONCE in the file)
  {{"action": "write_file", "path": "p", "content": "full file content"}}
      (for NEW files only; never overwrite an existing file this way. Paths are
      relative to the harness package. A new module must be imported from the
      entry module to do anything, and it IS included in the reviewed diff.)
  {{"action": "done", "summary": "one-line summary of this candidate",
   "edits": [
     {{"id": "C1",
      "component": "one of: {components}",
      "hypothesis": "one sentence: the mechanism and WHY it should move the score",
      "targets_mode": "failure mode / capability gap / habit it targets",
      "why_not_lower_lever": "why a plain instruction edit would NOT fix this
          (or, for a prompt edit, why prose IS the right lever here)",
      "trigger_condition": "the exact, checkable condition under which the
          mechanism activates ('always' is almost never right)",
      "predicted_affected": ["<task id>", "..."],
      "retroactive_check": "three-part counterfactual: (corrective) which cited
          failing tasks would have moved had this existed, walking the actual
          trajectory; (preservative) which success habits could this disrupt
          and why it won't; (transfer) why it generalizes beyond this evolve set",
      "regression_risk": "what could break outside predicted_affected"}},
     ...]}}

THERE IS NO ABORT ACTION. You must ship a candidate. A round that ships
nothing tests nothing: the history records what was MEASURED, and a mechanism
you declined to build has no measurement behind it. If your best idea violates
a hard rule, it is not your best idea; construct a different one, preferably on
a component the history shows was never exercised.

done() contract:
- An edit = ONE independent, attributable change (it works on its own and can
  answer "which tasks will it move" by itself). Dependent parts are ONE edit.
  Ship at most THIS ROUND'S EDIT BUDGET b_t given in the context, never more.
  Ship fewer if the evidence supports fewer.
- Every edit names its `component` from the fixed vocabulary above. The tag
  is validated against the diff; a mislabelled edit is re-tagged from the diff.
- If the context says a RESERVED EXPLORATION SLOT applies to you, at least one
  edit must be on one of the listed never-exercised components.
- If the context lists COMPONENTS TO PRUNE that hold accepted machinery, an
  edit that removes that machinery is a legitimate edit (component = the
  pruned component, hypothesis = "prune: ..."). Prefer it when the evidence
  says the machinery stopped earning its place.
- predicted_affected lists CONCRETE task ids from this round's traces.
  Predictions are checked against the evaluation and your hit/miss record
  (scoreboard) is shown back to you; over-claiming counts against you.

You must follow the constitution (SKILL.md) in the context: no task-specific
entities, names or values anywhere in code, prompts or state; every edit
targets a reported failure mode, capability gap or habit; keep each edit's
diff scoped to its mechanism; the code runs unattended on every task in the
set, so an unhandled exception kills the whole candidate. The interface
contract (module entry points, class names, trajectory output schema) stays
unchanged; model name, step budget and timeouts are injected externally so
editing them has no effect. All code and comments in English."""


class Workspace:
    """Path-jailed file operations inside one candidate's harness directory."""

    def __init__(self, harness_dir: Path, source_exts: set):
        self.dir = Path(harness_dir).resolve()
        self.exts = source_exts

    def _safe(self, rel: str) -> Path:
        rel = str(rel).lstrip("/")
        name = self.dir.name
        if rel == name or rel.startswith(name + "/"):
            rel = rel[len(name):].lstrip("/")
        p = (self.dir / rel).resolve()
        if not str(p).startswith(str(self.dir) + os.sep) and p != self.dir:
            raise ValueError(f"path escapes harness dir: {rel}")
        return p

    def list_files(self) -> str:
        return "\n".join(f"{p.relative_to(self.dir)} ({p.stat().st_size}B)"
                         for p in sorted(self.dir.rglob("*"))
                         if p.is_file() and "__pycache__" not in p.parts)

    def read(self, rel: str) -> str:
        p = self._safe(rel)
        return p.read_text() if p.exists() else f"ERROR: no such file {rel}"

    def edit(self, rel: str, old: str, new: str) -> str:
        p = self._safe(rel)
        if not p.exists():
            return f"ERROR: no such file {rel}"
        src = p.read_text()
        n = src.count(old)
        if n == 0:
            return "ERROR: old string not found (must match exactly, including whitespace)"
        if n > 1:
            return f"ERROR: old string occurs {n} times; provide a longer unique context"
        p.write_text(src.replace(old, new, 1))
        return f"OK: edited {rel}"

    def write(self, rel: str, content: str) -> str:
        p = self._safe(rel)
        if p.suffix not in self.exts:
            return f"ERROR: extension {p.suffix} not allowed"
        if p.exists():
            return f"ERROR: {rel} exists; use edit_file"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"OK: created {rel}"

    def dump(self) -> str:
        parts = []
        for p in sorted(self.dir.rglob("*")):
            if p.is_file() and p.suffix in self.exts and "__pycache__" not in p.parts:
                parts.append(f"===== FILE: {p.relative_to(self.dir)} =====\n{p.read_text()}")
        return "\n\n".join(parts)


def _list_traces(domain, traces: dict, per_task: dict, findings: list) -> str:
    modes_by_task: dict = {}
    for f in findings or []:
        label = f.get("blocker") or f.get("wanted") or f.get("lens") or "?"
        modes_by_task.setdefault(str(f.get("task_id")), []).append(str(label)[:80])
    rows = []
    for tid, rec in traces.items():
        tr = per_task.get(tid)
        rows.append((tr.mean if tr else 0.0,
                     domain.task_row(tid, rec, tr)
                     + f" | modes={modes_by_task.get(tid, [])}"))
    return "\n".join(r for _, r in sorted(rows, key=lambda x: x[0])) or \
        "ERROR: no traces available"


def _read_trace(domain, traces: dict, task_id, from_step, to_step, cache: dict,
                detail: bool) -> str:
    task_id = str(task_id)
    rec = traces.get(task_id)
    if rec is None:
        return f"ERROR: no trace for {task_id} (use list_traces for valid ids)"
    key = (task_id, bool(detail))
    if key not in cache:
        cache[key] = domain.render_trace(rec, detail=bool(detail))
    rendered = cache[key]
    if from_step is None and to_step is None:
        return rendered[:TRACE_READ_CAP] + (
            f"\n...[capped at {TRACE_READ_CAP} chars; use from_step/to_step]"
            if len(rendered) > TRACE_READ_CAP else "")
    lo, hi = int(from_step or 0), int(to_step or 10**9)
    keep, in_grades = [], False
    for line in rendered.splitlines():
        if line.startswith("=== ") and ("GRAD" in line or "VERIFIER" in line):
            in_grades = True
        m = re.match(r"\[step (\d+)\]", line)
        if in_grades or m is None or lo <= int(m.group(1)) <= hi:
            keep.append(line)
    out = "\n".join(keep)
    return out[:TRACE_READ_CAP] + (
        f"\n...[capped at {TRACE_READ_CAP} chars]" if len(out) > TRACE_READ_CAP else "")


def propose(domain, harness_dir: Path, report: dict, history_rows: list,
            skill_md: str, patterns_md: str, budget: int, explore: dict,
            reserved_slot: bool, prune_set: list, traces: dict | None = None,
            per_task: dict | None = None, findings: list | None = None,
            scoreboard: list | None = None, variant_brief: str = "",
            repair_brief: dict | None = None, model: str | None = None) -> dict:
    """Run the proposer agent for one candidate variant. Edits files in place
    under harness_dir. Returns {status, edits, summary, n_edits, log}."""
    ws = Workspace(harness_dir, domain.source_exts)
    traces = traces or {}
    per_task = per_task or {}
    render_cache: dict = {}
    system = SYSTEM_TMPL.format(domain_brief=domain.briefs["proposer"],
                                components=" | ".join(K))
    stable = "\n\n".join([
        "=== CONSTITUTION (SKILL.md) ===", skill_md,
        "=== PATTERN LIBRARY (PATTERNS.md) ===", patterns_md,
    ])
    explore_text = explore.get("text", "")
    if reserved_slot:
        explore_text += ("\n\nRESERVED EXPLORATION SLOT: this variant holds one. At "
                         "least one of your edits MUST be on a never-exercised "
                         f"component from: {explore.get('untried')}.")
    prune_text = (json.dumps(prune_set, ensure_ascii=False, indent=1)
                  if prune_set else "(none)")
    context = "\n\n".join([
        *(["=== THIS VARIANT'S BRIEF ===", variant_brief] if variant_brief else []),
        "=== EDIT HISTORY L_t (every measured edit: component, hypothesis, "
        "Delta S, Delta C, accepted). A rejected mechanism is negative evidence; "
        "do not redraw it unchanged. An accepted one carries the gain it "
        "produced; refine what has known credit, not what merely preceded a "
        "rise. ===",
        json.dumps(history_rows, ensure_ascii=False, indent=1),
        "=== ATTRIBUTION SCOREBOARD (how past edits' predictions fared; "
        "unpredicted_regressions are tasks an edit likely broke) ===",
        json.dumps(scoreboard or [], ensure_ascii=False, indent=1),
        "=== EXPLORATION DIRECTIVES E_t ===", explore_text,
        "=== COMPONENTS TO PRUNE B_t (exercised, no strictly improving edit in "
        "the recent window; remove the accepted machinery listed, it has "
        "stopped earning its place) ===", prune_text,
        "=== THREE-LENS ANALYSIS REPORT F_t (failure modes ranked; capability "
        "gaps often need tool/plumbing fixes; success_habits are behaviors "
        "your change MUST NOT break) ===",
        json.dumps(report, ensure_ascii=False, indent=1),
        "=== PER-TASK DIGESTS (evidence anchors; use read_trace for raw evidence) ===",
        json.dumps(findings or [], ensure_ascii=False, indent=1),
        "=== CURRENT HARNESS SOURCE H_t ===", ws.dump(),
        "=== THIS ROUND'S EDIT BUDGET b_t ===",
        f"You may ship AT MOST {budget} independent edit(s) in this candidate "
        f"(the budget anneals over the run: early rounds explore, late rounds "
        f"make single attributable changes). Ship fewer if the evidence "
        f"supports fewer.",
        "=== TASK ===",
        ("REPAIR ROUND: your previous edits for this candidate are already in "
         "the working tree (reflected in CURRENT HARNESS SOURCE above). The "
         "reviewer raised the objections below. Fix ONLY what the objections "
         "require (remove leaked content, split or re-declare edits, wire up "
         "dead code, or delete the offending part) with minimal additional "
         "edits, then call done again with the corrected edits array.\n\n"
         "=== REVIEWER OBJECTIONS ===\n"
         + json.dumps(repair_brief, ensure_ascii=False, indent=1) + "\nFirst action:")
        if repair_brief else
        "Address the highest-impact failure modes / capability gaps within your "
        "edit budget. Implement via the JSON actions, then call done with the "
        "edits array. First action:",
    ])

    log, transcript, n_edits, aborts_left = [], "", 0, 3
    for _turn in range(MAX_TURNS):
        prompt = ("=== ROUND CONTEXT ===\n" + context
                  + "\n\n=== INTERACTION LOG ===\n" + transcript
                  + "\nReply with exactly one JSON action object.")
        raw = generate(prompt, system=system, json_only=True, model=model,
                       cache_prefix=stable)
        try:
            act = json.loads(raw)
        except json.JSONDecodeError:
            transcript += f"\n[you] {raw[:500]}\n[result] ERROR: not valid JSON"
            continue
        if isinstance(act, list):
            act = next((x for x in act if isinstance(x, dict)), None)
        if not isinstance(act, dict):
            transcript += ("\n[you] (non-object)\n[result] ERROR: reply with "
                           "EXACTLY ONE JSON action object")
            continue
        a = act.get("action")
        log.append(act if a in ("done", "abort") else
                   {k: (v if len(str(v)) < 200 else str(v)[:200] + "...")
                    for k, v in act.items()})
        if a == "done":
            edits = act.get("edits") or act.get("candidates") or []
            problems = []
            if n_edits == 0 and edits:
                transcript += ("\n[you] done\n[result] ERROR: you declared edits but "
                               "made ZERO file changes. Implement them with "
                               "edit_file/write_file, then call done.")
                continue
            if n_edits > 0 and not edits:
                problems.append("edits array is empty")
            if len(edits) > budget:
                problems.append(f"{len(edits)} edits exceed the budget b_t = {budget}")
            for e in edits:
                miss = [f for f in ("id", "component", "hypothesis", "targets_mode",
                                    "predicted_affected", "retroactive_check")
                        if not e.get(f)]
                if miss:
                    problems.append(f"edit {e.get('id', '?')} missing {miss}")
                if e.get("component") and str(e["component"]).lower() not in K:
                    problems.append(f"edit {e.get('id')} component "
                                    f"{e['component']!r} not in {K}")
            if reserved_slot and explore.get("untried") and edits and not any(
                    str(e.get("component", "")).lower() in explore["untried"]
                    for e in edits):
                problems.append("this variant holds a RESERVED EXPLORATION SLOT: at "
                                "least one edit must be on a never-exercised "
                                f"component from {explore['untried']}")
            if problems and n_edits > 0:
                transcript += (f"\n[you] done\n[result] ERROR: {problems}. Call done "
                               f"again fixed (drop or merge edits if over budget; "
                               f"add the required edit if a slot is reserved).")
                continue
            for e in edits:
                e["component"] = str(e.get("component", "")).lower()
                e.setdefault("mechanism", e.get("hypothesis"))
            return {"status": "done", "summary": act.get("summary"),
                    "edits": edits,
                    "mechanism": act.get("summary") or
                    "; ".join(str(e.get("hypothesis")) for e in edits)[:200],
                    "targets_mode": ", ".join(str(e.get("targets_mode"))
                                              for e in edits)[:200],
                    "n_edits": n_edits, "log": log}
        if a == "abort":
            if aborts_left > 0:
                aborts_left -= 1
                transcript += ("\n[you] abort\n[result] ERROR: there is no abort action. "
                               "Pick the most defensible mechanism you can build "
                               "within the hard rules, implement it, and call done. "
                               "A rejection is data; an abort is not.")
                continue
            return {"status": "abort", "reason": act.get("reason"),
                    "edits": [], "n_edits": n_edits, "log": log}
        try:
            if a == "list_files":
                result = ws.list_files()
            elif a == "read_file":
                result = ws.read(act["path"])
            elif a == "list_traces":
                result = _list_traces(domain, traces, per_task, findings or [])
            elif a == "read_trace":
                result = _read_trace(domain, traces, act.get("task_id", ""),
                                     act.get("from_step"), act.get("to_step"),
                                     render_cache, act.get("detail", False))
            elif a == "edit_file":
                if n_edits >= MAX_EDITS:
                    result = "ERROR: edit budget exhausted; call done"
                else:
                    result = ws.edit(act["path"], act["old"], act["new"])
                    n_edits += result.startswith("OK")
            elif a == "write_file":
                if n_edits >= MAX_EDITS:
                    result = "ERROR: edit budget exhausted; call done"
                else:
                    result = ws.write(act["path"], act["content"])
                    n_edits += result.startswith("OK")
            else:
                result = f"ERROR: unknown action {a}"
        except Exception as e:  # noqa: BLE001
            result = f"ERROR: {e}"
        cap = 150_000 if a in ("read_file", "list_files", "read_trace",
                               "list_traces") else 4000
        transcript += f"\n[you] {json.dumps(act)[:1500]}\n[result] {str(result)[:cap]}"
    return {"status": "max_turns", "edits": [], "n_edits": n_edits, "log": log}
