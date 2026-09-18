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
"""Render one graded trial into the text form the analyst/digester read.

A trial record bundles everything the ground truth depends on:

  - the task, its engineering domain, the run status / step count / tokens
  - the GRADING block: the three verifier axes plus -- the part that carries the
    diagnosis -- the verifier's own per-sub-criterion `details`
  - the PAYLOAD the agent actually submitted (the only graded artefact)
  - the trajectory (tool calls + results), clipped per message

The `details` block is the reason this benchmark is worth evolving on. A scalar
score says a design was wrong; `details` says WHICH part was wrong -- "iou_score
0.71 against pass_threshold 0.8", "phase_margin correct, settling_time wrong",
"schema validation failed on field `config`". Those distinguish failure families
that need opposite mechanisms, and none of them is visible in the score.

"""

import json

TOOL_RESULT_HEAD = 700
TOOL_ARGS_HEAD = 400
ASSISTANT_HEAD = 1200
MAX_CHARS = 300_000
DETAILS_HEAD = 2500
# The problem statement was hardcoded at 3000 characters and, unlike every other
# cap, was NOT widened by detail mode -- so the one thing a reader most needs to
# reason about a failure was the one thing always truncated. 26 of the 61 task
# prompts exceed 3000 characters (the longest is 111k), meaning the proposer was
# diagnosing 43% of tasks against a cut-off list of constraints.
TASK_PROMPT_HEAD = 3000
TASK_PROMPT_HEAD_DETAIL = 40_000


def _clip(s, n):
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[:n] + f" ...[+{len(s) - n} chars]"


def render_messages(messages: list, tool_head: int = TOOL_RESULT_HEAD,
                    args_head: int = TOOL_ARGS_HEAD,
                    asst_head: int = ASSISTANT_HEAD,
                    prompt_head: int = TASK_PROMPT_HEAD) -> str:
    # Tool RESULT messages usually carry no `name`, so the rendered trace said
    # "TOOL_RESULT None: ..." and the reader could not tell which tool produced
    # which output -- on a run whose whole diagnosis is "what did code_exec
    # actually return", that is the single most important label in the file.
    # The name is recoverable from the CALL that the result answers.
    call_names = {}
    for m in messages:
        for tc in (m.get("tool_calls") or []):
            if tc.get("id"):
                call_names[tc["id"]] = (tc.get("function") or {}).get("name")

    lines = []
    step = 0
    for m in messages:
        role = m.get("role")
        if role == "system":
            lines.append(f"[SYSTEM PROMPT] {_clip(m.get('content'), 2500)}")
        elif role == "user":
            lines.append(f"[TASK PROMPT] {_clip(m.get('content'), prompt_head)}")
        elif role == "assistant":
            step += 1
            content = m.get("content")
            if content:
                lines.append(f"[step {step}] ASSISTANT: {_clip(content, asst_head)}")
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                lines.append(f"[step {step}] TOOL_CALL {fn.get('name')}"
                             f"({_clip(fn.get('arguments', ''), args_head)})")
        elif role == "tool":
            nm = (m.get("name")
                  or call_names.get(m.get("tool_call_id"))
                  or "unknown_tool")
            lines.append(f"[step {step}] TOOL_RESULT {nm}: "
                         f"{_clip(m.get('content'), tool_head)}")
    text = "\n".join(lines)
    if len(text) > MAX_CHARS:
        head, tail = int(MAX_CHARS * 0.6), int(MAX_CHARS * 0.4)
        text = (text[:head] + f"\n...[TRUNCATED {len(text) - head - tail} chars]...\n"
                + text[-tail:])
    return text


def render_grading(task: dict, verdict: dict, diag: dict | None = None,
                   details_head: int = DETAILS_HEAD) -> str:
    """The ground truth for this trial, most diagnostic line first.

    Read the FAILURE CLASS line before anything else. It splits every loss into
    kinds that need different mechanisms:

      NO SUBMISSION  - the run ended without writing a payload at all. Nothing
                       about engineering explains this; it is termination,
                       budget, or plumbing.
      MALFORMED      - a payload was written but the verifier could not use it:
                       not a literal, missing a schema field, wrong type. The
                       design may have been right and was thrown away at the
                       last step. This is a contract-compliance bug.
      INVALID DESIGN - the submission parsed but violated the task's hard
                       constraints. The agent proposed something the physics or
                       the spec forbids.
      UNDER TARGET   - a valid design that simply did not perform well enough.
                       This is the genuinely engineering-limited case, and the
                       only one where "the agent needed to compute better" is
                       the right reading.
    """
    verdict = verdict or {}
    diag = diag or {}
    status = verdict.get("status") or "ok"
    passed = bool(verdict.get("passed"))
    valid = float(verdict.get("valid") or 0.0)
    score = verdict.get("combined_score")
    err = str(verdict.get("error") or "")

    if diag.get("no_payload"):
        cls = ("NO SUBMISSION - the run never wrote a payload. This is a "
               "termination / budget / plumbing failure, not an engineering one.")
    elif status != "ok":
        cls = (f"VERIFIER {status} - the verifier itself did not complete "
               f"({_clip(err, 200)}). Treat as infrastructure, not evidence "
               f"about the design.")
    elif "PAYLOAD" in err or "literal" in err.lower() or "validation" in err.lower():
        cls = ("MALFORMED SUBMISSION - a payload was written but the verifier "
               "could not use it (not a literal / missing or mistyped schema "
               "field). Whatever the design was, it was discarded at the last "
               "step. This is a contract-compliance failure.")
    elif valid < 1.0:
        cls = ("INVALID DESIGN - the submission parsed but violated the task's "
               "hard constraints.")
    elif passed:
        cls = "PASSED."
    else:
        cls = ("UNDER TARGET - a valid design that did not meet the task's "
               "performance gate. This is the engineering-limited case.")

    lines = [f"FAILURE CLASS: {cls}"]
    lines.append(
        f"SCORE: combined_score={score} (raw {verdict.get('raw_score')} of "
        f"{verdict.get('score_max')}) | passed={passed} | valid={valid}")
    if err:
        lines.append(f"VERIFIER ERROR: {_clip(err, 500)}")
    det = verdict.get("details")
    if det is not None:
        lines.append("VERIFIER DETAILS (which sub-checks the design met and "
                     "missed -- read this to see WHAT was wrong, not just that "
                     "something was):")
        try:
            lines.append(_clip(json.dumps(det, indent=1, default=str), details_head))
        except Exception:  # noqa: BLE001
            lines.append(_clip(str(det), details_head))
    lines.append(
        f"AGENT ACTIVITY: {diag.get('n_steps')} steps, "
        f"{diag.get('n_code_exec')} code_exec calls "
        f"({diag.get('n_code_exec_failed')} of them errored), "
        f"{diag.get('n_write_file')} file writes, "
        f"{diag.get('n_read')} reads | wrote the payload "
        f"{diag.get('n_payload_writes')} time(s)")
    return "\n".join(lines)


def render_full(rec: dict, detail: bool = False) -> str:
    """rec = {task, meta, verdict, payload, traj, diag}. detail=True expands the
    per-message caps so content-level failures (what the agent actually computed,
    what it actually submitted) are visible."""
    task = rec.get("task") or {}
    meta = rec.get("meta") or {}
    traj = rec.get("traj") or {}
    caps = ({"tool_head": 4000, "args_head": 1200, "asst_head": 4000,
             "prompt_head": TASK_PROMPT_HEAD_DETAIL}
            if detail else {})
    payload_cap = 30000 if detail else 8000
    messages = traj.get("messages") or []
    return "\n".join([
        f"TASK: {task.get('id')} | domain: {task.get('domain')} | "
        f"trial: {rec.get('trial')}",
        f"STATUS: {meta.get('status')} | assistant turns: "
        f"{sum(1 for m in messages if m.get('role') == 'assistant')} | "
        f"total_tokens: {meta.get('total_tokens')} | wall_s: {meta.get('wall_s')}",
        "",
        "=== GRADING (ground truth) ===",
        render_grading(task, rec.get("verdict") or {}, rec.get("diag") or {},
                       details_head=(8000 if detail else DETAILS_HEAD)),
        "",
        "=== SUBMITTED PAYLOAD (the only graded artefact) ===",
        _clip(rec.get("payload"), payload_cap),
        "",
        "=== TRAJECTORY ===",
        render_messages(messages, **caps),
    ])


def load_trial(job_dir, task: dict, trial: int) -> dict | None:
    """Assemble one trial record from a job dir; None if it never ran."""
    from pathlib import Path

    td = Path(job_dir) / task["id"] / f"t{trial}"
    if not (td / "meta.json").is_file():
        return None
    rec = {"task": task, "trial": trial,
           "meta": json.loads((td / "meta.json").read_text())}
    vp, pp, tp = td / "verdict.json", td / "payload.py", td / "traj.json"
    rec["verdict"] = json.loads(vp.read_text()) if vp.is_file() else {}
    rec["payload"] = pp.read_text(errors="replace") if pp.is_file() else ""
    rec["traj"] = json.loads(tp.read_text()) if tp.is_file() else {}
    try:
        import sys as _sys
        from pathlib import Path as _P
        _sys.path.insert(0, str(_P(__file__).resolve().parent / "bench"))
        _sys.path.insert(0, str(_P(__file__).resolve().parent))
        import diagnostics as _D
        rec["diag"] = _D.trial_stats(rec["traj"], rec["meta"])
    except Exception:  # noqa: BLE001 - evidence rendering must never kill a round
        rec["diag"] = {}
    return rec
