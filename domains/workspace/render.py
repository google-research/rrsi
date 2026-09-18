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
"""Render a LAB trial record (trajectory + rubric verdicts) into the text the
analyst, digester and proposer read."""

TOOL_RESULT_HEAD, TOOL_ARGS_HEAD, ASSISTANT_HEAD, MAX_CHARS = 700, 400, 1200, 300_000


def _clip(s, n):
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[:n] + f" ...[+{len(s) - n} chars]"


def render_messages(messages: list, tool_head=TOOL_RESULT_HEAD, args_head=TOOL_ARGS_HEAD,
                    asst_head=ASSISTANT_HEAD) -> str:
    call_names = {}
    for m in messages:
        for tc in (m.get("tool_calls") or []):
            if tc.get("id"):
                call_names[tc["id"]] = (tc.get("function") or {}).get("name")
    lines, step = [], 0
    for m in messages:
        role = m.get("role")
        if role == "system":
            lines.append(f"[SYSTEM PROMPT] {_clip(m.get('content'), 2000)}")
        elif role == "user":
            lines.append(f"[TASK PROMPT] {_clip(m.get('content'), 4000)}")
        elif role == "assistant":
            step += 1
            if m.get("content"):
                lines.append(f"[step {step}] ASSISTANT: {_clip(m['content'], asst_head)}")
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                lines.append(f"[step {step}] TOOL_CALL {fn.get('name')}"
                             f"({_clip(fn.get('arguments', ''), args_head)})")
        elif role == "tool":
            nm = m.get("name") or call_names.get(m.get("tool_call_id")) or "unknown_tool"
            lines.append(f"[step {step}] TOOL_RESULT {nm}: {_clip(m.get('content'), tool_head)}")
    text = "\n".join(lines)
    if len(text) > MAX_CHARS:
        head, tail = int(MAX_CHARS * 0.6), int(MAX_CHARS * 0.4)
        text = text[:head] + f"\n...[TRUNCATED {len(text) - head - tail} chars]...\n" + text[-tail:]
    return text


def render_grades(criteria: list) -> str:
    lines = []
    for i, c in enumerate(criteria):
        verdict = "PASS" if c.get("verdict") == "pass" else "FAIL"
        lines.append(f"criterion {i} [{verdict}] {_clip(c.get('title') or c.get('criteria'), 400)}\n"
                     f"  judge: {_clip(c.get('reasoning', ''), 1200)}")
    return "\n".join(lines)


def render_full(rec: dict, detail: bool = False) -> str:
    tr = rec.get("trajectory") or {}
    usage = tr.get("usage") or {}
    caps = ({"tool_head": 4000, "args_head": 1200, "asst_head": 4000} if detail else {})
    msgs = tr.get("messages") or []
    scores = rec.get("scores") or {}
    return "\n".join([
        f"TASK: {rec.get('task_id')} | practice area: {rec.get('domain')} | "
        f"trial: {rec.get('trial')}",
        f"STATUS: {tr.get('status')} | steps(assistant turns): "
        f"{sum(1 for m in msgs if m.get('role') == 'assistant')} | "
        f"compactions: {usage.get('compaction_count')} | total_tokens: {usage.get('total_tokens')}",
        f"CRITERIA: {scores.get('passed')} / {scores.get('total')} passed "
        f"(all_pass={scores.get('all_pass')}) | deliverables written: "
        f"{rec.get('output_files')}",
        "", "=== TRAJECTORY ===", render_messages(msgs, **caps),
        "", "=== FINAL OUTPUT ===", _clip(tr.get("output"), 6000),
        "", "=== RUBRIC GRADING (ground truth verdicts) ===",
        render_grades(scores.get("criteria") or []),
    ])
