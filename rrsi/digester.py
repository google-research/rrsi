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
"""Trace digester: a READ-ONLY subagent that inspects one rendered trajectory
on behalf of the analyst, so the analyst's context carries structured
conclusions, never whole traces.

Tools (jailed to the round's rendered-traces directory): read_file, glob,
grep, bash (write operations rejected). Returns structured JSON hard-capped at
DIGEST_MAX_CHARS. Lenses: failure / capability_gap / success.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from .llm import generate

MAX_TURNS = 15
DIGEST_MAX_CHARS = 6000
TOOL_OUT_CAP = 25_000
BASH_TIMEOUT = 30

ALLOWED_BASH = {"grep", "egrep", "head", "tail", "wc", "cat", "ls", "find",
                "cut", "sort", "uniq", "awk", "jq", "sed", "tr", "paste"}
DENY_BASH = re.compile(r"(>>?|`|\$\(|sed\s+[^|]*-i|\brm\b|\bmv\b|\bcp\b|"
                       r"\btee\b|\btouch\b|\bmkdir\b|\bchmod\b|\bpython)")

SCHEMAS = {
    "failure": """{"task_id": "...", "lens": "failure",
 "blocker": "one sentence: what mechanism lost the points",
 "narrative": "2-5 sentences: how the failure unfolded, concrete",
 "evidence": [{"where": "step 42", "quote": "short exact quote"}],
 "verifier_evidence": "what the grader/verifier itself reported as missed",
 "capability_note": "optional: anything the agent tried but could not do",
 "needed_instead": "1-2 sentences: what the successful path required"}""",
    "capability_gap": """{"task_id": "...", "lens": "capability_gap",
 "wanted": "what the agent was trying to accomplish",
 "why_couldnt": "what stopped it (tool limits, missing info, dead ends)",
 "evidence": [{"where": "step 12", "quote": "..."}],
 "workaround_seen": "optional: any partial workaround it attempted"}""",
    "success": """{"task_id": "...", "lens": "success",
 "habits": [{"habit": "reusable behavior that made this run clean",
             "where_shown": "step range"}],
 "risk_if_removed": "which habit is load-bearing and what breaks without it"}""",
}

SYSTEM_TMPL = """You are a trajectory digester: a read-only investigator that
inspects ONE agent trajectory in depth and returns a compact structured
digest. Another agent (the batch analyst) will rely on your digest without
reading the trace itself, so be precise and evidence-anchored.

{domain_brief}

The rendered trajectory files live in your working directory as <task_id>.txt.

Your lens for this assignment: {lens}
{focus}

You interact via STRICT JSON, one action per turn:
  {{"action": "read_file", "path": "<task_id>.txt", "offset": 1200, "limit": 300}}
      (line-based; omit offset/limit to read from the start)
  {{"action": "glob", "pattern": "*.txt"}}
  {{"action": "grep", "pattern": "regex", "path": "<task_id>.txt", "max_hits": 40}}
      (returns matching lines with line numbers)
  {{"action": "bash", "cmd": "grep -n 'FAILED' <task_id>.txt | head -30"}}
      (read-only shell: grep/head/tail/awk/jq/sed(no -i)/wc/...; any write
       operation is rejected)
  {{"action": "return", "digest": {{...}}}}

Investigate efficiently: read the grading / verifier section first (what
actually failed), grep for anchors (step numbers, tool names, key values,
error strings), then read the relevant slices. Do not read whole files top
to bottom.

Finish with action "return". The digest MUST follow this schema and MUST be
under {cap} characters total:
{schema}

Rules: quotes must be exact and short; every claim needs a "where"; do not
speculate beyond what the trace shows; no blame attribution to "model vs
harness"."""


def _bash_ok(cmd: str) -> bool:
    if DENY_BASH.search(cmd):
        return False
    for seg in re.split(r"[|;&]+", cmd):
        seg = seg.strip()
        if seg and seg.split()[0] not in ALLOWED_BASH:
            return False
    return True


def _safe(root: Path, rel: str) -> Path:
    p = (root / rel).resolve()
    if not str(p).startswith(str(root.resolve())):
        raise ValueError("path escapes the traces dir")
    return p


def _read_file(root: Path, rel: str, offset, limit) -> str:
    p = _safe(root, rel)
    if not p.exists():
        return f"ERROR: no such file {rel}"
    lines = p.read_text().splitlines()
    lo = max(0, (offset or 1) - 1)
    hi = lo + (limit or 400)
    body = "\n".join(f"{i+1}: {l}" for i, l in enumerate(lines[lo:hi], start=lo))
    return body + (f"\n...[file has {len(lines)} lines]" if hi < len(lines) else "")


def _grep(root: Path, pattern: str, rel: str, max_hits: int) -> str:
    p = _safe(root, rel)
    if not p.exists():
        return f"ERROR: no such file {rel}"
    try:
        rx = re.compile(pattern)
    except re.error as e:
        return f"ERROR: bad regex: {e}"
    out = []
    for i, line in enumerate(p.read_text().splitlines(), 1):
        if rx.search(line):
            out.append(f"{i}: {line[:400]}")
            if len(out) >= max_hits:
                out.append("...[max hits reached]")
                break
    return "\n".join(out) if out else "(no matches)"


def digest_task(traces_dir: Path, task_id: str, lens: str, domain_brief: str,
                questions: list | None = None, model: str | None = None) -> dict:
    lens = lens if lens in SCHEMAS else "failure"
    focus = ""
    if questions:
        focus = "Specific questions from the analyst you MUST answer:\n" + \
            "\n".join(f"- {q}" for q in questions[:5])
    system = SYSTEM_TMPL.format(lens=lens, focus=focus, cap=DIGEST_MAX_CHARS,
                                schema=SCHEMAS[lens], domain_brief=domain_brief)
    transcript = f"Assigned trace: {task_id}.txt\nFirst action:"
    for _ in range(MAX_TURNS):
        raw = generate(transcript, system=system, json_only=True, model=model)
        try:
            act = json.loads(raw)
        except json.JSONDecodeError:
            transcript += f"\n[you] {raw[:300]}\n[result] ERROR: invalid JSON"
            continue
        if isinstance(act, list):
            act = next((x for x in act if isinstance(x, dict)), None)
        if not isinstance(act, dict):
            transcript += ("\n[you] (non-object)\n[result] ERROR: reply with "
                           "EXACTLY ONE JSON action object, not a list or value")
            continue
        a = act.get("action")
        if a == "return":
            digest = act.get("digest") or {}
            blob = json.dumps(digest, ensure_ascii=False)
            if len(blob) > DIGEST_MAX_CHARS:
                transcript += (f"\n[you] return ({len(blob)} chars)\n[result] "
                               f"ERROR: digest is {len(blob)} chars, cap is "
                               f"{DIGEST_MAX_CHARS}. Shorten and return again.")
                continue
            digest.setdefault("task_id", task_id)
            digest.setdefault("lens", lens)
            return digest
        try:
            if a == "read_file":
                result = _read_file(traces_dir, act["path"], act.get("offset"),
                                    act.get("limit"))
            elif a == "glob":
                result = "\n".join(sorted(
                    p.name for p in traces_dir.glob(act.get("pattern", "*"))))
            elif a == "grep":
                result = _grep(traces_dir, act.get("pattern", ""),
                               act.get("path", ""), int(act.get("max_hits", 40)))
            elif a == "bash":
                cmd = act.get("cmd", "")
                if not _bash_ok(cmd):
                    result = ("ERROR: command rejected (read-only shell; allowed: "
                              "grep/head/tail/awk/jq/sed/wc/cat/ls/find/cut/sort/"
                              "uniq/tr; no redirection)")
                else:
                    r = subprocess.run(cmd, shell=True, cwd=traces_dir,
                                       capture_output=True, text=True,
                                       timeout=BASH_TIMEOUT)
                    result = (r.stdout or "") + (r.stderr or "")
            else:
                result = f"ERROR: unknown action {a}"
        except Exception as e:  # noqa: BLE001
            result = f"ERROR: {e}"
        transcript += (f"\n[you] {json.dumps(act)[:600]}\n"
                       f"[result] {str(result)[:TOOL_OUT_CAP]}")
    return {"task_id": task_id, "lens": lens,
            "error": "digester hit max turns without returning"}
