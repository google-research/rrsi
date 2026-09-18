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
"""Render one harbor trial directory into the text form the analyst, digester
and proposer read: task instruction, every episode (agent analysis/plan,
keystrokes, terminal output), run metadata and the verifier's ground truth
(reward, per-test results, test log tail)."""

import json
from pathlib import Path

OBS_HEAD, KEYSTROKES_HEAD, ASSISTANT_HEAD = 900, 500, 1200
INSTRUCTION_HEAD, MAX_CHARS = 5000, 300_000
DETAIL_CAPS = {"obs_head": 4000, "keys_head": 1500, "asst_head": 4000}


def _clip(s, n):
    s = str(s)
    return s if len(s) <= n else s[:n] + f" ...[+{len(s) - n} chars]"


def _obs_text(step: dict) -> str:
    obs = step.get("observation") or {}
    return "\n".join(str(r.get("content")) for r in obs.get("results") or []
                     if r.get("content"))


def render_steps(steps: list, obs_head=OBS_HEAD, keys_head=KEYSTROKES_HEAD,
                 asst_head=ASSISTANT_HEAD) -> str:
    lines, ep = [], 0
    for s in steps:
        src = s.get("source")
        if src == "user":
            lines.append(f"[TASK PROMPT] {_clip(s.get('message'), INSTRUCTION_HEAD)}")
        elif src == "agent":
            ep += 1
            if s.get("message"):
                lines.append(f"[step {ep}] AGENT: {_clip(s['message'], asst_head)}")
            for tc in s.get("tool_calls") or []:
                args = tc.get("arguments") or {}
                keys = args.get("keystrokes", json.dumps(args)[:200])
                lines.append(f"[step {ep}] KEYSTROKES (duration={args.get('duration')}): "
                             f"{_clip(keys, keys_head)}")
            obs = _obs_text(s)
            if obs:
                lines.append(f"[step {ep}] TERMINAL: {_clip(obs, obs_head)}")
        else:
            lines.append(f"[{src}] {_clip(s.get('message'), 500)}")
    text = "\n".join(lines)
    if len(text) > MAX_CHARS:
        head, tail = int(MAX_CHARS * 0.6), int(MAX_CHARS * 0.4)
        text = text[:head] + f"\n...[TRUNCATED {len(text) - head - tail} chars]...\n" + text[-tail:]
    return text


def render_verifier(trial_dir: Path) -> str:
    parts = []
    ctrf = trial_dir / "verifier" / "ctrf.json"
    if ctrf.exists():
        try:
            res = json.loads(ctrf.read_text()).get("results") or {}
            summ = res.get("summary") or {}
            parts.append(f"tests: {summ.get('passed', 0)} passed / "
                         f"{summ.get('failed', 0)} failed / {summ.get('tests', 0)} total")
            for t in res.get("tests") or []:
                line = f"  [{t.get('status')}] {t.get('name')}"
                msg = t.get("message") or t.get("trace")
                if msg and t.get("status") != "passed":
                    line += f"\n    {_clip(msg, 1200)}"
                parts.append(line)
        except json.JSONDecodeError:
            parts.append("(ctrf.json unreadable)")
    so = trial_dir / "verifier" / "test-stdout.txt"
    if so.exists():
        parts.append(f"--- test-stdout tail ---\n{so.read_text(errors='replace')[-4000:]}")
    return "\n".join(parts) if parts else "(no verifier output)"


def render_full(rec: dict, detail: bool = False) -> str:
    trial_dir = Path(rec["trial_dir"])
    result, traj = {}, {}
    rj, tj = trial_dir / "result.json", trial_dir / "agent" / "trajectory.json"
    for p, target in ((rj, "result"), (tj, "traj")):
        if p.exists():
            try:
                (result if target == "result" else traj).update(json.loads(p.read_text()))
            except json.JSONDecodeError:
                pass
    ar = result.get("agent_result") or {}
    meta = ar.get("metadata") or {}
    vr = (result.get("verifier_result") or {}).get("rewards") or {}
    caps = DETAIL_CAPS if detail else {}
    kw = {"obs_head": caps.get("obs_head", OBS_HEAD),
          "keys_head": caps.get("keys_head", KEYSTROKES_HEAD),
          "asst_head": caps.get("asst_head", ASSISTANT_HEAD)}
    exc = result.get("exception_info")
    parts = [
        f"TASK: {result.get('task_name')} | trial: {result.get('trial_name')}",
        f"REWARD: {vr.get('reward')} | episodes: {meta.get('n_episodes')} | "
        f"summarizations: {meta.get('summarization_count')} | "
        f"input_tokens: {ar.get('n_input_tokens')} | output_tokens: {ar.get('n_output_tokens')}",
        f"EXCEPTION: {_clip(json.dumps(exc), 1500)}" if exc else "",
        "", "=== TRAJECTORY ===", render_steps(traj.get("steps") or [], **kw),
        "", "=== VERIFIER (ground truth) ===", render_verifier(trial_dir),
    ]
    return "\n".join(p for p in parts if p != "")
