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
"""Evolvable mechanism substrate for the structural levers (apex port of the


These are inert primitives: importing this module changes nothing. The
proposer WIRES a primitive into main.py to activate a lever, then the
critic/smoke/eval gates judge whether it helps. Dependency-free and
defensive — the harness runs unattended on 240 trials.

Levers:
  skill    — author harness/skills/<name>/SKILL.md and wire a skill_use
             meta-tool in tools.py so the frozen policy can discover+cat them
             (progressive disclosure). This substrate does not force a skills
             pipeline; build one when the evidence calls for it.
  memory   — Memory: fcntl-locked JSONL under APEX_STATE_DIR, semantic +
             episodic. ENTITY-FREE ONLY (critic + SKILL.md enforce): store
             general procedures/lessons, never task-specific I/O — the
             benchmark is evolved and evaluated on the same tasks, so
             persisting concrete task data across trials is memorization.
  client_tool — ToolRegistry: in-process callables the loop can run and inject.
  subagent — subcall(): one bounded, tightly-briefed extra policy call.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
from pathlib import Path


def _state_dir() -> Path:
    d = os.environ.get("APEX_STATE_DIR", "/tmp/apex_mh_state")
    p = Path(d)
    p.mkdir(parents=True, exist_ok=True)
    return p


# crude entity-ish guard for memory writes (defence-in-depth; critic is primary)
_LEAKY = re.compile(
    r"/filesystem/|\.apps_data|task_[0-9a-f]{6}|expected|reference|"
    r"gold\s*answer|answer\s*[:=]", re.I)


# ---------------------------------------------------------------- memory ----
class Memory:
    """Append-only JSONL memory with fcntl locking under APEX_STATE_DIR. Two
    granularities: semantic (general timeless lessons) and episodic (pattern
    tied to a task TYPE/situation, still entity-free). Reads are ALWAYS
    filtered; never dump the whole store into context."""

    def __init__(self, state_dir: Path | str | None = None):
        self.dir = Path(state_dir) if state_dir else _state_dir()
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, kind: str) -> Path:
        kind = kind if kind in ("semantic", "episodic") else "semantic"
        return self.dir / f"mem_{kind}.jsonl"

    def write(self, kind: str, note: str, tags: list | None = None,
              dedupe: bool = True) -> bool:
        note = (note or "").strip()
        if not note or _LEAKY.search(note):
            return False
        rec = {"note": note[:600], "tags": (tags or [])[:6]}
        p = self._path(kind)
        try:
            with open(p, "a+") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                if dedupe:
                    f.seek(0)
                    for line in f:
                        try:
                            if json.loads(line).get("note") == rec["note"]:
                                fcntl.flock(f, fcntl.LOCK_UN)
                                return False
                        except json.JSONDecodeError:
                            pass
                f.seek(0, os.SEEK_END)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fcntl.flock(f, fcntl.LOCK_UN)
        except Exception:  # noqa: BLE001 - never crash the rollout
            return False
        return True

    def read(self, kind: str, match: str | None = None, limit: int = 8) -> list:
        p = self._path(kind)
        if not p.exists():
            return []
        out = []
        try:
            with open(p) as f:
                fcntl.flock(f, fcntl.LOCK_SH)
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if match and match.lower() not in (
                            r.get("note", "") + " " + " ".join(r.get("tags", []))
                    ).lower():
                        continue
                    out.append(r)
                fcntl.flock(f, fcntl.LOCK_UN)
        except Exception:  # noqa: BLE001
            return []
        return out[-limit:]

    def digest(self, kind: str = "semantic", match: str | None = None,
               limit: int = 8) -> str:
        recs = self.read(kind, match=match, limit=limit)
        if not recs:
            return ""
        lines = "\n".join(f"- {r['note']}" for r in recs)
        return f"\n\n[MEMORY — {kind} lessons from prior tasks]\n{lines}"


# ---------------------------------------------------------- client tools ----
class ToolRegistry:
    """In-process meta-tools the agent loop can run and feed back into context.
    Register callables that compute/parse/transform (NOT touch the judge)."""

    def __init__(self):
        self._tools: dict = {}

    def register(self, name: str, fn, desc: str):
        self._tools[name] = (fn, desc)

    def spec(self) -> str:
        if not self._tools:
            return ""
        rows = "\n".join(f"- {n}: {d}" for n, (_, d) in self._tools.items())
        return f"\n\n[CLIENT TOOLS you may request]\n{rows}"

    def call(self, name: str, args: dict) -> str:
        if name not in self._tools:
            return f"ERROR: unknown tool {name}"
        try:
            return str(self._tools[name][0](**(args or {})))[:4000]
        except Exception as e:  # noqa: BLE001 - never crash the rollout
            return f"ERROR: tool {name} raised {type(e).__name__}: {e}"


# -------------------------------------------------------------- subagent ----
async def subcall(model: str, extra_args: dict, brief: str,
                  max_tokens: int = 800, timeout: int = 120) -> str:
    """One bounded extra call to the SAME frozen policy via litellm. Keep the
    brief tight ('return only X'); never delegate open-ended reasoning or a
    whole task. Returns '' on any error (never crashes the rollout)."""
    try:
        import litellm
        r = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": brief}],
            max_tokens=max_tokens, timeout=timeout,
            **{k: v for k, v in (extra_args or {}).items()
               if k not in ("api_key",)} if extra_args else {})
        return (r.choices[0].message.content or "").strip()
    except Exception:  # noqa: BLE001
        return ""
