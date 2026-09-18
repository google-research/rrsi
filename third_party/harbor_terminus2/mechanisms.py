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
"""Evolvable mechanism substrate for the four structural levers.

These are inert primitives: importing this module changes nothing. The
proposer WIRES a primitive into terminus_2.py to activate a lever, then the
critic/smoke/eval gates judge whether it helps. Kept dependency-free and
defensive — the harness runs unattended on 178 trials.

Levers:
  skill    — author harness/skills/<name>/SKILL.md; call upload_skills() in
             run()/setup() so the model can discover+cat them in-container
             (progressive disclosure). See Terminus2._build_skills_section.
  memory   — Memory: fcntl-locked host-side JSONL, semantic + episodic.
             ENTITY-FREE ONLY (critic + SKILL.md enforce): store general
             procedures/lessons, never task-specific I/O — the benchmark is
             evolved and evaluated on the same tasks, so persisting concrete
             task data across trials is memorization, not improvement.
  client_tool — ToolRegistry: host-side callables the loop can run and inject.
  subagent — subcall(): one bounded, tightly-briefed extra policy call.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shlex
from pathlib import Path

STATE_DIR = Path(os.environ.get("RRSI_STATE_DIR", "/tmp/tbmh_state"))

# crude entity-ish guard for memory writes (defence-in-depth; critic is primary)
_LEAKY = re.compile(r"/app/|/tests?/|\.py::|expected|reference|answer\s*[:=]",
                    re.I)


# ---------------------------------------------------------------- memory ----
class Memory:
    """Append-only JSONL memory with fcntl locking. Two granularities:
      semantic  — general, timeless procedures/lessons (task-agnostic).
      episodic  — pattern tied to a task TYPE/situation (still entity-free).
    Reads are ALWAYS filtered (by kind + optional predicate/limit); never
    dump the whole store into context."""

    def __init__(self, state_dir: Path | str = STATE_DIR):
        self.dir = Path(state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, kind: str) -> Path:
        kind = kind if kind in ("semantic", "episodic") else "semantic"
        return self.dir / f"{kind}.jsonl"

    def write(self, kind: str, note: str, tags: list[str] | None = None,
              dedupe: bool = True) -> bool:
        """Persist one entity-free lesson. Returns False if rejected/duplicate.
        Rejects notes that look like they carry task-specific data."""
        note = (note or "").strip()
        if not note or _LEAKY.search(note):
            return False
        rec = {"note": note[:600], "tags": (tags or [])[:6]}
        p = self._path(kind)
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
        return True

    def read(self, kind: str, match: str | None = None, limit: int = 8) -> list:
        p = self._path(kind)
        if not p.exists():
            return []
        out = []
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
        return out[-limit:]

    def digest(self, kind: str = "semantic", match: str | None = None,
               limit: int = 8) -> str:
        """Compact, injectable block of filtered lessons (or '')."""
        recs = self.read(kind, match=match, limit=limit)
        if not recs:
            return ""
        lines = "\n".join(f"- {r['note']}" for r in recs)
        return f"\n\n[MEMORY — {kind} lessons from prior tasks]\n{lines}"


# ---------------------------------------------------------- client tools ----
class ToolRegistry:
    """Host-side meta-tools the agent loop can run and feed back into context.
    Register callables that compute/parse/transform (NOT touch the verifier)."""

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
def subcall(llm, brief: str, max_tokens: int = 800) -> str:
    """One bounded extra call to the SAME frozen policy. Keep the brief tight
    ('return only X'); never delegate open-ended reasoning or a whole task.
    PRIOR: sub-calls tend to hurt on small policies (early exit); measure,
    don't assume. Returns '' on any error."""
    try:
        from harbor.llms.chat import Chat
        c = Chat(llm)
        return (c.chat(brief, max_tokens=max_tokens) or "").strip()
    except Exception:  # noqa: BLE001
        return ""


# --------------------------------------------------------------- skills -----
async def upload_skills(environment, repo_skills_dir: Path | str,
                        container_dir: str = "/root/skills") -> str | None:
    """Copy authored skills (harness/skills/<name>/SKILL.md) into the task
    container and return the container path to assign to self.skills_dir so
    Terminus2._build_skills_section can advertise them. Returns None if there
    are no skills. Wire this in run() before building the skills section."""
    repo = Path(repo_skills_dir)
    mds = sorted(repo.glob("*/SKILL.md"))
    if not mds:
        return None
    try:
        await environment.exec(f"mkdir -p {shlex.quote(container_dir)}",
                               timeout_sec=10)
        for md in mds:
            name = md.parent.name
            dest = f"{container_dir}/{name}"
            await environment.exec(f"mkdir -p {shlex.quote(dest)}", timeout_sec=10)
            body = md.read_text()
            # write via heredoc-safe base64 to avoid quoting issues
            import base64
            b64 = base64.b64encode(body.encode()).decode()
            await environment.exec(
                f"echo {shlex.quote(b64)} | base64 -d > "
                f"{shlex.quote(dest + '/SKILL.md')}", timeout_sec=15)
        return container_dir
    except Exception:  # noqa: BLE001
        return None
