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

#!/usr/bin/env python3
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
"""Run the scaffold over Frontier-Engineering tasks (the OOD verdict surface).

Separate from bench/run_tasks.py because the DELIVERABLE CONTRACT is different,
and that difference is the point of the surface: on EngDesign the agent writes a
literal `PAYLOAD` dict describing a design, while here it edits a PROGRAM that
the verifier then executes. A scaffold that only learned to fill in a dict has
learned nothing that transfers; one that learned how to work an unfamiliar
engineering problem end to end should carry over.

Everything else is deliberately identical to the evolve-set runner -- same
engine, same jailed gateway, same frozen policy, same k -- so the only thing
that differs between the two surfaces is the benchmark.

TWO DEVIATIONS FROM UPSTREAM, both recorded because they affect what the numbers
mean:

  * Upstream's benchmark is iterate-to-improve-within-budget. This is SINGLE
    SHOT: the agent gets one pass and the result is graded. Absolute numbers are
    therefore not comparable to Frontier's leaderboard; they are a consistent
    A/B between two scaffolds.
  * Frontier lists `verification/` as read-only context the solver may see, and
    this runner honours that (it copies what the task's own manifest exposes).
    That is upstream's design, not ours, and both arms face it identically.
"""

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent.parent / "third_party" / "archipelago"))
os.environ.setdefault("RRSI_HARNESS_MODULE", "harness_eng.main")
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from runner.agents.models import AgentRunInput  # noqa: E402
from runner.agents.registry import get_agent_impl  # noqa: E402
from harness_eng.prompts import SYSTEM_PROMPT  # noqa: E402

JOBS = Path(config.RUNS) / "jobs"
EXCLUDED_DOMAINS = {"EngDesign"}

# The task wrapper. Kept structurally parallel to the evolve set's, because a
# scaffold that only works with one phrasing of the contract has learned the
# phrasing rather than the job. What changes is the deliverable: a program the
# verifier will run, not a literal dict.
#
# IT MUST STAY AS NEUTRAL AS THE EVOLVE SET'S WRAPPER. A single helpful line
# here (for example telling the agent to pass `workdir` on every code_exec
# call) can pre-empt exactly the failure an accepted scaffold mechanism exists
# to recover from; the OOD measurement would then read a handicap written into
# the wrapper, not a property of the scaffold. Anything the wrapper teaches,
# the scaffold no longer gets credit for.
TASK_TEMPLATE = """=== ENGINEERING TASK ===
The full problem statement is in the workspace. Read it first:

  {task_md}

=== WORKSPACE ===
{workspace}

Everything the task provides is inside this workspace. There is no web access.
Do not read or write anything outside it.

=== DELIVERABLE (this file is the ONLY thing that counts) ===
{candidate_file}

Edit that file. It is a Python program which will be EXECUTED by an automated
evaluator after you finish, and its behaviour is what is scored. It already
contains a working baseline solution, so improving it is the job; leaving it
unchanged scores whatever the baseline scores.

Work is only credited through this file. Anything you print, explain in chat, or
leave in a scratch file earns nothing.
"""


def load_tasks(manifest: Path) -> list[dict]:
    out = []
    for t in json.loads(manifest.read_text()):
        fr = t.get("frontier") or {}
        rel = fr.get("task_rel") or ""
        if not rel or rel.split("/")[0] in EXCLUDED_DOMAINS:
            continue
        out.append({"id": rel.replace("/", "__"), "rel": rel,
                    "domain": rel.split("/")[0], "frontier": fr})
    return out


def gradeable(froot: Path, t: dict) -> bool:
    """A task whose family venv was never built cannot be graded, so running the
    agent on it would burn a rollout for a number nobody can read. Skipped in
    BOTH arms, which keeps the comparison paired."""
    env = (t["frontier"].get("env_name") or "frontier-eval-driver")
    return (froot / ".venvs" / env / "bin" / "python").is_file()


def trial_dir(job: str, tid: str, trial: int) -> Path:
    return JOBS / job / tid / f"t{trial}"


def is_done(d: Path) -> bool:
    m = d / "meta.json"
    return m.is_file() and m.stat().st_size > 0


def build_workspace(froot: Path, t: dict, job: str, trial: int) -> tuple[Path, Path]:
    src = froot / "benchmarks" / t["rel"]
    ws = Path(config.WORKSPACE_BASE) / job / f"{t['id']}_t{trial}"
    if ws.exists():
        shutil.rmtree(ws, ignore_errors=True)
    ws.mkdir(parents=True)
    listed = t["frontier"].get("copy_files") or ["."]
    for rel in listed:
        s = src if rel == "." else src / rel
        d = ws if rel == "." else ws / rel
        if s.is_dir():
            shutil.copytree(s, d, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("frontier_eval", "__pycache__"))
        elif s.is_file():
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)
    # frontier_eval/ holds the harness wiring (which venv, which command), not
    # task content. It is never exposed: it would tell the agent exactly how it
    # is about to be invoked.
    shutil.rmtree(ws / "frontier_eval", ignore_errors=True)
    cand = ws / (t["frontier"].get("candidate_destination") or "baseline/init.py")
    return ws, cand


async def run_trial(froot: Path, t: dict, trial: int, job: str,
                    sem: asyncio.Semaphore) -> dict:
    tid = t["id"]
    out_dir = trial_dir(job, tid, trial)
    async with sem:
        if is_done(out_dir):
            return {"id": tid, "trial": trial, "status": "skipped"}
        out_dir.mkdir(parents=True, exist_ok=True)
        ws, cand = build_workspace(froot, t, job, trial)
        task_md = next((ws / n for n in ("Task.md", "README.md")
                        if (ws / n).is_file()), ws / "Task.md")

        user_msg = TASK_TEMPLATE.format(workspace=str(ws), task_md=str(task_md),
                                        candidate_file=str(cand))
        eps = config.POLICY_ENDPOINTS
        h = int(hashlib.md5(tid.encode()).hexdigest()[:8], 16)
        ep = eps[(h + trial) % len(eps)] if eps else {
            "model": config.ORCHESTRATOR_MODEL, "extra_args": {}}
        run_input = AgentRunInput(
            trajectory_id=f"fr-{job}-{tid}-t{trial}",
            initial_messages=[{"role": "system", "content": SYSTEM_PROMPT},
                              {"role": "user", "content": user_msg}],
            mcp_gateway_url=config.GATEWAY_URL,
            mcp_gateway_auth_token=None,
            orchestrator_model=ep["model"],
            orchestrator_extra_args={**config.ORCHESTRATOR_EXTRA_ARGS,
                                     **ep["extra_args"]},
            agent_config_values=dict(config.AGENT_CONFIG_VALUES),
        )
        impl = get_agent_impl("react_toolbelt_agent")
        t0 = time.time()
        status, output = "error", None
        try:
            output = await asyncio.wait_for(impl(run_input),
                                            timeout=config.TIMEOUT_PER_TASK)
            status = str(output.status)
        except asyncio.TimeoutError:
            status = "timeout"
        except Exception as e:  # noqa: BLE001
            print(f"[error] {tid} t{trial}: {e!r}", flush=True)
            traceback.print_exc()
        wall = time.time() - t0

        meta = {"id": tid, "trial": trial, "status": status, "domain": t["domain"],
                "wall_s": round(wall, 1), "model": ep["model"]}
        if output is not None:
            try:
                dump = output.model_dump(mode="json")
                (out_dir / "traj.json").write_text(json.dumps(dump, indent=1, default=str))
                meta["n_messages"] = len(dump.get("messages") or [])
                meta["usage"] = dump.get("usage") or {}
            except Exception as e:  # noqa: BLE001
                meta["traj_error"] = repr(e)
        # The graded artefact is named payload.py for symmetry with the evolve
        # runner, so bench/verify_frontier.py finds it at the same path.
        text = cand.read_text(errors="replace") if cand.is_file() else ""
        (out_dir / "payload.py").write_text(text)
        meta["payload_chars"] = len(text)
        meta["no_payload"] = not text.strip()
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
        shutil.rmtree(ws, ignore_errors=True)
        print(f"[done] {tid} t{trial} status={status} {wall:.0f}s", flush=True)
        return meta


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--frontier-root", required=True)
    ap.add_argument("--manifest", default=os.environ.get("FRONTIER_MANIFEST", "data/frontier_manifest.json"))
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=config.MAX_CONCURRENT)
    ap.add_argument("--tasks", default="")
    a = ap.parse_args()

    froot = Path(a.frontier_root).resolve()
    tasks = load_tasks(Path(a.manifest))
    if a.tasks:
        want = {x.strip() for x in a.tasks.split(",") if x.strip()}
        tasks = [t for t in tasks if t["id"] in want or t["rel"] in want]
    runnable = [t for t in tasks if gradeable(froot, t)]
    skipped = [t["rel"] for t in tasks if t not in runnable]
    print(f"[frun] job={a.job} {len(runnable)} runnable of {len(tasks)} "
          f"(EngDesign domain excluded; {len(skipped)} skipped for a missing "
          f"family venv)", flush=True)
    if skipped:
        print(f"[frun] skipped: {', '.join(sorted(skipped))}", flush=True)
    os.makedirs(config.WORKSPACE_BASE, exist_ok=True)
    (JOBS / a.job).mkdir(parents=True, exist_ok=True)

    sem = asyncio.Semaphore(a.concurrency)
    todo = [(t, k) for t in runnable for k in range(a.n)]
    results = await asyncio.gather(*(run_trial(froot, t, k, a.job, sem)
                                     for t, k in todo))
    by = {}
    for r in results:
        by[r.get("status", "?")] = by.get(r.get("status", "?"), 0) + 1
    print(f"[frun] SUMMARY {json.dumps(by)}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
