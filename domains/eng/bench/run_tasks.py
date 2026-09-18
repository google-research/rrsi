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
"""Run the react_toolbelt agent over EngDesign tasks, k trials each.

One job = one (task list x n_attempts) sweep into runs/jobs/<job>/:

    runs/jobs/<job>/<task_id>/t<trial>/payload.py   the graded deliverable
    runs/jobs/<job>/<task_id>/t<trial>/traj.json    full trajectory (analyst)
    runs/jobs/<job>/<task_id>/t<trial>/meta.json    status/steps/tokens/wall

Resume-safe: a trial with a non-empty meta.json is skipped, so a killed job can
be re-run and only fills the holes. This is what lets the CONFIRM stage reuse
the SCREEN stage's trials instead of throwing them away.

THE WORKSPACE IS BUILT FROM `copy_files.txt`, NOTHING ELSE. Each task's own
frontier_eval/copy_files.txt lists the files the benchmark intends the solver to
see: the problem statement, the response schema, the payload stub, and images
where a task has them. `evaluate.py`, `solution.txt`, `rubrics.txt` and
`reference.txt` sit next to those in the pristine tree and are never copied. The
gateway additionally jails code_exec inside the workspace, because a shell can
reach anything the path checks do not cover.

The agent scaffold (third_party/archipelago/harness_eng) is imported through the
engine registry via RRSI_HARNESS_MODULE, so the code that runs here is exactly
the code the proposer edits and git diffs.

Usage:
  python3 bench/run_tasks.py --job baseline --n 4
  python3 bench/run_tasks.py --job r3A_screen --ids AB_01,XY_05 --n 1
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
from harness_eng.prompts import (  # noqa: E402
    SYSTEM_PROMPT,
    TASK_TEMPLATE,
)

JOBS = Path(config.RUNS) / "jobs"
BENCH = Path(config.BENCH_TASKS)


# ------------------------------------------------------------------- tasks ---
def all_tasks() -> list[dict]:
    """One record per benchmark task: {id, domain, dir}."""
    out = []
    for td in sorted(BENCH.glob("*/*")):
        if (td / "frontier_eval").is_dir():
            out.append({"id": td.name,
                        "domain": td.parent.name.replace("EngDesignOpen_", ""),
                        "dir": str(td)})
    return out


def select_tasks(set_name: str, ids: str, limit: int) -> list[dict]:
    split_p = Path(config.SPLIT_PATH)
    if split_p.is_file():
        wanted = {t["id"] for t in json.loads(split_p.read_text())[set_name]}
        tasks = [t for t in all_tasks() if t["id"] in wanted]
    else:
        tasks = all_tasks()
    if ids:
        want = {i.strip() for i in ids.split(",") if i.strip()}
        tasks = [t for t in tasks if t["id"] in want]
    return tasks[:limit] if limit else tasks


def trial_dir(job: str, task_id: str, trial: int) -> Path:
    return JOBS / job / task_id / f"t{trial}"


def is_done(d: Path) -> bool:
    m = d / "meta.json"
    return m.is_file() and m.stat().st_size > 0


# --------------------------------------------------------------- workspace ---
def build_workspace(task: dict, job: str, trial: int) -> Path:
    """Materialise the agent-visible slice of a task. Fresh every trial."""
    src = Path(task["dir"])
    ws = Path(config.WORKSPACE_BASE) / job / f"{task['id']}_t{trial}"
    if ws.exists():
        shutil.rmtree(ws, ignore_errors=True)
    ws.mkdir(parents=True)
    listed = (src / "frontier_eval" / "copy_files.txt").read_text().split()
    for rel in listed:
        s = src / rel
        d = ws / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        if s.is_dir():
            shutil.copytree(s, d, dirs_exist_ok=True)
        elif s.is_file():
            shutil.copy2(s, d)
    # The stub must exist even if a task's copy_files did not list it: it is the
    # graded path, and an agent that has nowhere to write scores zero for a
    # reason that has nothing to do with its engineering.
    cand = ws / config.CANDIDATE_REL
    if not cand.exists():
        cand.parent.mkdir(parents=True, exist_ok=True)
        cand.write_text("PAYLOAD = {}\n")
    return ws


def harvest(ws: Path, out_dir: Path) -> tuple[str, bool]:
    """Copy the graded deliverable out of the workspace. Returns (text, wrote)."""
    cand = ws / config.CANDIDATE_REL
    text = ""
    if cand.is_file():
        text = cand.read_text(encoding="utf-8", errors="replace")
    (out_dir / "payload.py").write_text(text)
    # "wrote" means the agent left something other than the untouched stub.
    stub_marker = "TODO: solve the task described in LLM_prompt.txt"
    return text, bool(text.strip()) and stub_marker not in text


# ------------------------------------------------------------------ trials ---
def _total_tokens(usage: dict) -> int | None:
    for key in ("total_tokens", "total_token_count"):
        if isinstance((usage or {}).get(key), int):
            return usage[key]
    tot, found = 0, False
    for v in (usage or {}).values():
        if isinstance(v, dict):
            for key in ("total_tokens", "prompt_tokens", "completion_tokens"):
                if isinstance(v.get(key), int):
                    tot += v[key]
                    found = True
    return tot if found else None


async def run_trial(task: dict, trial: int, job: str, sem: asyncio.Semaphore) -> dict:
    tid = task["id"]
    out_dir = trial_dir(job, tid, trial)
    async with sem:
        if is_done(out_dir):
            return {"id": tid, "trial": trial, "status": "skipped"}
        out_dir.mkdir(parents=True, exist_ok=True)
        ws = build_workspace(task, job, trial)

        user_msg = TASK_TEMPLATE.format(
            workspace=str(ws),
            prompt_file=str(ws / "LLM_prompt.txt"),
            schema_file=str(ws / "output_structure.py"),
            candidate_file=str(ws / config.CANDIDATE_REL),
        )
        # Same frozen model over several Vertex projects: spread trials so
        # throughput is not capped by one project's quota. Deterministic in
        # (task, trial) so resuming a job does not reshuffle routes -- champion
        # and candidate must draw the same route for the same task.
        eps = config.POLICY_ENDPOINTS
        h = int(hashlib.md5(tid.encode()).hexdigest()[:8], 16)
        ep = eps[(h + trial) % len(eps)] if eps else {
            "model": config.ORCHESTRATOR_MODEL, "extra_args": {}}
        run_input = AgentRunInput(
            trajectory_id=f"engd-{job}-{tid}-t{trial}",
            initial_messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
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
            status = "error"
            print(f"[error] {tid} t{trial}: {e!r}", flush=True)
            traceback.print_exc()
        wall = time.time() - t0

        meta = {"id": tid, "trial": trial, "status": status,
                "domain": task["domain"], "wall_s": round(wall, 1),
                "model": ep["model"],
                "vertex_project": ep["extra_args"].get("vertex_project")}
        if output is not None:
            try:
                dump = output.model_dump(mode="json")
                (out_dir / "traj.json").write_text(
                    json.dumps(dump, indent=1, default=str))
                meta["n_messages"] = len(dump.get("messages") or [])
                meta["usage"] = dump.get("usage") or {}
                meta["total_tokens"] = _total_tokens(meta["usage"])
            except Exception as e:  # noqa: BLE001
                meta["traj_error"] = repr(e)

        text, wrote = harvest(ws, out_dir)
        meta["payload_chars"] = len(text)
        meta["no_payload"] = not wrote
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
        shutil.rmtree(ws, ignore_errors=True)
        print(f"[done] {tid} t{trial} status={status} "
              f"payload={len(text)}c{' NONE' if not wrote else ''} "
              f"{wall:.0f}s", flush=True)
        return meta


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    ap.add_argument("--set", default="evolve")
    ap.add_argument("--ids", default="")
    ap.add_argument("--n", type=int, default=config.N_ATTEMPTS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=config.MAX_CONCURRENT)
    args = ap.parse_args()

    tasks = select_tasks(args.set, args.ids, args.limit)
    if not tasks:
        print("[run] no tasks selected")
        sys.exit(2)
    os.makedirs(config.WORKSPACE_BASE, exist_ok=True)
    (JOBS / args.job).mkdir(parents=True, exist_ok=True)

    todo = [(t, k) for t in tasks for k in range(args.n)]
    pending = [x for x in todo if not is_done(trial_dir(args.job, x[0]["id"], x[1]))]
    print(f"[run] job={args.job} tasks={len(tasks)} x{args.n} "
          f"({len(pending)} pending) model={config.ORCHESTRATOR_MODEL} "
          f"gateway={config.GATEWAY_URL} conc={args.concurrency}", flush=True)

    sem = asyncio.Semaphore(args.concurrency)
    results = await asyncio.gather(*(run_trial(t, k, args.job, sem) for t, k in todo))

    by_status: dict[str, int] = {}
    for r in results:
        by_status[r.get("status", "?")] = by_status.get(r.get("status", "?"), 0) + 1
    nopay = sum(1 for r in results if r.get("no_payload"))
    print(f"[run] SUMMARY {json.dumps(by_status)} no_payload={nopay}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
