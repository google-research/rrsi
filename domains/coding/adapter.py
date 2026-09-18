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
"""Coding instance: Terminal-Bench 2.1 through harbor with a Terminus-2 fork.

Evaluate = `scripts/run_eval.sh <job> <dataset> <k> <concurrency> [-i task ...]`
run from the candidate's worktree (PYTHONPATH = that worktree's third_party/,
so `--agent harbor_terminus2:AgentHarness` resolves to the candidate's harness). One
harbor job = one evaluation; trial directories are `<task>__<hash>` under
runs/coding/jobs/<job>/. A trial's reward is 1.0 iff the task's hidden tests
pass; a missing or errored trial counts 0.0 with the full denominator.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))
from rrsi.domain import Domain          # noqa: E402
from rrsi.evaluate import TaskResult    # noqa: E402
import briefs                            # noqa: E402
import render                            # noqa: E402

CFG = json.loads((HERE / "rrsi.json").read_text())
PYBIN = os.environ.get("RRSI_CODING_PYTHON", str(HERE / ".venv" / "bin" / "python"))


def _load_result(trial_dir: Path) -> dict | None:
    rj = trial_dir / "result.json"
    if not rj.exists():
        return None
    try:
        return json.loads(rj.read_text())
    except json.JSONDecodeError:
        return None


def _short(result: dict) -> str:
    return str(result.get("task_name", "")).split("/")[-1]


def _reward(result: dict) -> float | None:
    val = ((result.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    return float(val) if val is not None else None


_INFRA_EXC = re.compile(r"docker|compose|image|pull|registry|no space left|connection refused|"
                        r"daemon|network .* (not found|already exists)", re.I)


def infra_failure(result: dict) -> bool:
    """True when the trial never got a working environment (harbor/docker error),
    as opposed to the agent timing out or the tests failing."""
    e = result.get("exception_info")
    if not e:
        return False
    if isinstance(e, dict):
        typ, msg = str(e.get("exception_type", "")), str(e.get("exception_message", ""))
    else:
        typ, msg = "", str(e)
    if typ in ("AgentTimeoutError", "VerifierTimeoutError"):
        return False
    return bool(_INFRA_EXC.search(msg)) or typ in ("RuntimeError", "OSError")


def _tokens(result: dict) -> int | None:
    if result.get("exception_info"):
        return None
    ar = result.get("agent_result") or {}
    tok = (ar.get("n_input_tokens") or 0) + (ar.get("n_output_tokens") or 0)
    return tok or None


class CodingDomain(Domain):
    name = "coding"
    harness_path = "../../third_party/harbor_terminus2"
    source_exts = {".py", ".txt", ".md", ".json", ".sh"}
    briefs = {k: v.replace("{policy}", CFG.get("policy_label", "the frozen policy"))
              for k, v in {"analyst": briefs.ANALYST, "digester": briefs.DIGESTER,
                           "proposer": briefs.PROPOSER, "critic": briefs.CRITIC}.items()}
    critic_patterns = [
        (r"/tests\b", "verifier tests path in diff"),
        (r"test_outputs", "verifier test file reference in diff"),
        (r"\breward\.(txt|json)\b", "reward file reference in diff"),
        (r"\bctrf\b", "verifier report reference in diff"),
        (r"/solution\b", "oracle solution path in diff"),
        (r"task\.toml", "task config reference in diff"),
        (r"/logs/verifier", "verifier logs path in diff"),
    ]
    component_signals = [
        ("memory",          [r"TBMH_STATE_DIR"]),
        ("subagent",        [r"generate_response"]),
        ("context_mgmt",    [r"_summarize", r"summariz", r"proactive_summarization",
                             r"unwind", r"free_tokens"]),
        ("output_plumbing", [r"_limit_output_length", r"terminal_output", r"truncat",
                             r"_collapse", r"paste.?buffer", r"load-?buffer"]),
        ("control_flow",    [r"_run_agent_loop", r"episode", r"is_task_complete",
                             r"_pending_completion", r"keystrokes"]),
        ("config",          [r"= *\d+ *#", r"threshold", r"max_bytes", r"duration"]),
        ("prompt",          [r"templates/.*\.txt", r"_VERIFICATION", r"_get_.*hint",
                             r"prompt", r"system"]),
    ]

    def __init__(self):
        self._tasks = [x.strip() for x in (HERE / "data" / "tb21_tasks.txt")
                       .read_text().splitlines() if x.strip()]
        self.critic_patterns = list(self.critic_patterns) + [
            (rf"(?<![\w-]){re.escape(n)}(?![\w-])", f"benchmark task name '{n}' in diff")
            for n in self._tasks]

    # ---- task sets ---------------------------------------------------------
    def evolve_ids(self) -> list[str]:
        return list(self._tasks)

    def heldout_ids(self) -> list[str]:
        return []          # SWE-bench Verified is run as a separate harbor dataset

    def smoke_ids(self, incumbent_per_task=None) -> list[str]:
        return list(CFG["smoke_tasks"])

    # ---- Evaluate ----------------------------------------------------------
    def _harbor(self, root: Path, runs_dir: Path, job: str, ids: list[str] | None,
                k: int, dataset: str, log_prefix: str) -> None:
        jobs = runs_dir / "jobs"
        jobs.mkdir(parents=True, exist_ok=True)
        jdir = jobs / job
        if jdir.exists():
            n_have = sum(1 for rj in jdir.glob("*/result.json")
                         if not infra_failure(_load_result(rj.parent) or {}))
            n_want = len(ids if ids is not None else self._tasks) * k
            if n_have >= n_want:
                print(f"[coding] job {job} already complete ({n_have} trials); not re-running",
                      flush=True)
                return
            # harbor cannot resume a job; a partial dir (killed run, or an earlier
            # candidate under the same name) would be scored as mostly-missing.
            stale = jdir.with_name(f"{job}.stale.{int(time.time())}")
            jdir.rename(stale)
            print(f"[coding] job {job} was incomplete ({n_have}/{n_want} trials); moved to "
                  f"{stale.name} and re-running from scratch", flush=True)
        script = root / "domains" / "coding" / "scripts" / "run_eval.sh"
        cmd = [str(script), job, dataset, str(k), str(CFG.get("concurrency", 10)),
               "--jobs-dir", str(jobs)]
        if ids is not None and set(ids) != set(self._tasks):
            for t in ids:
                cmd += ["-i", f"terminal-bench/{t}"]
        env = {**os.environ, "RRSI_CODING_ROOT": str(root / "domains" / "coding"),
               "RRSI_CODING_VENV": str((HERE / ".venv").resolve()),
               "MODEL": CFG.get("policy_model", "vertex_ai/gemini-3.5-flash")}
        log = runs_dir / "logs" / f"{log_prefix or job}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a") as lf:
            r = subprocess.run(cmd, cwd=str(root / "domains" / "coding"),
                               stdout=lf, stderr=subprocess.STDOUT, env=env)
        if r.returncode != 0:
            print(f"[coding] WARNING harbor rc={r.returncode} (see {log})", flush=True)

    def run(self, root, runs_dir, job, ids, k, log_prefix=""):
        self._harbor(Path(root), Path(runs_dir), job, ids, k, CFG["dataset"], log_prefix)

    def _trials(self, runs_dir: Path, job: str) -> dict[str, list[Path]]:
        by_task: dict[str, list[Path]] = {}
        jdir = Path(runs_dir) / "jobs" / job
        if not jdir.exists():
            return by_task
        for td in sorted(jdir.iterdir()):
            if not td.is_dir():
                continue
            res = _load_result(td)
            if res is None:
                continue
            by_task.setdefault(_short(res), []).append(td)
        return by_task

    def score(self, runs_dir, job, ids, k):
        trials = self._trials(Path(runs_dir), job)
        per = {}
        total_pass = 0
        for t in ids:
            dirs = trials.get(t, [])[:k]
            rewards, toks = [], []
            for td in dirs:
                res = _load_result(td) or {}
                if infra_failure(res):
                    continue                  # counted as missing below
                rw = _reward(res)
                rewards.append(rw if rw is not None else 0.0)
                toks.append(_tokens(res))
            missing = k - len(rewards)
            rewards += [0.0] * missing
            toks += [None] * missing
            passes = sum(1 for x in rewards if x == 1.0)
            total_pass += passes
            per[t] = TaskResult(rewards=rewards, tokens=toks, missing=missing,
                                extra={"passes": passes,
                                       "trial_dirs": [str(d) for d in dirs]})
        return per, {"total_passes": total_pass, "n_trials": len(ids) * k,
                     "pass_rate": total_pass / max(1, len(ids) * k)}

    # ---- evidence ----------------------------------------------------------
    def load_trial(self, runs_dir, job, task_id, trial):
        dirs = self._trials(Path(runs_dir), job).get(task_id, [])
        if trial >= len(dirs):
            return None
        res = _load_result(dirs[trial]) or {}
        meta = (res.get("agent_result") or {}).get("metadata") or {}
        return {"task_id": task_id, "trial_dir": str(dirs[trial]),
                "reward": _reward(res), "episodes": meta.get("n_episodes"),
                "exception": (str(res.get("exception_info"))[:200]
                              if res.get("exception_info") else None)}

    def render_trace(self, rec, detail=False):
        return render.render_full(rec, detail=detail)

    def task_row(self, task_id, rec, tr):
        return (f"{task_id} | rewards={tr.rewards if tr else '?'} | "
                f"rendered_trial_reward={rec.get('reward')} | "
                f"episodes={rec.get('episodes')} | exception={str(rec.get('exception'))[:60]}")

    # ---- gates -------------------------------------------------------------
    def smoke(self, root, runs_dir, job, ids):
        root, runs_dir = Path(root), Path(runs_dir)
        cdir = root / "domains" / "coding"
        comp = subprocess.run([PYBIN, "-m", "compileall", "-q", str(self.harness_dir(root))],
                              capture_output=True, text=True)
        if comp.returncode != 0:
            return False, {"stage": "compile", "detail": (comp.stdout + comp.stderr)[-1500:]}
        code = ("from pathlib import Path\nfrom harbor_terminus2 import AgentHarness\n"
                "a = AgentHarness(logs_dir=Path('/tmp/rrsi_ctor_check'), "
                f"model_name='{CFG.get('policy_model', 'vertex_ai/gemini-3.5-flash')}')\n"
                "assert AgentHarness.name() == 'rrsi-terminus2'\nprint('CTOR_OK')\n")
        ctor = subprocess.run([PYBIN, "-c", code], cwd=str(cdir), capture_output=True,
                              text=True, env={**os.environ, "PYTHONPATH": str(root / "third_party")})
        if "CTOR_OK" not in ctor.stdout:
            return False, {"stage": "ctor", "detail": (ctor.stderr or ctor.stdout)[-1200:]}
        jdir = runs_dir / "jobs" / job
        subprocess.run(["rm", "-rf", str(jdir)])
        self._harbor(root, runs_dir, job, ids, 1, CFG["dataset"], job)
        found, details = 0, {}
        for td in (jdir.glob("*") if jdir.exists() else []):
            res = _load_result(td)
            if res is None:
                continue
            found += 1
            if res.get("exception_info"):
                details[td.name] = str(res["exception_info"])[:400]
        if found < len(ids):
            details["missing_trials"] = f"{found}/{len(ids)}"
        if details:
            return False, {"stage": "smoke_run", **details}
        return True, {"stage": "smoke_run", "n": found}


DOMAIN = CodingDomain()
