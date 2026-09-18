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
"""Engineering-design instance: EngDesign-Open (61 tasks) through the vendored
official react_toolbelt engine and each task's frozen code verifier.

Evaluate = bench/run_tasks.py (k trials per task, resume-safe) from the
candidate's worktree, then bench/verify.py (frozen verifiers, deterministic).
Trial i of task X in job J is runs/eng/jobs/J/X/t<i>/{payload.py, traj.json,
meta.json, verdict.json}. The reward is the verifier's binary `passed` (the
paper's headline metric); combined_score, valid and no_payload are kept as
extras, and the last two feed the two domain guards (Sec. 3.3: a candidate
must satisfy several non-compensatory criteria).
"""

from __future__ import annotations

import json
import os
import socket
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
SPLIT = json.loads((HERE / "data" / "split_engd.json").read_text())
AGENT_PY = os.environ.get("RRSI_AGENT_PYTHON", CFG.get("agent_python") or sys.executable)
BENCH_ROOT = Path(os.environ.get("BENCH_ROOT", str(HERE / "engdesign_bench")))
GRADING_PY = Path(os.environ.get("GRADING_PYTHON",
                                 str((HERE / ".venvs").resolve() / "engdesign" / "bin" / "python")))
GATEWAY_PORT = int(os.environ.get("GATEWAY_PORT", str(CFG.get("gateway_port", 8996))))


def _port_up(port: int) -> bool:
    s = socket.socket()
    s.settimeout(1)
    try:
        return s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        s.close()


class EngDomain(Domain):
    name = "eng"
    harness_path = "../../third_party/archipelago/harness_eng"
    briefs = {"analyst": briefs.ANALYST, "digester": briefs.DIGESTER,
              "proposer": briefs.PROPOSER, "critic": briefs.CRITIC}
    critic_patterns = [
        (r"evaluate\.py|solution\.txt|solution\.sv|rubrics?\.txt|rubics\.txt|"
         r"reference\.txt|metrics\.json|artifacts\.json|frontier_eval|run_eval\.sh",
         "names grading / reference-solution artefacts"),
        (r"engdesign_bench|benchmarks/EngDesignOpen|\.venvs/engdesign",
         "reaches for the pristine benchmark tree"),
        (r"os\.environ\[[\"\']HOME|/home/|\.\./\.\./\.\.|os\.chdir\(\s*[\"\']/",
         "absolute or upward path escape out of the agent workspace"),
        (r"pass_threshold|score_max|combined_score|\bcombined score\b",
         "encodes the verifier's scoring internals into the scaffold"),
        (r"EngDesign|eng_design|engdesign", "benchmark name hardcoded in scaffold"),
        (r"\b(?:AB|AM|AV|CY|DL|HC|HJ|JY|KV|LX|NS_PA_SS|RK|WJ|XG|XW|XY|XZ|YJ|YX|"
         r"Yiqi|Yuqi|ZC|ZH|libin2)_\d\d\b", "benchmark task id hardcoded in scaffold"),
        (r"iverilog|vvp\b|octave|ffmpeg|cvxopt|\bcontrol\.matlab\b",
         "keys the scaffold to the toolchain of some tasks in THIS suite"),
        (r"pip[\"'\],\s]+install|apt-get|conda[\"'\],\s]+install|urllib|"
         r"requests\.(?:get|post)|curl\s+-|wget\s|socket\.socket|http://|https://",
         "network or package install from inside the jail"),
    ]
    component_signals = [
        ("memory",          [r"DRMH_STATE_DIR", r"digest\("]),
        ("context_mgmt",    [r"resum\.py", r"ReSum", r"summariz", r"compact",
                             r"_should_compress", r"keep_last"]),
        ("output_plumbing", [r"tool_result\.py", r"truncat", r"HEAD_CHARS", r"TAIL_CHARS",
                             r"MAX_RESULT_TOKENS", r"_process_result"]),
        ("control_flow",    [r"_handle_tool_calls", r"def step", r"final_answer",
                             r"has_incomplete_todos", r"max_steps", r"_execute_mcp_tool"]),
        ("config",          [r"= *\d+ *#", r"threshold", r"_CAP\b", r"_LIMIT\b"]),
        ("prompt",          [r"prompts\.py", r"SYSTEM_PROMPT", r"TASK_TEMPLATE", r"prompt",
                             r"description"]),
    ]

    def __init__(self):
        self._tasks = {t["id"]: t for t in SPLIT["evolve"]}

    # ---- task sets ---------------------------------------------------------
    def evolve_ids(self) -> list[str]:
        return [t["id"] for t in SPLIT["evolve"]]

    def heldout_ids(self) -> list[str]:
        return []          # EngDesign v1 / Frontier-Eng are run by scripts/final_eval.sh

    def smoke_ids(self, incumbent_per_task=None) -> list[str]:
        n = CFG.get("smoke_n", 4)
        ids = self.evolve_ids()
        if incumbent_per_task:
            solid = [i for i in ids if i in incumbent_per_task
                     and min(incumbent_per_task[i].rewards or [0]) >= 1.0]
            if len(solid) >= n:
                return solid[:n]
        return ids[:n]

    # ---- Evaluate ----------------------------------------------------------
    def _env(self, runs_dir: Path) -> dict:
        return {**os.environ, "RRSI_RUNS_DIR": str(runs_dir),
                "RRSI_SPLIT_PATH": str(HERE / "data" / "split_engd.json"),
                "BENCH_ROOT": str(BENCH_ROOT), "GRADING_PYTHON": str(GRADING_PY),
                "RRSI_HARNESS_MODULE": "harness_eng.main",
                "WORKSPACE_BASE": str(runs_dir / "workspaces"),
                "GATEWAY_PORT": str(GATEWAY_PORT), "AGENT_PYTHON": AGENT_PY}

    def _ensure_gateway(self, root: Path, runs_dir: Path) -> None:
        if _port_up(GATEWAY_PORT):
            return
        (runs_dir / "workspaces").mkdir(parents=True, exist_ok=True)
        subprocess.run(["bash", str(HERE / "scripts" / "gateway.sh"), "start"],
                       env=self._env(runs_dir), cwd=str(HERE))
        for _ in range(40):
            if _port_up(GATEWAY_PORT):
                return
            time.sleep(1)
        raise RuntimeError("engdesign gateway failed to start (see logs/gateway.log)")

    def _run_and_verify(self, root: Path, runs_dir: Path, job: str, ids: list[str],
                        k: int, log_name: str) -> None:
        self._ensure_gateway(root, runs_dir)
        eng = root / "domains" / "eng"
        log = runs_dir / "logs" / f"{log_name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        env = self._env(runs_dir)
        cmd = [AGENT_PY, str(eng / "bench" / "run_tasks.py"), "--job", job, "--n", str(k),
               "--ids", ",".join(ids)]
        with open(log, "a") as lf:
            r = subprocess.run(cmd, cwd=str(eng), env=env, stdout=lf, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            print(f"[eng] WARNING run rc={r.returncode} (see {log})", flush=True)
        with open(log, "a") as lf:
            v = subprocess.run([sys.executable, str(eng / "bench" / "verify.py"), "--job", job],
                               cwd=str(eng), env=env, stdout=lf, stderr=subprocess.STDOUT)
        if v.returncode != 0:
            raise RuntimeError(f"verify failed for job={job} (rc={v.returncode}); see {log}")

    def run(self, root, runs_dir, job, ids, k, log_prefix=""):
        self._run_and_verify(Path(root), Path(runs_dir), job, ids, k, log_prefix or job)

    def _records(self, runs_dir: Path, job: str, tid: str) -> list[dict]:
        out = []
        tdir = runs_dir / "jobs" / job / tid
        if not tdir.is_dir():
            return out
        for trial in sorted(tdir.glob("t*")):
            if not (trial / "meta.json").is_file() or not (trial / "verdict.json").is_file():
                continue
            try:
                v = json.loads((trial / "verdict.json").read_text())
                meta = json.loads((trial / "meta.json").read_text())
            except Exception:  # noqa: BLE001
                continue
            out.append({"score": float(v.get("combined_score") or 0.0),
                        "passed": bool(v.get("passed")), "valid": float(v.get("valid") or 0.0),
                        "no_payload": bool(meta.get("no_payload")),
                        "tokens": meta.get("total_tokens")})
        return out

    def score(self, runs_dir, job, ids, k):
        runs_dir = Path(runs_dir)
        per = {}
        passes = valid = nopay = 0
        scores = []
        for t in ids:
            recs = self._records(runs_dir, job, t)[:k]
            rewards = [1.0 if r["passed"] else 0.0 for r in recs]
            toks = [r["tokens"] if isinstance(r["tokens"], int) else None for r in recs]
            missing = k - len(recs)
            rewards += [0.0] * missing
            toks += [None] * missing
            cs = [r["score"] for r in recs] + [0.0] * missing
            vs = [r["valid"] for r in recs] + [0.0] * missing
            npay = [1.0 if r["no_payload"] else 0.0 for r in recs] + [1.0] * missing
            passes += int(sum(rewards))
            valid += sum(vs)
            nopay += sum(npay)
            scores += cs
            per[t] = TaskResult(rewards=rewards, tokens=toks, missing=missing,
                                extra={"combined_scores": cs, "valid": vs, "no_payload": npay})
        n = max(1, len(ids) * k)
        return per, {"total_passes": passes, "n_trials": len(ids) * k, "pass_rate": passes / n,
                     "mean_combined_score": sum(scores) / n, "valid_rate": valid / n,
                     "no_payload_rate": nopay / n}

    def guards(self, incumbent, candidate) -> list[str]:
        out = []
        d_valid = (incumbent.extra.get("valid_rate") or 0.0) - (candidate.extra.get("valid_rate") or 0.0)
        if d_valid > CFG["max_valid_rate_drop"]:
            out.append(f"valid rate fell {d_valid:+.3f} (limit {CFG['max_valid_rate_drop']})")
        d_np = (candidate.extra.get("no_payload_rate") or 0.0) - (incumbent.extra.get("no_payload_rate") or 0.0)
        if d_np > CFG["max_no_payload_rise"]:
            out.append(f"no-submission rate rose {d_np:+.3f} (limit {CFG['max_no_payload_rise']})")
        return out

    # ---- evidence ----------------------------------------------------------
    def load_trial(self, runs_dir, job, task_id, trial):
        task = self._tasks.get(task_id) or {"id": task_id, "domain": "?"}
        return render.load_trial(Path(runs_dir) / "jobs" / job, task, trial)

    def render_trace(self, rec, detail=False):
        return render.render_full(rec, detail=detail)

    def task_row(self, task_id, rec, tr):
        meta, verdict, diag = rec.get("meta") or {}, rec.get("verdict") or {}, rec.get("diag") or {}
        task = rec.get("task") or {}
        msgs = (rec.get("traj") or {}).get("messages") or []
        err = str(verdict.get("error") or "")
        if diag.get("no_payload"):
            kind = "NO-SUBMISSION(nothing was written)"
        elif (verdict.get("status") or "ok") != "ok":
            kind = f"INFRA({verdict.get('status')}) -- not evidence about the design"
        elif verdict.get("passed"):
            kind = "passed"
        elif "literal" in err.lower() or "validation" in err.lower() or "PAYLOAD" in err:
            kind = "MALFORMED(design discarded at the last step)"
        elif float(verdict.get("valid") or 0.0) < 1.0:
            kind = "INVALID-DESIGN(violated a hard constraint)"
        else:
            kind = "UNDER-TARGET(valid but missed the gate)"
        flag = " NEVER-RAN-ANY-CODE" if (not verdict.get("passed")
                                          and (diag.get("n_code_exec") or 0) == 0) else ""
        return (f"{task_id} | {task.get('domain')} | {kind}{flag} | pass_rate={tr.mean:.2f} "
                f"score={float(verdict.get('combined_score') or 0.0):.3f} "
                f"valid={verdict.get('valid')} | status={meta.get('status')} | "
                f"steps={sum(1 for m in msgs if m.get('role') == 'assistant')} | "
                f"code_exec={diag.get('n_code_exec')}({diag.get('n_code_exec_failed')} err) "
                f"payload_writes={diag.get('n_payload_writes')}")

    # ---- gates -------------------------------------------------------------
    def smoke(self, root, runs_dir, job, ids):
        root, runs_dir = Path(root), Path(runs_dir)
        eng = root / "domains" / "eng"
        comp = subprocess.run([AGENT_PY, "-m", "compileall", "-q", str(self.harness_dir(root))],
                              capture_output=True, text=True)
        if comp.returncode != 0:
            return False, {"stage": "compile", "err": (comp.stdout + comp.stderr)[-1500:]}
        code = (
            "import sys, os; sys.path.insert(0, '../../third_party/archipelago'); os.environ['RRSI_HARNESS_MODULE'] = 'harness_eng.main'\n"
            "from runner.agents.registry import get_agent_impl\n"
            "from harness_eng.prompts import SYSTEM_PROMPT, TASK_TEMPLATE\n"
            "impl = get_agent_impl('react_toolbelt_agent')\n"
            "assert callable(impl), 'agent impl not callable'\n"
            "assert SYSTEM_PROMPT.strip(), 'empty system prompt'\n"
            "body = TASK_TEMPLATE.format(workspace='SENTINEL-WS', prompt_file='SENTINEL-PROMPT',\n"
            "                            schema_file='SENTINEL-SCHEMA', candidate_file='SENTINEL-CAND')\n"
            "for tok in ('SENTINEL-WS', 'SENTINEL-PROMPT', 'SENTINEL-SCHEMA', 'SENTINEL-CAND'):\n"
            "    assert tok in body, f'task template drops {tok}'\n"
            "assert 'PAYLOAD' in body, 'deliverable contract lost from task template'\n"
            "print('CTOR OK')\n")
        ctor = subprocess.run([AGENT_PY, "-c", code], cwd=str(eng), capture_output=True, text=True)
        if ctor.returncode != 0 or "CTOR OK" not in ctor.stdout:
            return False, {"stage": "ctor", "err": (ctor.stderr or ctor.stdout)[-1500:]}
        subprocess.run(["rm", "-rf", str(runs_dir / "jobs" / job)])
        self._run_and_verify(root, runs_dir, job, ids, 1, job)
        per, extra = self.score(runs_dir, job, ids, 1)
        missing = sum(tr.missing for tr in per.values())
        ok = missing == 0 and extra["no_payload_rate"] == 0.0 and extra["mean_combined_score"] > 0.0
        return ok, {"stage": "run", "smoke_ids": ids, "missing": missing, **extra}


DOMAIN = EngDomain()
