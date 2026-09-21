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
"""Single-task EngDesign verifier adapter.

Dropped into each generated task's `frontier_eval/` by build_engdesign_bench.py
and invoked by `run_eval.sh` with the agent's candidate. It writes `metrics.json`
with a top-level `combined_score` and `valid`, which bench/verify.py reads.

Contract it adapts (upstream EngDesign, per task dir):
  * output_structure.py : pydantic `Response_structure(reasoning, config, ...)`
  * evaluate.py         : `evaluate_llm_response(resp) -> (passed, details, score, score_max_or_conf)`

The agent's candidate is a small python file defining a dict `PAYLOAD` (aliases:
`SUBMISSION`, `submission`) = the `Response_structure` kwargs, e.g.
    PAYLOAD = {"reasoning": "...", "config": {...}}

Scoring (normalized, cross-task-comparable):
  combined_score = score / score_max   (clamped to [0,1])
  valid          = 1.0 if it evaluated without a hard error, else 0.0
score_max is the 4th return value when it looks like a max (numeric, >= score,
> 1), else falls back to a rubric parse, else 100.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import importlib.util
import io
import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any


def _tail(text: str, limit: int = 12000) -> str:
    return text if len(text) <= limit else text[-limit:]


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n",
                    encoding="utf-8")


@contextlib.contextmanager
def _pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _load_module(name: str, path: Path, extra_paths: list[Path]) -> Any:
    original = list(sys.path)
    prev = sys.modules.get(name)
    try:
        sys.path = [str(p) for p in extra_paths] + original
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load spec: {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path = original
        if prev is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = prev


def _load_payload(candidate_path: Path) -> dict[str, Any]:
    """Extract the PAYLOAD dict WITHOUT executing the candidate.

    The candidate is graded in a sandbox that also holds the task's evaluate.py /
    solution.txt, so executing it (runpy/exec) would let a candidate import or
    open those to recover the gold answer (verified leak). We therefore parse the
    file and literal-eval ONLY the assigned value — ast.literal_eval runs no
    imports, calls, or I/O, so it cannot read the grader/solution files.
    Candidates must provide literal values (numbers, strings, lists, dicts).
    """
    src = candidate_path.read_text(encoding="utf-8")
    tree = ast.parse(src)
    keys = ("PAYLOAD", "SUBMISSION", "submission", "RESPONSE", "response")
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id in keys:
                    val = ast.literal_eval(node.value)  # literals only — safe
                    if isinstance(val, dict):
                        return val
    raise ValueError("candidate must define a LITERAL dict named PAYLOAD "
                     "(no imports/calls/file I/O; literal values only)")


def _normalize_payload(section: dict[str, Any]) -> dict[str, Any]:
    """Pass PAYLOAD through as Response_structure kwargs, essentially untouched.

    EngDesign schemas are heterogeneous: some nest the answer under `config`
    (e.g. AB_01's RadiomicsOutput), some are flat (NS_PA_SS_* = reasoning + a
    `code` string), and some carry extra required top-level siblings beyond
    config (XY_01 = config + tetromino_pattern + ...). An earlier version forced
    every payload into a fixed `{reasoning, config}` shape, which dropped the
    siblings and broke the flat schemas outright. We now only guarantee a
    `reasoning` key and hand everything else to the task's own schema, which is
    the single source of truth for field names/nesting.
    """
    if not isinstance(section, dict):
        raise TypeError("PAYLOAD must be a dict of Response_structure kwargs")
    payload = dict(section)
    payload.setdefault("reasoning", "")
    return payload


def _safe_float(v: Any) -> float | None:
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip())
        except Exception:
            return None
    return None


def _rubric_max(benchmark_dir: Path) -> float | None:
    """Best-effort max-points parse from rubrics.txt / rubics.txt."""
    for name in ("rubrics.txt", "rubics.txt", "rubric.txt"):
        p = benchmark_dir / name
        if not p.is_file():
            continue
        nums = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*(?:points|pts|分)", p.read_text("utf-8", "ignore"), re.I)]
        if nums:
            return sum(nums) if len(nums) > 1 else nums[0]
    return None


def _resolve_max(score: float | None, fourth: Any, benchmark_dir: Path) -> float:
    f = _safe_float(fourth)
    if f is not None and f > 1 and (score is None or f >= score):
        return f
    rm = _rubric_max(benchmark_dir)
    if rm and rm > 0:
        return rm
    return 100.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--benchmark-dir", default=".")
    ap.add_argument("--metrics-out", default="metrics.json")
    ap.add_argument("--artifacts-out", default="artifacts.json")
    ap.add_argument("--task-timeout-s", default=180.0, type=float)
    args = ap.parse_args()

    bdir = Path(args.benchmark_dir).expanduser().resolve()
    cand = Path(args.candidate).expanduser()
    cand = cand if cand.is_absolute() else (bdir / cand)
    metrics_out = Path(args.metrics_out)
    metrics_out = metrics_out if metrics_out.is_absolute() else (bdir / metrics_out)
    artifacts_out = Path(args.artifacts_out)
    artifacts_out = artifacts_out if artifacts_out.is_absolute() else (bdir / artifacts_out)

    metrics: dict[str, Any] = {"combined_score": 0.0, "valid": 0.0}
    artifacts: dict[str, Any] = {"candidate_path": str(cand), "benchmark_dir": str(bdir)}

    try:
        payload = _normalize_payload(_load_payload(cand))
        out_mod = _load_module("engdesign_output", bdir / "output_structure.py", [bdir])
        eval_mod = _load_module("engdesign_evaluate", bdir / "evaluate.py", [bdir])
        if not hasattr(out_mod, "Response_structure"):
            raise AttributeError("output_structure.py has no Response_structure")
        if not hasattr(eval_mod, "evaluate_llm_response"):
            raise AttributeError("evaluate.py has no evaluate_llm_response")

        resp = out_mod.Response_structure(**payload)
        sout, serr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(sout), contextlib.redirect_stderr(serr), _pushd(bdir):
            ret = eval_mod.evaluate_llm_response(resp)

        passed, details, score, fourth = (list(ret) + [None] * 4)[:4]
        score_f = _safe_float(score)
        score_max = _resolve_max(score_f, fourth, bdir)
        combined = 0.0 if score_f is None else max(0.0, min(1.0, score_f / score_max))

        metrics = {
            "combined_score": combined,
            "valid": 1.0 if score_f is not None else 0.0,
            "passed": bool(passed),
            "raw_score": score_f,
            "score_max": score_max,
        }
        artifacts.update({
            "details": details, "raw_return": [str(passed), str(score), str(fourth)],
            "eval_stdout": _tail(sout.getvalue()), "eval_stderr": _tail(serr.getvalue()),
        })
    except Exception as exc:  # noqa: BLE001
        metrics = {"combined_score": 0.0, "valid": 0.0, "error": f"{type(exc).__name__}: {exc}"}
        artifacts["traceback"] = _tail(traceback.format_exc())

    _write_json(metrics_out, metrics)
    _write_json(artifacts_out, artifacts)
    print(json.dumps({k: metrics.get(k) for k in ("combined_score", "valid", "passed", "error")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
