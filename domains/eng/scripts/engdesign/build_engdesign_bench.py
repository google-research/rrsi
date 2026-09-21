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
"""Build the EngDesign benchmark tree used by the engineering-design instance.

For every task directory of an official EngDesign-Open checkout
(https://github.com/AGI4Engineering/EngDesign, `EngDesign-Open/<task_id>/`) this
writes `<out>/benchmarks/<EngDesignOpen_Domain>/<task_id>/` holding the full
upstream task (so the verifier can load `evaluate.py` and
`output_structure.py`), a seed candidate `submission/payload.py`, and a
`frontier_eval/` folder with the grading wiring that bench/verify.py runs:
`run_eval.sh` calls `evaluate_submission.py`, which parses the payload as a
literal, instantiates the task's `Response_structure` and runs the task's own
`evaluate_llm_response`. `copy_files.txt` restricts what the AGENT sees to the
prompt, the response schema, the payload stub and the images; graders,
solutions and rubrics stay in the tree but are never copied into a workspace.

Tasks that need MATLAB or Webots are skipped (six of the 67 EngDesign-Open
tasks), which leaves the 61 tasks of data/split_engd.json.

Usage:
  python3 domains/eng/scripts/engdesign/build_engdesign_bench.py \
      --engdesign-open /path/to/EngDesign/EngDesign-Open \
      --out domains/eng/engdesign_bench [--tasks AB_01 XY_01] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

_ASSETS = Path(__file__).resolve().parent   # evaluate_submission.py and run_eval.sh live alongside

# never copied into the generated task dir at all
_SKIP_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints"}
# present in the task dir (grader may need none of these) but WITHHELD from the
# agent's container via copy_files — these leak the answer / rubric. Matched by
# exact name or glob; tolerant of upstream's filename variants.
# NB: output_structure.py is NOT withheld — it is the response *schema* (field
# names/types the answer must conform to), leaks nothing about the solution, and
# upstream EngDesign gives it to the model. Withholding it forced the agent to
# guess field names. The grader/solution/rubric files below stay withheld.
_WITHHOLD_GLOBS = (
    "evaluate.py", "run_eval.py",
    "solution.*", "reference.txt", "references.txt",
    "rubric.txt", "rubrics.txt", "rubics.txt",
    "logs", "logs/*", "*.pyc",
)

# Tasks that cannot be graded on a headless host without proprietary software.
# They are recorded in the manifest under "skipped".
_SKIP_TASKS = {
    "RS_01": "needs MATLAB (RunOpenLapSim); all baseline logs score 0",
    "RS_02": "needs MATLAB (RunOpenLapSim); all baseline logs score 0",
    "RS_03": "needs MATLAB (RunOpenLapSim); all baseline logs score 0",
    "Ziheng_02": "MATLAB-typed answers (matlab.double); baseline logs score 0",
    "XZ_03": "needs Webots simulator (subprocess) — not installable headless",
    "XZ_04": "needs Webots simulator (subprocess) — not installable headless",
}

_SEED_PAYLOAD = '''\
"""EngDesign submission for {task_id}.

Define a module-level dict `PAYLOAD` whose keys are the kwargs of the task's
`Response_structure` (see output_structure.py for the exact fields/nesting —
some tasks nest the answer under `config`, others are flat). `PAYLOAD` MUST be a
pure literal (only numbers/strings/lists/dicts/bools/None) — the grader parses
it with ast.literal_eval, so NO imports, function calls, comprehensions, or file
I/O. Only this file is graded.
"""

# EVOLVE-BLOCK-START
PAYLOAD = {{
    "reasoning": "TODO: solve the task described in LLM_prompt.txt",
    # TODO: add the remaining fields required by Response_structure
    # (see output_structure.py — provide literal values only).
}}
# EVOLVE-BLOCK-END
'''


# leading boilerplate labels the upstream files prepend before the real topic,
# e.g. "Main domain topic: Operating system" / "Domain: Computer Architecture"
_LABEL_RE = re.compile(r"^\s*(main\s+)?(domain(\s+topic)?|topic)\s*:\s*", re.I)


def _domain(task_dir: Path, _allow_sibling: bool = True) -> str:
    p = task_dir / "domain_topic.txt"
    first = ""
    if p.is_file():
        for ln in p.read_text("utf-8", "ignore").splitlines():
            if ln.strip():
                first = ln.strip()
                break
    if not first:
        # no domain_topic.txt — borrow from a sibling sharing the author prefix
        # (e.g. JY_02/JY_03 inherit JY_01's "Computer Vision")
        if _allow_sibling and "_" in task_dir.name:
            prefix = task_dir.name.rsplit("_", 1)[0]
            for sib in sorted(task_dir.parent.iterdir()):
                if (sib.is_dir() and sib != task_dir
                        and sib.name.rsplit("_", 1)[0] == prefix
                        and (sib / "domain_topic.txt").is_file()):
                    d = _domain(sib, _allow_sibling=False)
                    if d != "EngDesignOpen_Engdesign":
                        return d
        first = "EngDesign"
    # drop a leading label ("Main domain topic:" etc.) so we keep the value
    first = _LABEL_RE.sub("", first).strip() or first
    # coarse domain = text before the first real separator: ':' '/' '(' ',' or a
    # space-flanked dash. A bare hyphen ("Multi-Agent", "Anti-Aliasing") is kept.
    coarse = re.split(r"[:/(,]|\s-\s|\s-|-\s", first, 1)[0].strip() or first
    slug = re.sub(r"[^0-9A-Za-z]+", "", coarse.title())
    return f"EngDesignOpen_{slug}" if slug else "EngDesignOpen"


def _is_withheld(rel: str) -> bool:
    from fnmatch import fnmatch
    return any(fnmatch(rel, g) or fnmatch(rel.split("/")[0], g) for g in _WITHHOLD_GLOBS)


def _copy_task_tree(src: Path, dst: Path) -> None:
    for p in sorted(src.rglob("*")):
        rel = p.relative_to(src)
        if any(part in _SKIP_DIRS for part in rel.parts):
            continue
        if p.is_dir():
            continue
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, out)


def _agent_visible(src: Path) -> list[str]:
    """Files shipped into the agent's container: the prompt, images, and the
    seed candidate. Everything else (graders, solutions, rubrics) is withheld."""
    out = ["LLM_prompt.txt", "output_structure.py", "submission/payload.py"]
    img = src / "images"
    if img.is_dir():
        out.append("images")
    return out


def build_one(task_dir: Path, out_root: Path, env_name: str, dry: bool) -> dict:
    task_id = task_dir.name
    domain = _domain(task_dir)
    rel = f"{domain}/{task_id}"
    dst = out_root / "benchmarks" / domain / task_id
    info = {"task_id": task_id, "domain": domain, "rel": rel,
            "withheld": [], "agent_files": _agent_visible(task_dir)}

    # what would be withheld (for reporting / the README)
    info["withheld"] = sorted({
        r.relative_to(task_dir).as_posix() for r in task_dir.rglob("*")
        if r.is_file() and _is_withheld(r.relative_to(task_dir).as_posix())
    })

    if dry:
        return info

    if dst.exists():
        shutil.rmtree(dst)
    _copy_task_tree(task_dir, dst)

    # surface the task text as Task.md for tooling that reads Task.md/README.md
    # rather than LLM_prompt.txt
    prompt_txt = (task_dir / "LLM_prompt.txt")
    if prompt_txt.is_file():
        (dst / "Task.md").write_text(prompt_txt.read_text("utf-8", "ignore"), encoding="utf-8")

    # seed candidate
    sub = dst / "submission"
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "payload.py").write_text(_SEED_PAYLOAD.format(task_id=task_id), encoding="utf-8")

    # frontier_eval metadata
    fe = dst / "frontier_eval"
    fe.mkdir(parents=True, exist_ok=True)
    (fe / "candidate_destination.txt").write_text("submission/payload.py\n")
    (fe / "initial_program.txt").write_text("submission/payload.py\n")
    (fe / "eval_command.txt").write_text("bash frontier_eval/run_eval.sh {python} {benchmark} {candidate}\n")
    (fe / "eval_cwd.txt").write_text(".\n")
    (fe / "copy_files.txt").write_text("\n".join(_agent_visible(task_dir)) + "\n")
    (fe / "env_name.txt").write_text(env_name + "\n")
    shutil.copy2(_ASSETS / "evaluate_submission.py", fe / "evaluate_submission.py")
    shutil.copy2(_ASSETS / "run_eval.sh", fe / "run_eval.sh")
    (fe / "run_eval.sh").chmod(0o755)
    return info


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engdesign-open", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--env-name", default="engdesign")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    src_root = args.engdesign_open.expanduser().resolve()
    if not src_root.is_dir():
        print(f"error: {src_root} not a dir", file=sys.stderr)
        return 2
    out_root = args.out.expanduser().resolve()

    task_dirs = [d for d in sorted(src_root.iterdir())
                 if d.is_dir() and not d.name.startswith("__")
                 and (d / "evaluate.py").is_file()]
    if args.tasks:
        want = {t.strip("/") for t in args.tasks}
        task_dirs = [d for d in task_dirs if d.name in want]
    if not task_dirs:
        print("no matching tasks", file=sys.stderr)
        return 2

    skipped = [{"task_id": d.name, "reason": _SKIP_TASKS[d.name]}
               for d in task_dirs if d.name in _SKIP_TASKS]
    task_dirs = [d for d in task_dirs if d.name not in _SKIP_TASKS]
    for s in skipped:
        print(f"[skip] {s['task_id']:43s} {s['reason']}")

    infos = []
    for d in task_dirs:
        info = build_one(d, out_root, args.env_name, args.dry_run)
        infos.append(info)
        tag = "[dry]" if args.dry_run else "[ok]"
        print(f"{tag} {info['rel']:48s} agent_files={info['agent_files']} "
              f"withheld={len(info['withheld'])}")

    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "engdesign_manifest.json").write_text(
            json.dumps({"env_name": args.env_name, "tasks": infos,
                        "skipped": skipped}, indent=2), encoding="utf-8")
        print(f"\nwrote {len(infos)} tasks ({len(skipped)} skipped) -> {out_root}/benchmarks/")
        print(f"manifest -> {out_root}/engdesign_manifest.json")
        print(f"\nNOTE: the verifiers run under the grading venv .venvs/{args.env_name} "
              f"(see domains/eng/README.md).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
