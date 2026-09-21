#!/usr/bin/env python3
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
"""Build the Frontier-Eng task manifest read by bench/run_tasks_frontier.py and
bench/verify_frontier.py.

Every task of a Frontier-Engineering checkout
(https://github.com/EinsiaLab/Frontier-Engineering, `benchmarks/<Domain>/<Task>/`)
carries a `frontier_eval/` folder naming its candidate file, its eval command,
its working directory and the files the agent may see. The venv the verifier
must run under is NOT in that folder for most tasks: it is resolved the way the
benchmark's own runner resolves it, from `frontier_eval/conf/batch/v1.yaml`,
then per domain, then `frontier_eval/conf/task/*.yaml`, then the global default
`frontier-eval-driver`. This script does that resolution once and writes it
down, so the runner and the grader share one source of truth for which venv
grades which task.

Usage:
  python3 domains/eng/scripts/build_frontier_manifest.py \
      --frontier-root domains/eng/Frontier-Engineering \
      --out domains/eng/data/frontier_manifest.json [--tasks Domain/Task ...]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

_DEFAULT_ENV = "frontier-eval-driver"   # frontier_eval/conf/task/unified.yaml default


def _read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _read_lines(p: Path) -> list[str]:
    return [ln.strip() for ln in _read(p).splitlines() if ln.strip() and not ln.strip().startswith("#")]


class EnvResolver:
    """Resolve a task's verifier venv (`env_name`) with transparent provenance:
    task env_name.txt > exact v1.yaml label > majority env of the domain in
    v1.yaml > conf/task/<name>.yaml > global default."""

    def __init__(self, froot: Path) -> None:
        self.froot = froot
        self.exact: dict[str, str] = {}
        self.domain_major: dict[str, str] = {}
        self.conf: dict[str, str] = {}
        txt = _read(froot / "frontier_eval" / "conf" / "batch" / "v1.yaml")
        per_domain: dict[str, Counter] = defaultdict(Counter)
        for block in re.split(r"\n- name:", txt):
            lm = re.search(r"label:\s*(\S+)", block)
            if not lm:
                continue
            label = lm.group(1).strip()
            em = (re.search(r"env_name=(\S+)", block)
                  or re.search(r"python_path=uv-env:(\S+)", block))
            env = em.group(1).strip() if em else _DEFAULT_ENV
            if "/" in label:
                self.exact[label] = env
                per_domain[label.split("/")[0]][env] += 1
        for dom, c in per_domain.items():
            self.domain_major[dom] = c.most_common(1)[0][0]
        for y in sorted((froot / "frontier_eval" / "conf" / "task").glob("*.yaml")):
            name, env = None, None
            for ln in _read(y).splitlines():
                m = re.match(r"\s*name:\s*(\S+)", ln)
                if m and name is None:
                    name = m.group(1).strip().strip("\"'")
                m = re.match(r"\s*env_name:\s*(\S+)", ln)
                if m:
                    env = m.group(1).strip().strip("\"'")
            if env:
                for k in {y.stem, name}:
                    if k:
                        self.conf[re.sub(r"[^0-9a-z]+", "", k.lower())] = env

    def resolve(self, rel: Path) -> tuple[str, str]:
        label = rel.as_posix()
        task_env = _read(self.froot / "benchmarks" / rel / "frontier_eval" / "env_name.txt")
        if task_env:
            return task_env, "task:env_name.txt"
        if label in self.exact:
            return self.exact[label], f"v1:{label}"
        dom = rel.parts[0]
        if dom in self.domain_major:
            return self.domain_major[dom], f"v1-domain:{dom}"
        for c in (rel.name, "_".join(rel.parts), dom):
            k = re.sub(r"[^0-9a-z]+", "", c.lower())
            if k in self.conf:
                return self.conf[k], f"conf:{c}"
        return _DEFAULT_ENV, "default"


def discover_tasks(froot: Path) -> list[Path]:
    """Task roots: the parent of each frontier_eval/ folder that names an eval command."""
    return [fe.parent for fe in sorted((froot / "benchmarks").rglob("frontier_eval"))
            if fe.is_dir() and (fe / "eval_command.txt").exists()]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frontier-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tasks", nargs="*", default=None, help="subset by rel path, e.g. KernelEngineering/MLA")
    args = ap.parse_args(argv)

    froot = args.frontier_root.expanduser().resolve()
    if not (froot / "benchmarks").is_dir():
        print(f"error: {froot} has no benchmarks/ dir", file=sys.stderr)
        return 2
    resolver = EnvResolver(froot)
    roots = discover_tasks(froot)
    if args.tasks:
        want = {t.strip("/") for t in args.tasks}
        roots = [r for r in roots if r.relative_to(froot / "benchmarks").as_posix() in want]

    entries = []
    for task_root in roots:
        rel = task_root.relative_to(froot / "benchmarks")
        fe = task_root / "frontier_eval"
        env_name, env_src = resolver.resolve(rel)
        entries.append({
            "task_id": re.sub(r"[^0-9A-Za-z_]+", "_", "frontier__" + "__".join(rel.parts)),
            "domain": rel.parts[0],
            "frontier": {
                "task_rel": rel.as_posix(),
                "candidate_destination": _read(fe / "candidate_destination.txt") or _read(fe / "initial_program.txt"),
                "eval_command": _read(fe / "eval_command.txt"),
                "eval_cwd": _read(fe / "eval_cwd.txt") or ".",
                "env_name": env_name,
                "env_name_source": env_src,
                "copy_files": _read_lines(fe / "copy_files.txt"),
            },
        })
        print(f"[ok] {rel.as_posix():55s} env={env_name} ({env_src})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    envs = Counter(e["frontier"]["env_name"] for e in entries)
    print(f"\nwrote {len(entries)} tasks -> {args.out}; venvs needed under {froot}/.venvs: "
          + ", ".join(f"{k} ({v})" for k, v in envs.most_common()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
