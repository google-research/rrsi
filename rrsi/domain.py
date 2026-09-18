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
"""The Domain interface: everything RRSI needs from a benchmark environment.

RRSI itself never runs an agent, grades a deliverable or reads a trajectory
format. A domain adapter (domains/<name>/adapter.py) supplies those, and the
texts that make the three search roles speak the benchmark's language.

    run / score        Evaluate(H', D, k)              rrsi.evaluate
    load_trial         trajectories for Analyze        rrsi.loop.build_traces
    render_trace       what the analyst/digester/proposer read
    task_row           one-line summary per trace in the task tables
    smoke              liveness check before evaluation (not a selection rule)
    critic_patterns    deterministic leakage denylist   rrsi.critic
    component_signals  diff regexes -> component tag    rrsi.components
    guards             non-compensatory domain checks   rrsi.selection (Sec. 3.3)
    briefs             domain paragraphs for analyst / digester / proposer / critic
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOMAINS = ROOT / "domains"


class Domain:
    name: str = "base"
    harness_path: str = "harness"        # relative to domains/<name>/
    root: Path                            # domains/<name> in the MAIN checkout

    # ---- task sets ---------------------------------------------------------
    def evolve_ids(self) -> list[str]: raise NotImplementedError
    def heldout_ids(self) -> list[str]: return []
    def smoke_ids(self, incumbent_per_task: dict | None = None) -> list[str]:
        raise NotImplementedError

    # ---- Evaluate ----------------------------------------------------------
    def run(self, root: Path, runs_dir: Path, job: str, ids: list[str], k: int,
            log_prefix: str = "") -> None:
        """Run the harness checked out under `root` (a worktree) on `ids`
        with k trials, writing to runs_dir/jobs/<job>. Must be resume-safe."""
        raise NotImplementedError

    def score(self, runs_dir: Path, job: str, ids: list[str], k: int
              ) -> tuple[dict, dict]:
        """-> ({task_id: TaskResult}, extra aggregates)."""
        raise NotImplementedError

    def guards(self, incumbent, candidate) -> list[str]:
        """Violated non-compensatory domain criteria (empty = none)."""
        return []

    def regression_threshold(self, k: int) -> float:
        """Per-task mean drop that counts as a regression in attribution."""
        return 1.0 / max(1, k)

    # ---- evidence ----------------------------------------------------------
    def load_trial(self, runs_dir: Path, job: str, task_id: str, trial: int):
        raise NotImplementedError

    def render_trace(self, rec, detail: bool = False) -> str:
        raise NotImplementedError

    def task_row(self, task_id: str, rec, tr) -> str:
        raise NotImplementedError

    # ---- gates and texts -----------------------------------------------------
    def smoke(self, root: Path, runs_dir: Path, job: str, ids: list[str]
              ) -> tuple[bool, dict]:
        raise NotImplementedError

    critic_patterns: list = []
    component_signals: list = []
    briefs: dict = {}                    # analyst / digester / proposer / critic
    source_exts: set = {".py", ".txt", ".md", ".json"}

    def harness_dir(self, root: Path) -> Path:
        return (root / "domains" / self.name / self.harness_path).resolve()

    def constitution(self, root: Path) -> tuple[str, str]:
        d = root / "domains" / self.name
        return (d / "SKILL.md").read_text(), (d / "PATTERNS.md").read_text()


def load_domain(name: str) -> Domain:
    p = DOMAINS / name / "adapter.py"
    if not p.is_file():
        raise SystemExit(f"unknown domain {name!r} (no {p})")
    spec = importlib.util.spec_from_file_location(f"domains.{name}.adapter", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    dom = mod.DOMAIN
    dom.root = DOMAINS / name
    return dom
