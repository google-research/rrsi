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
"""The proposer's done() contract, with generate() scripted: no API, no benchmark.

    python3 -m pytest -q tests/        or        python3 tests/test_propose.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rrsi.propose as P                                      # noqa: E402


class _Domain:
    briefs = {"proposer": ""}
    source_exts = {".py"}

    def task_row(self, tid, rec, tr):
        return tid

    def render_trace(self, rec, detail=False):
        return ""


def _edit(eid):
    return {"id": eid, "component": "prompt", "hypothesis": "h", "targets_mode": "m",
            "predicted_affected": ["t1"], "retroactive_check": "r"}


def _propose(replies, harness_dir, repair_brief=None):
    """propose() with generate() replaying `replies`; -> (result, prompts seen)."""
    replies, prompts = list(replies), []

    def fake_generate(prompt, **_kw):
        prompts.append(prompt)
        assert replies, "proposer asked for more turns than scripted"
        return json.dumps(replies.pop(0))

    orig, P.generate = P.generate, fake_generate
    try:
        out = P.propose(_Domain(), harness_dir, {}, [], "", "", 2,
                        {"text": "", "untried": []}, False, [],
                        repair_brief=repair_brief)
    finally:
        P.generate = orig
    return out, prompts


def test_repair_can_redeclare_edits_without_new_file_changes():
    # An undeclared-bundling objection is answered by splitting the declared
    # edits; the code itself is already in the working tree from attempt 0.
    with tempfile.TemporaryDirectory() as td:
        out, _ = _propose([{"action": "done", "summary": "s",
                            "edits": [_edit("C1"), _edit("C2")]}], Path(td),
                          repair_brief={"reasons": ["C1 bundles two mechanisms"]})
        assert out["status"] == "done"
        assert [e["id"] for e in out["edits"]] == ["C1", "C2"]


def test_repair_still_validates_redeclared_edits():
    with tempfile.TemporaryDirectory() as td:
        out, prompts = _propose([
            {"action": "done", "edits": []},
            {"action": "done", "edits": [_edit("C1"), _edit("C2"), _edit("C3")]},
            {"action": "done", "edits": [_edit("C1"), _edit("C2")]},
        ], Path(td), repair_brief={"reasons": ["split C1"]})
        assert "edits array is empty" in prompts[1]
        assert "exceed the budget" in prompts[2]              # b_t = 2
        assert out["status"] == "done" and len(out["edits"]) == 2


def test_fresh_proposal_must_change_files_before_declaring_edits():
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "agent.py").write_text("PROMPT = 'a'\n")
        out, prompts = _propose([
            {"action": "done", "edits": [_edit("C1")]},
            {"action": "edit_file", "path": "agent.py", "old": "'a'", "new": "'b'"},
            {"action": "done", "edits": [_edit("C1")]},
        ], Path(td))
        assert "ZERO file changes" in prompts[1]
        assert out["status"] == "done" and out["n_edits"] == 1
        assert (Path(td) / "agent.py").read_text() == "PROMPT = 'b'\n"


if __name__ == "__main__":
    import inspect
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and inspect.isfunction(fn):
            try:
                fn()
                print(f"ok   {name}")
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f"FAIL {name}: {e!r}")
    sys.exit(1 if fails else 0)
