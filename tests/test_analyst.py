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
"""Offline coverage for analysis of flat and benchmark-relative task IDs."""

import json
from types import SimpleNamespace

import pytest

from rrsi import analyst, digester


@pytest.mark.parametrize("task_id", ["flat", "contracts/task", "contracts/subarea/task"])
def test_prepare_rendered_preserves_task_paths_and_cached_traces(tmp_path, task_id):
    domain = SimpleNamespace(render_trace=lambda rec, detail: rec["text"])
    rendered = tmp_path / "rendered"

    analyst.prepare_rendered(domain, {task_id: {"text": "original trace"}}, rendered)
    assert (rendered / f"{task_id}.txt").read_text() == "original trace"

    analyst.prepare_rendered(domain, {task_id: {"text": "replacement"}}, rendered)
    assert (rendered / f"{task_id}.txt").read_text() == "original trace"


def test_analyze_roundtrips_nested_digests_without_basename_collisions(tmp_path, monkeypatch):
    traces = {
        "flat": {"text": "flat evidence"},
        "contracts/task": {"text": "contract evidence"},
        "litigation/task": {"text": "litigation evidence"},
        "contracts/subarea/task": {"text": "subarea evidence"},
    }
    domain = SimpleNamespace(
        render_trace=lambda rec, detail: rec["text"],
        task_row=lambda tid, rec, result: tid,
        briefs={"analyst": "Offline analyst", "digester": "Offline digester"},
    )
    actions = iter([
        json.dumps({"action": "digest_many", "requests": [
            {"task_id": tid, "lens": "failure"} for tid in traces
        ]}),
        json.dumps({"action": "report"}),
    ])
    monkeypatch.setattr(analyst, "generate", lambda *args, **kwargs: next(actions))

    def generate_digest(prompt, **kwargs):
        # Drive the real digester's read_file action before returning its evidence.
        task_id = prompt.splitlines()[0].removeprefix("Assigned trace: ").removesuffix(".txt")
        if "[result] " not in prompt:
            return json.dumps({"action": "read_file", "path": f"{task_id}.txt"})
        return json.dumps({"action": "return", "digest": {
            "evidence": prompt.split("[result] ", 1)[1],
        }})

    monkeypatch.setattr(digester, "generate", generate_digest)
    report = analyst.analyze(domain, traces, {}, tmp_path)
    loaded = analyst.load_digests(tmp_path)

    assert report["n_digests"] == 4
    assert len(loaded) == 4
    assert {d["task_id"]: d["evidence"] for d in loaded} == {
        "flat": "1: flat evidence",
        "contracts/task": "1: contract evidence",
        "litigation/task": "1: litigation evidence",
        "contracts/subarea/task": "1: subarea evidence",
    }
    assert all(d["lens"] == "failure" for d in loaded)


def test_load_digests_includes_existing_nested_files(tmp_path):
    saved = tmp_path / "analysis" / "digests"
    (saved / "area" / "subarea").mkdir(parents=True)
    (saved / "flat_failure.json").write_text(json.dumps({"task_id": "flat"}))
    (saved / "area" / "subarea" / "task_success.json").write_text(
        json.dumps({"task_id": "area/subarea/task", "lens": "success"})
    )

    assert analyst.load_digests(tmp_path) == [
        {"task_id": "area/subarea/task", "lens": "success"},
        {"task_id": "flat"},
    ]


def test_load_digests_without_analysis_returns_empty(tmp_path):
    assert analyst.load_digests(tmp_path) == []
