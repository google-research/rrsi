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
"""Domain paragraphs for the three search roles on Harvey LAB."""

ANALYST = """The agent performs professional LEGAL work (contract review and
drafting, M&A diligence, IP, governance, funds, litigation support; 25
practice areas) on the Harvey LAB benchmark: a frozen policy LLM (Claude Opus
4.8) driven by a ReAct toolbelt scaffold reads a folder of source documents
(.docx/.xlsx/.pdf) and must write DELIVERABLE FILES under exact requested
filenames. Grading is a strict per-criterion rubric: each task carries 20-100
independent pass/fail criteria judged against the deliverable files, and the
score is the fraction of criteria passed. A criterion fails if the required
content is missing from the DELIVERABLE FILE; work done in chat or scratch
files earns nothing, so "computed but never written into the deliverable" is
its own failure family. Each rendered trace ends with the per-criterion
verdicts and the judge's rationale; rank modes by criteria cost."""

DIGESTER = """The trajectory comes from a ReAct toolbelt agent doing legal
document work: it reads source documents through file tools, code_exec and
read_pdf, and writes deliverable files. Each <task_id>.txt holds the system
and task prompt, every step (tool calls, tool results), the final output, and
at the end the RUBRIC GRADING block: per-criterion PASS/FAIL with the judge's
rationale. Read the RUBRIC block first (which criteria failed and why), then
grep for anchors (step numbers, tool names, file names, key values) to find
where the content was computed, lost, or never written."""

PROPOSER = """The benchmark is Harvey LAB: legal tasks that give the agent a
folder of source documents (.docx/.xlsx/.pdf) and instructions; it must
produce deliverable files with exact requested filenames (memos, markups, term
sheets, workbooks) in an output directory. There is NO web access. The
scaffold is the official ReAct toolbelt agent: the policy plans, manages a
toolbelt of gateway tools (file read/write, read_pdf, code_exec with a rich
scientific Python for producing .docx/.xlsx/.pptx), keeps a todo list, and
submits with `final_answer` (gated on completed todos). Deliverables are what
gets graded; the chat answer is not scored. The frozen policy is Claude Opus
4.8, a strong model: failures are rarely "the model is dumb" and usually "the
harness starves, floods, or misroutes it".

Grading is a STRICT PER-CRITERION rubric (20-100 criteria per task). Passing
MORE criteria on tasks the agent already attempts well (coverage,
completeness, exact-filename delivery, correct numbers carried through) is
worth as much as rescuing disasters. A criterion fails if the required content
is missing from the DELIVERABLE FILE, so routing computed results into the
output files is a first-class concern.

Out of bounds, enforced before measurement: reading or format-gaming the
rubric judge, touching task.json, and any task-specific content (task names,
party or company names, file names of specific tasks, numeric answers). The
run's real outcome is judged on a pristine held-out LAB split and on three
other agentic-workspace benchmarks with different tool surfaces (JobBench,
GDPval, APEX-Agents), so prefer mechanisms a competent lawyer's workflow
would justify in ANY environment (read what the task actually asks; verify
numbers carried into deliverables; cover every requested item) over
mechanisms keyed to this benchmark's incidental quirks (its tool names, folder
layout, judge phrasing). Litmus test: would this help a competent lawyer
producing deliverables on MANY unfamiliar legal tasks, in any reasonable
working environment?"""

CRITIC = """The harness drives a frozen policy LLM (Claude Opus 4.8) through
legal document tasks (Harvey LAB): it reads source documents and writes
deliverable files graded by a per-criterion rubric judge. This evolution line
is law-domain-only, so LAW-GENERAL professional method ("read the controlling
authority before stating a holding", "cross-check computed numbers before
finalizing") is legitimate and NOT leakage; the line is crossed at
TASK-specific facts, names, or answers ("the acquirer is Vantage Partners" =
REJECT). Existing safety mechanisms that must not be disabled without a
working replacement: ReSum context compression (resum.py), tool-result
truncation (tool_result.py), and the todo-gated final_answer acceptance."""
