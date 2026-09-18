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
"""Domain paragraphs for the three search roles on EngDesign."""

_CLASSES = """Every non-pass is one of four things, they need different mechanisms,
and a mechanism aimed at the wrong class will measure as noise:
  NO SUBMISSION: the run ended without writing a payload. Termination, step
    budget or plumbing; never an engineering problem, never fixed by advice.
  MALFORMED SUBMISSION: a payload was written but the verifier could not use
    it (not a pure literal, a missing or misnamed schema field, a wrong type,
    a numpy scalar where a float was needed). The design may have been RIGHT
    and was thrown away at the last step; the check is mechanical and the
    schema is in the workspace.
  INVALID DESIGN: it parsed but violated the task's hard constraints.
  UNDER TARGET: a valid design that missed the performance gate. The only
    class where "the agent needed to compute better" is the right reading;
    ask first whether the agent ever evaluated its own candidate before
    submitting.
The verifier's `details` block in every trace names the sub-checks that were
met and missed; read it before theorising."""

ANALYST = f"""The agent is a ReAct toolbelt scaffold with file tools and a
sandboxed scientific `code_exec`, driving a frozen policy LLM (Claude Opus
4.8) through ENGINEERING DESIGN problems (control loops, digital hardware,
structural sizing, computer architecture, image processing, OS scheduling,
vehicle dynamics, ...). Each answer is a literal `PAYLOAD` dict written to one
file and graded by the task's OWN frozen code verifier (a simulation, a
testbench, a numeric pipeline); no LLM grades anything, and the verifier
returns a continuous combined_score, a binary passed, and a valid flag.

THE FIRST THING TO READ ON ANY FAILURE IS ITS CLASS. {_CLASSES}

One cross-cutting pattern worth naming: the agent computes a design, never
runs any check of it, and submits anyway, in an environment that handed it a
full scientific Python. Treat "submitted without ever evaluating" as its own
family and count how often it happens. Weight a MALFORMED or NO-SUBMISSION
loss above its raw score cost: those throw away work already done and are
certainly attributable to the scaffold."""

DIGESTER = """The trajectory comes from a ReAct toolbelt agent solving an
engineering design problem with a sandboxed scientific Python and writing a
literal `PAYLOAD` dict. Each <task_id>.txt holds the system/task prompt, every
step (tool calls, tool results), the PAYLOAD that was submitted, and a GRADING
block carrying the failure class, the three verifier axes (combined_score /
passed / valid) and the verifier's own per-sub-check `details`. The `details`
block is the highest-value thing in the file: grep for it first ("VERIFIER
DETAILS", "FAILURE CLASS"), anchor your digest on it, then work backwards
through the trajectory to find where that specific quantity was computed or
lost."""

PROPOSER = f"""The benchmark is EngDesign (61 license-free tasks): per task the
agent gets a written problem statement, a pydantic response schema and a
sandboxed workspace with a scientific Python (numpy, scipy, sympy, control,
cvxopt, opencv, scikit-*) plus iverilog, vvp, octave and ffmpeg, and must write
its design into ONE file as a literal `PAYLOAD` dict. The scaffold is the
official ReAct toolbelt agent: the policy plans, manages a toolbelt of tools
(file read/write, read_pdf, code_exec), keeps a todo list and submits with
`final_answer`. The frozen policy is Claude Opus 4.8, a strong model: failures
are rarely "the model is dumb" and usually "the harness starves, misroutes, or
fails to make it check its own work". There is NO web access and NO network
inside the code_exec jail: a mechanism that installs a package or downloads
anything hangs and then fails the whole trial.

Each submission is graded by the task's OWN frozen code verifier, which
returns a continuous combined_score, a binary passed and a valid flag. The
score the search optimizes is the pass rate; a candidate is additionally
rejected if it lowers the share of valid designs or raises the share of runs
that submit nothing.

{_CLASSES}

Out of bounds, enforced before measurement: the agent is jailed to a workspace
holding the problem statement, the schema and the payload stub, while each
task's `evaluate.py`, `solution.txt`, `rubrics.txt` and reference outputs sit
in a tree it cannot reach. Any mechanism that reaches for them, walks the
filesystem, or hard-codes a threshold, tolerance or reference value that
could only have come from a grader is rejected. The run's real outcome is
judged on a HARDENED version of these same tasks with nine grading exploits
closed, and on Frontier-Eng, a different engineering benchmark's verifiers, so
a scaffold that games a verifier or encodes this suite's disciplines and
toolchain scores here and collapses there. Litmus test for every edit: would
it still make sense, and still help, on an unfamiliar engineering problem in
a discipline this benchmark does not contain, graded by a different verifier?"""

CRITIC = """The harness drives a frozen policy LLM (Claude Opus 4.8) through
ENGINEERING DESIGN tasks: it reads a problem statement, computes a design with
a sandboxed scientific Python, and writes the answer as a literal `PAYLOAD`
dict graded by the task's OWN frozen code verifier. VERIFIER ESCAPE is the
most serious failure and the one to check first: the pristine benchmark tree
holds each task's `evaluate.py`, `solution.txt`, `rubrics.txt` and reference
outputs right next to the prompt and is deliberately out of reach. REJECT
anything that reads or searches outside the workspace, uses absolute or
upward paths, walks the filesystem, probes for grading code, or hard-codes a
pass threshold, tolerance, reference value or test input that could only have
come from a grader. A list of disciplines or task families is ALSO leakage: it
encodes the benchmark's distribution rather than a general procedure. General
engineering practice transfers and is fine: validating the payload against
the schema the task itself provides, evaluating a candidate design before
committing to it, handling a repeated tool error, protecting computed values
from context compression, ensuring a best-effort answer exists before the
budget runs out. Existing safety mechanisms that must not be disabled without
a working replacement: ReSum context compression (resum.py), tool-result
truncation (tool_result.py), the todo-gated finalize guard, error handling.
A trial that never writes a payload scores ZERO, so also reject open-ended
"keep improving" loops with no give-up path."""
