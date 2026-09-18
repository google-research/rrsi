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
"""Domain paragraphs for the three search roles on Terminal-Bench 2.1."""

ANALYST = """The agent is a terminal-use agent: a frozen policy LLM ({policy})
driven by a Terminus-2 scaffold solves command-line / coding tasks
(builds, debugging, data processing, security, system administration) inside
a Docker container by emitting batches of keystrokes into a tmux session and
reading the terminal output back. A task counts as solved only if its hidden
unit tests pass after the agent stops. There is one rendered trace per task
(a failing trial where one exists); the VERIFIER section at the end of each
trace shows the per-test pass/fail with assertion messages. Near-miss tasks
that failed a single verifier test are high-leverage. Where a failure digest
hints the agent was blocked by scaffold mechanics (output truncation, fixed
wait durations on long commands, context summarization, JSON parse errors,
completion signalled before running the deliverable) rather than by
judgment, use the capability_gap lens."""

DIGESTER = """The trajectory comes from a terminal-use agent: a frozen policy LLM
solving a command-line / coding task inside a Docker container by emitting
batches of keystrokes into a tmux session and reading the terminal output
back. Each <task_id>.txt holds the task prompt, every step (AGENT
analysis/plan, KEYSTROKES sent, TERMINAL output), run metadata (reward,
episodes, tokens, summarizations, exceptions) and at the end the VERIFIER
ground truth: per-test pass/fail with assertion messages and the test log
tail. Read the VERIFIER section first (what actually failed), then grep for
anchors (step numbers, commands, error strings)."""

PROPOSER = """The benchmark is Terminal-Bench 2.1: 89 containerized terminal tasks
(builds, debugging, data processing, security, sysadmin). The scaffold is a
fork of Terminus-2: the policy sees the terminal state and replies with JSON
containing analysis/plan/commands; each command is a raw keystroke string
sent into a tmux session with a wait duration; the loop feeds the resulting
terminal output back; the run only scores if the agent signals task_complete
(twice, by design) before the wall clock runs out and the task's hidden unit
tests then pass. The frozen policy is {policy}; do not assume it shares your habits
or judgment, read the trajectories for how it actually behaves. The scaffold runs on the HOST, not in the container: it may compute and
transform text freely, but every interaction with the task environment goes
through tmux keystrokes.

Out of bounds, enforced before measurement: reading, detecting or referencing
/tests, test files, reward paths or pytest output at runtime; gaming the
completion signal; benchmark task names or task-specific file names, values
or triggers anywhere in code, prompts or state. The run's real outcome is
judged on SWE-bench Verified, a repository-level bug-fixing benchmark the
search never sees, so memorized task knowledge is worthless. Litmus test for
every edit: would this help a competent human operator working on MANY
unfamiliar terminal tasks?"""

CRITIC = """The harness drives a frozen policy LLM through terminal/coding tasks
(Terminal-Bench 2.1) by emitting keystrokes into a tmux session inside a
Docker container; a task is solved when its hidden unit tests pass. General
terminal craft is fine ("poll long-running builds instead of guessing a fixed
wait" = OK); task knowledge is not ("when extracting an ELF, segments are at
offset 32" = REJECT). Existing safety mechanisms that must not be disabled
without a working replacement: context summarization, terminal-output
truncation, parse-error recovery and the completion double-confirmation."""
