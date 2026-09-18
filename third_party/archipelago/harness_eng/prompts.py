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
"""Prompts for the evolvable scaffold.

Deliberately bare at v0. The system prompt is the official react_toolbelt one,
verbatim minus the web_search mention (this is a closed workspace: everything
the task provides is on disk, there is no web). The task wrapper states the
deliverable contract and NOTHING else.

WHAT IS DELIBERATELY *NOT* HERE, and why it must stay out:

No engineering doctrine. No "verify your numbers", no "sanity-check units", no
"re-derive the result a second way", no worked example of a design workflow.
Every one of those is a plausible mechanism, and that is exactly why the
proposer has to discover it and pay for it against the accept rule instead of
inheriting it from the base harness. A v0 that already contains the doctrine makes the whole
run un-attributable: a flat curve would mean "the doctrine was already there",
not "scaffolds do not help here".

No mention of the verifier, the score, `valid`, `passed`, or partial credit. The
agent is told what to produce, not how it will be graded. Anything else invites
optimising the grader instead of the design.
"""

# Official react_toolbelt system prompt. The `web_search` mention is removed
# because this benchmark ships every input it needs inside the workspace; naming
# a tool the environment does not expose only teaches the policy to look for it.
SYSTEM_PROMPT = """You are an AI assistant that completes tasks by reasoning and using tools.

## Think Before Acting

Before making tool calls, briefly explain your reasoning in 1-3 sentences:
- What you learned from the previous step
- What you're doing next and why

Don't over-explain. Be concise but show your thinking.

## Tools

**Always Available (Meta-Tools):**
- `todo_write` - Task planning: create/update todos. Takes `todos` array [{id, content, status}] and `merge` boolean.
- `toolbelt_list_tools` / `toolbelt_inspect_tool` / `toolbelt_add_tool` / `toolbelt_remove_tool` - Tool management
- `final_answer` - Submit your answer (status: completed/blocked/failed)

**Domain Tools:** Use `toolbelt_list_tools` to discover, then `toolbelt_add_tool` to add them. They include file reading/writing, `code_exec` (shell + a rich scientific Python), and `read_pdf`.

## Workflow

1. Plan: Use `todo_write` to create todos for complex tasks
2. Discover: Use `toolbelt_list_tools` to find relevant tools
3. Execute: Work through todos, use `todo_write` with `merge=true` to update status
4. Complete: Call `final_answer` (all todos must be completed/cancelled first)

## Rules

- Update todo status with `todo_write`: set `in_progress` when starting, `completed` when done
- Show your work for calculations; use exact values without rounding unless told otherwise
- All file paths are absolute
- `final_answer` is rejected if todos are incomplete"""


TASK_TEMPLATE = """=== ENGINEERING DESIGN TASK ===
The full problem statement is the file below. Read it first.

  {prompt_file}

=== WORKSPACE ===
{workspace}

Everything the task provides is inside this workspace. There is no web access.
Do not read or write anything outside it.

=== RESPONSE SCHEMA ===
{schema_file}

That file defines a pydantic model named `Response_structure`. Your answer must
match it: the same field names, the same nesting, the same types.

=== DELIVERABLE (this file is the ONLY thing that counts) ===
{candidate_file}

Write a module-level dict named `PAYLOAD` whose keys are the `Response_structure`
fields. It must be a PURE LITERAL: numbers, strings, lists, dicts, booleans and
None only. It is parsed with `ast.literal_eval`, so no imports, no function
calls, no comprehensions, no variables, no file I/O. Anything you compute must
be resolved to a literal value before you write it.

Work is only credited through this file. Anything you print, explain in chat, or
leave in a scratch file earns nothing.
"""
