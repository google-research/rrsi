#!/usr/bin/env bash
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

# APEX-Agents (OOD for the agentic-workspace instance) with the harness at <ref>.
#   domains/workspace/ood/run_apex.sh <ref> <label> [runner args...]
# Exports the harness into an APEX-Agents runner checkout that vendors the
# archipelago react_toolbelt engine (APEX_REPO; the agent directory inside it is
# APEX_AGENT_DIR, default agents/runner/agents/react_toolbelt_agent as in
# archipelago), then runs that checkout's own driver: APEX_RUN_CMD is executed
# with `bash -c` in APEX_REPO with MODEL_LABEL exported, e.g.
#   APEX_RUN_CMD='python3 runner/run_batch.py --tasks-file ids.json --out-root runs/$MODEL_LABEL'
# Grading is the benchmark's own rubric judge; report pass@1 over the full task set.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AP="${APEX_REPO:?set APEX_REPO to the checkout of the APEX-Agents runner}"
AGENT_DIR="${APEX_AGENT_DIR:-agents/runner/agents/react_toolbelt_agent}"
CMD="${APEX_RUN_CMD:?set APEX_RUN_CMD to the runner command (run in APEX_REPO)}"
REF="${1:?ref}"; LABEL="${2:?label}"; shift 2
"$HERE/export_harness.sh" "$REF" "$AP/$AGENT_DIR"
cd "$AP" && MODEL_LABEL="$LABEL" bash -c "$CMD $*"
