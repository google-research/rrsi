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
# JobBench (OOD for the agentic-workspace instance) with the harness at <ref>.
#   domains/workspace/ood/run_jobbench.sh <ref> <label> [runner args...]
# Exports the harness into a JobBench runner checkout that vendors the archipelago
# react_toolbelt engine (JOBBENCH_REPO; agent directory JOBBENCH_AGENT_DIR, default
# engine/runner/agents/react_toolbelt_agent), then runs that checkout's own driver:
# JOBBENCH_RUN_CMD is executed with `bash -c` in JOBBENCH_REPO with MODEL_LABEL and a free
# GATEWAY_PORT exported. Grading uses the benchmark's own judge.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JB="${JOBBENCH_REPO:?set JOBBENCH_REPO to the checkout of the JobBench runner}"
AGENT_DIR="${JOBBENCH_AGENT_DIR:-engine/runner/agents/react_toolbelt_agent}"
CMD="${JOBBENCH_RUN_CMD:?set JOBBENCH_RUN_CMD to the runner command (run in JOBBENCH_REPO)}"
free_port() {  # first free TCP port in 9100-9199 unless GATEWAY_PORT is already set
  [ -n "${GATEWAY_PORT:-}" ] && { echo "$GATEWAY_PORT"; return; }
  for p in $(seq 9100 9199); do
    python3 -c "import socket,sys;s=socket.socket();s.settimeout(0.5);sys.exit(1 if s.connect_ex(('127.0.0.1',$p))==0 else 0)" && { echo "$p"; return; }
  done; echo 9100; }
REF="${1:?ref}"; LABEL="${2:?label}"; shift 2
"$HERE/export_harness.sh" "$REF" "$JB/$AGENT_DIR"
export GATEWAY_PORT="$(free_port)"; echo "[ood] gateway port $GATEWAY_PORT"; cd "$JB" && MODEL_LABEL="$LABEL" bash -c "$CMD $*"
