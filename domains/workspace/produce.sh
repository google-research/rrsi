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
# One production pass over a task list with the CURRENT engine tree.
# Usage: ./produce.sh <label> <tasks_json> [extra driver args...]
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$REPO"
VENV="${VENV:-${RRSI_AGENT_PYTHON:?set RRSI_AGENT_PYTHON (python with the archipelago runner deps)}}"
label="${1:?label}"; tasks="${2:?tasks json}"; shift 2

export VERTEX_PROJECT="${VERTEX_PROJECT:?set VERTEX_PROJECT (GCP project for Vertex AI)}"
export VERTEX_LOCATION="${VERTEX_LOCATION:-global}"
export VERTEXAI_PROJECT="$VERTEX_PROJECT"
export VERTEXAI_LOCATION="$VERTEX_LOCATION"
export GATEWAY_PORT="${GATEWAY_PORT:-8992}"
export CODE_EXEC_PYTHON="${CODE_EXEC_PYTHON:-/opt/conda/bin/python3}"

"$VENV" -m compileall -q "$REPO/../../third_party/archipelago/harness_workspace" || { echo "compile fail"; exit 1; }
mkdir -p "${RRSI_LOGS:-logs}"
GW_LOG="${RRSI_LOGS:-logs}/gateway_$(echo "$label" | tr / _).log"
setsid "$VENV" "$REPO/mcp_gateway/gateway.py" >"$GW_LOG" 2>&1 &
GW_PID=$!
trap 'kill -- -$GW_PID 2>/dev/null' EXIT
for i in $(seq 1 30); do
  "$VENV" -c "import socket,sys; s=socket.socket(); s.settimeout(1); sys.exit(0 if s.connect_ex(('127.0.0.1',$GATEWAY_PORT))==0 else 1)" && break
  sleep 1
done

MODEL_LABEL="$label" ORCHESTRATOR_MODEL="${ORCHESTRATOR_MODEL:-vertex_ai/claude-opus-4-8}" \
MAX_CONCURRENT="${MAX_CONCURRENT:-12}" \
  "$VENV" "$REPO/workspace_driver.py" --subset "$tasks" "$@"
