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
# Start / stop / check the frozen tool gateway.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PY="${AGENT_PYTHON:-${RRSI_AGENT_PYTHON:?set RRSI_AGENT_PYTHON}}"
PORT="${GATEWAY_PORT:-8994}"
case "${1:-start}" in
  start)
    if "$PY" -c "import socket,sys;s=socket.socket();s.settimeout(1);sys.exit(0 if s.connect_ex(('127.0.0.1',$PORT))==0 else 1)"; then
      echo "[gateway] already up on $PORT"; exit 0; fi
    mkdir -p "${RRSI_LOGS:-logs}" "${WORKSPACE_BASE:-$REPO/workspaces}"
    GATEWAY_PORT="$PORT" WORKSPACE_BASE="${WORKSPACE_BASE:-$REPO/workspaces}" \
      setsid "$PY" "$REPO/mcp_gateway/gateway.py" >> "${RRSI_LOGS:-logs}/gateway.log" 2>&1 &
    for i in $(seq 1 40); do
      "$PY" -c "import socket,sys;s=socket.socket();s.settimeout(1);sys.exit(0 if s.connect_ex(('127.0.0.1',$PORT))==0 else 1)" && { echo "[gateway] up on $PORT"; exit 0; }
      sleep 1
    done
    echo "[gateway] FAILED to come up; see logs/gateway.log"; exit 1;;
  stop)   pkill -f "mcp_gateway/gateway.py" && echo "[gateway] stopped";;
  status) "$PY" -c "import socket,sys;s=socket.socket();s.settimeout(1);print('UP' if s.connect_ex(('127.0.0.1',$PORT))==0 else 'DOWN')";;
esac
