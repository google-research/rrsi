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
# Export the Harvey-LAB harness at a git ref into the react_toolbelt agent dir of
# an external benchmark repo (JobBench, GDPval, APEX-Agents), which all run the
# same vendored engine. The previous agent dir is kept as <dir>.bak.<timestamp>.
#   domains/workspace/ood/export_harness.sh <ref> <target_agent_dir>
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
REF="${1:?git ref (e.g. evolve/workspace, main, a commit)}"; DST="${2:?target react_toolbelt_agent dir}"
[ -d "$DST" ] || { echo "no such dir: $DST"; exit 1; }
cp -a "$DST" "$DST.bak.$(date +%s)"
find "$DST" -mindepth 1 -maxdepth 1 ! -name __pycache__ -exec rm -rf {} +
tmp=$(mktemp -d); git -C "$REPO" archive "$REF" third_party/archipelago/harness_workspace | tar -x -C "$tmp"
cp -a "$tmp/third_party/archipelago/harness_workspace/." "$DST/"; rm -rf "$tmp"
echo "exported third_party/archipelago/harness_workspace@$(git -C "$REPO" rev-parse --short "$REF") -> $DST: $(ls "$DST" | grep -v __pycache__ | tr '\n' ' ')"
