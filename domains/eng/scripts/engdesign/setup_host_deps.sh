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
# Host-side system dependencies for grading EngDesign tasks.
#
# bench/verify.py scores a candidate by running the task's verifier on the HOST
# (under the grading venv and the host PATH), not inside the agent's sandbox, so
# these binaries must be on the grading host's PATH or the affected tasks
# silently score 0 / invalid. Idempotent: safe to re-run.
#
#   iverilog (+vvp, Icarus Verilog) : the digital-design tasks (NS_PA_SS_*)
#   ffmpeg                          : the path-planning tasks that render a video (AM_02, AM_03)
# Not covered: MATLAB and Webots tasks, which the builder skips.
set -euo pipefail

PKGS=(iverilog ffmpeg)

missing=()
for bin in iverilog vvp ffmpeg; do
  command -v "$bin" >/dev/null 2>&1 || missing+=("$bin")
done

if [[ ${#missing[@]} -eq 0 ]]; then
  echo "[engdesign-host-deps] all present: $(iverilog -V 2>&1 | head -1), ffmpeg ok"
  exit 0
fi

echo "[engdesign-host-deps] missing: ${missing[*]} -> apt-get install ${PKGS[*]}"
SUDO=""
[[ $(id -u) -ne 0 ]] && SUDO="sudo"
$SUDO apt-get update -qq
$SUDO apt-get install -y -q "${PKGS[@]}"

# verify
fail=0
for bin in iverilog vvp ffmpeg; do
  if command -v "$bin" >/dev/null 2>&1; then
    echo "[engdesign-host-deps]   ok: $bin -> $(command -v "$bin")"
  else
    echo "[engdesign-host-deps]  FAIL: $bin still missing" >&2; fail=1
  fi
done
exit $fail
