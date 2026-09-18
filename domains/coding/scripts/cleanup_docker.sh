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
# Clean up after a KILLED harbor evaluation. Force-removing the task containers alone
# leaves their compose networks behind; after ~20 of those docker reports "all
# predefined address pools have been fully subnetted" and EVERY later compose up
# fails, which scores as a total collapse of the next evaluation. Always prune.
set -u
D="$(dirname "$0")/../bin/docker"
n=0; for c in $($D ps -aq --filter 'name=__env'); do $D rm -f "$c" >/dev/null 2>&1 && n=$((n+1)); done
echo "removed $n harbor task containers"
$D network prune -f | tail -1
echo "networks left: $($D network ls -q | wc -l)"
