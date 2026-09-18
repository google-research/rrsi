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
"""RRSI: regularized test-time harness evolution.

Domain-agnostic implementation of the two algorithms in the paper
"Regularized Recursive Self-Improvement of Agent Harnesses":

  Algorithm 1 (proposal side)   rrsi.loop.propose_round
  Algorithm 2 (selection side)  rrsi.selection.select_round

Everything benchmark-specific (how a harness is run, scored, rendered and
screened for leakage) lives behind the Domain interface in rrsi.domain and is
implemented under domains/<name>/adapter.py.
"""

__all__ = ["config", "schedule", "history", "components", "select",
           "evaluate", "calibrate", "propose", "critic", "analyst",
           "digester", "loop", "driver", "domain", "gitops", "llm"]
