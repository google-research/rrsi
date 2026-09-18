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
# Cheap checks for the setup mistakes that otherwise cost a full evaluation.
# Run it before starting the driver.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
PY="${PYBIN:-${RRSI_AGENT_PYTHON:?set RRSI_AGENT_PYTHON}}"
fail=0
ok(){ printf '  \033[32mOK\033[0m   %s\n' "$1"; }
bad(){ printf '  \033[31mFAIL\033[0m %s\n' "$1"; fail=1; }

echo "== tree =="
[ -z "$(git status --porcelain -- harness)" ] \
  && ok "harness/ tree clean" \
  || bad "harness/ tree dirty -- a leftover candidate would be committed into the incumbent branch"
RUNS="${RRSI_RUNS_DIR:-$REPO/../../runs/eng}"
if [ -f "$RUNS/frontier.json" ]; then
  ft=$(python3 -c "import json;print(json.load(open('$RUNS/frontier.json'))['incumbent'].get('harness_tree',''))")
  ht=$(git rev-parse evolve/eng:third_party/archipelago/harness_eng | cut -c1-12)
  [ "$ft" = "$ht" ] && ok "evolve/eng harness tree matches the frontier ($ht)" \
                    || bad "evolve/eng harness tree $ht != frontier $ft"
else
  echo "  --   no frontier yet (expected before baseline)"
fi

echo "== benchmark =="
n=$(find engdesign_bench/benchmarks -maxdepth 2 -mindepth 2 -type d | wc -l)
[ "$n" -eq 61 ] && ok "61 tasks in the benchmark tree" || bad "found $n tasks, expected 61"
if [ -d EngDesign ]; then
  [ -z "$(git -C EngDesign status --porcelain 2>/dev/null)" ] \
    && ok "official EngDesign checkout unmodified" \
    || bad "the official EngDesign checkout is dirty -- the benchmark tree must stay pristine"
fi

echo "== grading =="
[ -x .venvs/engdesign/bin/python ] && ok "grading venv present" || bad "missing .venvs/engdesign"
for b in iverilog vvp octave ffmpeg bwrap; do
  command -v "$b" >/dev/null && ok "host binary $b" || bad "missing host binary $b (tasks needing it silently score 0)"
done

echo "== containment =="
"$PY" - <<'EOF' >/dev/null 2>&1
import subprocess, sys, os
ws = os.path.join(os.getcwd(), "workspaces"); os.makedirs(ws, exist_ok=True)
argv = ["bwrap"]
for b in ("/usr","/bin","/sbin","/lib","/lib64","/etc","/opt/conda"):
    if os.path.exists(b): argv += ["--ro-bind", b, b]
argv += ["--bind", ws, ws, "--proc","/proc","--dev","/dev","--tmpfs","/tmp",
         "--unshare-all","--die-with-parent","--new-session","--chdir", ws,
         "/bin/sh","-c","cat " + os.path.join(os.getcwd(), "engdesign_bench/benchmarks/*/*/solution.txt")]
r = subprocess.run(argv, capture_output=True, text=True)
# The jail passes only if the answer key is unreachable AND the network is dead.
net = subprocess.run(argv[:-1] + ["python -c \"import socket;s=socket.socket();s.settimeout(2);print(s.connect_ex(('8.8.8.8',53)))\""],
                     capture_output=True, text=True).stdout.strip()
sys.exit(0 if (not r.stdout.strip() and net not in ("0", "")) else 1)
EOF
rc=$?
[ $rc -eq 0 ] && ok "jail: answer key unreachable and network dead" \
              || bad "JAIL BREACH -- the agent could reach the answer key or the network"

echo "== config =="
python3 - "$RUNS" <<'EOF'
import json, sys, os
p = os.path.join(sys.argv[1], "calibration.json")
cfg = json.load(open("rrsi.json"))
if os.path.isfile(p):
    m = json.load(open(p))
    if m.get("k") != cfg["k"]:
        print(f"  \033[31mFAIL\033[0m calibration.json measured at k={m.get('k')} "
              f"but rrsi.json k={cfg['k']}"); sys.exit(1)
    print(f"  \033[32mOK\033[0m   noise band delta={m['delta']:.5f} at k={m['k']} ({m['method']})")
else:
    print("  --   no calibration.json yet (produced by `rrsi.py --domain eng baseline`)")
EOF
[ $? -ne 0 ] && fail=1

echo "== gateway =="
bash scripts/gateway.sh status | grep -q UP && ok "gateway up" || bad "gateway down (scripts/gateway.sh start)"

echo
[ $fail -eq 0 ] && echo "preflight: PASS" || { echo "preflight: FAIL"; exit 1; }
