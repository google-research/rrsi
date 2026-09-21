# Coding instance: Terminal-Bench 2.1

* Evolve set: all 89 Terminal-Bench 2.1 tasks (`data/tb21_tasks.txt`), `k = 2` trials per task.
* Starting harness H_0: the Terminus-2 agent from harbor, `third_party/harbor_terminus2/` (`harbor_terminus2:AgentHarness`).
* Score: fraction of trials whose hidden unit tests pass; a missing or crashed trial counts as a failure.
* Out-of-distribution test: SWE-bench Verified (500 instances, resolve rate).

## Setup

* Docker. The harness drives each task's container through a tmux session; `bin/docker` wraps `sudo -E docker`, edit it if your user is in the `docker` group.
* harbor: `python3 -m venv domains/coding/.venv && domains/coding/.venv/bin/pip install "harbor>=0.18"` (or point `RRSI_CODING_PYTHON` / `RRSI_CODING_VENV` at an existing environment). harbor pulls `terminal-bench/terminal-bench-2-1` and `swe-bench/swe-bench-verified` from its registry on first use.
* `VERTEXAI_PROJECT` / `VERTEXAI_LOCATION` for the policy model (`policy_model` in `rrsi.json`, a LiteLLM model string).

## Run

```bash
python3 rrsi.py --domain coding smoke        # two tasks, one trial each
python3 rrsi.py --domain coding baseline     # 89 tasks x k=2 = 178 trials
python3 rrsi.py --domain coding run          # T=20 rounds, 2 candidates per round
bash domains/coding/scripts/swe_eval.sh      # OOD: H_0 and the incumbent on SWE-bench Verified, then scripts/swe_summary.py
```

One harbor job per evaluation: `runs/coding/jobs/<job>/<task>__<hash>/` holds the trajectory, the terminal recording and the verifier result. A task whose container never came up is an infrastructure failure, not a harness failure: the round is marked invalid and re-measured once. After killing a run, `scripts/cleanup_docker.sh` removes leftover task containers and compose networks; leftover networks exhaust Docker's address pool and every later evaluation fails.
