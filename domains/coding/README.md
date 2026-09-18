# Coding instance: Terminal-Bench 2.1

* Evolve set: all 89 Terminal-Bench 2.1 tasks (`data/tb21_tasks.txt`), `k = 2` trials per task.
* Starting harness H_0: the Terminus-2 agent from harbor, `third_party/harbor_terminus2/` (imported as `harbor_terminus2:AgentHarness`).
* Score: fraction of trials whose hidden unit tests pass; a missing or crashed trial counts as a failure.
* Out-of-distribution test: SWE-bench Verified (500 instances, resolve rate).

## Prerequisites

* Docker (the harness talks to task containers through `bin/docker`, a `sudo -E docker` shim; adjust it to your setup).
* A virtual environment with harbor at `domains/coding/.venv` (or set `RRSI_CODING_VENV`); the Terminal-Bench 2.1 and SWE-bench Verified datasets are pulled by harbor.
* `VERTEXAI_PROJECT` / `VERTEXAI_LOCATION` for the policy model (`policy_model` in `rrsi.json`, a LiteLLM model string).

## Commands

```bash
python3 rrsi.py --domain coding smoke        # compile, construct, run two tasks
python3 rrsi.py --domain coding baseline     # Evaluate(H_0), calibrate delta
python3 rrsi.py --domain coding run          # rounds 0..T-1

# OOD: both arms on SWE-bench Verified, then compare
scripts/run_eval.sh swe_base swe-bench/swe-bench-verified 1 8 --jobs-dir runs/coding/jobs   # from a worktree of the harness under test
python3 scripts/swe_summary.py runs/coding/jobs/swe_base runs/coding/jobs/swe_best
```

After killing a harbor run, `scripts/cleanup_docker.sh` removes the task containers and their compose networks; leftover networks exhaust Docker's address pool and every later evaluation fails.
