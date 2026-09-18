# Engineering-design instance: EngDesign

* Evolve set: the 61 license-free EngDesign tasks that run without proprietary simulators (`data/split_engd.json`), no in-distribution held-out split, `k = 4`.
* Starting harness H_0: the react_toolbelt agent from archipelago with a bare system prompt and task wrapper, `third_party/archipelago/harness_eng/`, run with `RRSI_HARNESS_MODULE=harness_eng.main`.
* Score: pass rate under each task's own frozen code verifier; the share of valid designs and the share of runs that submit nothing are non-compensatory guards (`Domain.guards`).
* Out-of-distribution tests: EngDesign v1 (the same tasks with grading exploits closed) and Frontier-Eng minus its EngDesign domain, both through `scripts/final_eval.sh`. Frontier-Eng needs a task manifest (`FRONTIER_MANIFEST`, default `data/frontier_manifest.json`): a JSON list with one entry per task holding `task_id`, `domain`, `prompt`, `rubric` and a `frontier` block whose `task_rel` is the task's path inside the Frontier-Engineering checkout; `bench/run_tasks_frontier.py` and `bench/verify_frontier.py` read it.

## Prerequisites

* The benchmark tree under `domains/eng/engdesign_bench/`, built from the official EngDesign checkout; `scripts/build_testsurfaces.sh` builds the two verdict surfaces. The trees are not shipped.
* A grading environment at `domains/eng/.venvs/engdesign/bin/python` (or `GRADING_PYTHON`) with the verifiers' dependencies, plus `iverilog`, `vvp`, `octave`, `ffmpeg` and `bwrap` on the host (`scripts/preflight.sh` checks all of them and probes the code-execution jail).
* `RRSI_AGENT_PYTHON` with the archipelago runner dependencies; `VERTEX_PROJECT` (and optionally `ANTHROPIC_VERTEX_PROJECTS`) for the policy model.

## Commands

```bash
bash scripts/gateway.sh start        # the frozen MCP tool environment (jailed code_exec)
bash scripts/preflight.sh
python3 rrsi.py --domain eng smoke
python3 rrsi.py --domain eng baseline
python3 rrsi.py --domain eng run
bash scripts/final_eval.sh v1        # H_0 and the incumbent on the hardened surface
bash scripts/final_eval.sh frontier  # ... and on Frontier-Eng
```
