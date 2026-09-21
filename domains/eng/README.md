# Engineering-design instance: EngDesign

* Evolve set: the 61 EngDesign-Open tasks that run without proprietary simulators (`data/split_engd.json`), no in-distribution held-out split, `k = 4`.
* Starting harness H_0: the react_toolbelt agent from archipelago with a bare system prompt and task wrapper, `third_party/archipelago/harness_eng/`, run with `RRSI_HARNESS_MODULE=harness_eng.main`.
* Score: pass rate under each task's own frozen code verifier; the share of valid designs and the share of runs that submit nothing are non-compensatory guards (`Domain.guards`).
* Out-of-distribution tests: EngDesign v1 (a hardened variant of the same tasks) and Frontier-Eng minus its EngDesign domain, both through `scripts/final_eval.sh`.

## Setup

Paths below are relative to `domains/eng/`.

```bash
git clone https://github.com/AGI4Engineering/EngDesign.git                                                   # the official tasks
python3 scripts/engdesign/build_engdesign_bench.py --engdesign-open EngDesign/EngDesign-Open --out engdesign_bench
python3 -m venv .venvs/engdesign && .venvs/engdesign/bin/pip install -r scripts/engdesign/requirements.txt   # the verifiers' dependencies (GRADING_PYTHON)
bash scripts/engdesign/setup_host_deps.sh                                                                    # iverilog, vvp, ffmpeg
sudo apt-get install -y bubblewrap octave                                                                    # the code_exec jail; octave for the tasks that call it
export RRSI_AGENT_PYTHON=~/venvs/rrsi-agentic/bin/python VERTEX_PROJECT="your-project-id"
```

The builder writes `engdesign_bench/benchmarks/<domain>/<task>/`: the full upstream task next to a `frontier_eval/` wrapper that `bench/verify.py` runs. Only the prompt, the response schema, the payload stub and the images are ever copied into an agent workspace, and `code_exec` runs under bwrap with that workspace as the only writable mount and no network.

## Run

```bash
GATEWAY_PORT=8996 bash scripts/gateway.sh start   # the frozen MCP tool gateway: file tools, read_pdf, jailed code_exec
bash scripts/preflight.sh                          # tree, venvs, host tools, jail escape probes
python3 rrsi.py --domain eng smoke
python3 rrsi.py --domain eng baseline              # 61 tasks x k=4 = 244 trials
python3 rrsi.py --domain eng run                   # T=40 rounds
```

Each trial writes `runs/eng/jobs/<job>/<task>/t<k>/payload.py` (the graded deliverable, a literal `PAYLOAD` dict), `traj.json`, `meta.json` and, after `bench/verify.py`, `verdict.json` with `combined_score`, `passed` and `valid`.

## Out of distribution

```bash
git clone https://github.com/EinsiaLab/Frontier-Engineering.git
(cd Frontier-Engineering && bash init.sh && bash scripts/env/setup_v1_task_envs.sh)         # driver venv and the per-family task venvs
python3 scripts/build_frontier_manifest.py --frontier-root Frontier-Engineering --out data/frontier_manifest.json
bash scripts/final_eval.sh frontier          # H_0 and the incumbent from their own worktrees, compared within task
bash scripts/final_eval.sh v1                # EngDesign v1, once scripts/build_testsurfaces.sh has built engdesign_bench_v1/
```
