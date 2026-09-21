# Agentic-workspace instance: Harvey LAB

* Evolve set: 120 Harvey LAB tasks, with a 40-task in-distribution held-out split, `k = 2`. The split is generated deterministically from the checkout by `split_workspace.py` on first use (stratified by practice area; the paper's split was drawn at Harvey LAB commit `1da4750`, pinned in `rrsi.json`).
* Starting harness H_0: the react_toolbelt ReAct agent from archipelago, `third_party/archipelago/harness_workspace/`, run by the vendored archipelago runner with `RRSI_HARNESS_MODULE=harness_workspace.main`.
* Score: fraction of rubric criteria passed over all tasks (20 to 100 independently judged criteria per task); a missing deliverable fails every criterion of its task.
* Out-of-distribution tests: JobBench, GDPval and APEX-Agents through the wrappers in `ood/`.

## Setup

```bash
git clone https://github.com/harveyai/harvey-labs.git
(cd harvey-labs && git checkout 1da4750 && uv sync)              # tasks, judge code and the judge's environment
export HARVEY_LAB_ROOT=$PWD/harvey-labs HARVEY_PY=$HARVEY_LAB_ROOT/.venv/bin/python
export RRSI_AGENT_PYTHON=~/venvs/rrsi-agentic/bin/python         # Python 3.11+ with `pip install -e ".[agentic]"`: the archipelago runner and the MCP gateway
export CODE_EXEC_PYTHON=$(which python3)                          # behind the gateway's code_exec tool; needs python-docx, openpyxl, python-pptx, pymupdf, pandas
export VERTEX_PROJECT="your-project-id"                           # the policy model and the Gemini judge
```

## Run

```bash
python3 rrsi.py --domain workspace smoke
python3 rrsi.py --domain workspace baseline                       # 120 tasks x k=2
python3 rrsi.py --domain workspace run                            # T=20 rounds
python3 rrsi.py --domain workspace heldout --label champ          # the incumbent on the 40 held-out tasks (--ref <commit> for H_0)
```

One evaluation is k production passes (`produce.sh`: an MCP gateway with file tools, `read_pdf` and `code_exec`; `workspace_driver.py` over the task list with 12 concurrent trials) followed by the benchmark's own rubric judge (`workspace_judge.py`). Sample s of job J lands in `runs/workspace/jobs/J/s<s>/<task>/` with `trajectory.json`, `output/` and `scores.json`.

## Out of distribution

JobBench, GDPval and APEX-Agents run the same archipelago engine, so a wrapper exports the harness at a git ref into a checkout of the benchmark's runner and runs that checkout's own driver and judge:

```bash
export JOBBENCH_REPO=/path/to/checkout                                   # vendors the archipelago engine
export JOBBENCH_AGENT_DIR=engine/runner/agents/react_toolbelt_agent     # its react_toolbelt agent directory (the default)
export JOBBENCH_RUN_CMD='python3 run_batch.py --out runs/$MODEL_LABEL'  # its driver, run inside the checkout with MODEL_LABEL set
domains/workspace/ood/run_jobbench.sh evolve/workspace champ
domains/workspace/ood/run_jobbench.sh <H_0 commit> base
```

`GDPVAL_*` with `ood/run_gdpval.sh` and `APEX_*` with `ood/run_apex.sh` follow the same contract (APEX-Agents' default agent directory is `agents/runner/agents/react_toolbelt_agent`). The previous agent directory is kept as a `.bak`, and the wrapper picks a free MCP gateway port unless `GATEWAY_PORT` is set. Report pass@1 over the benchmark's full task set.
