# Agentic-workspace instance: Harvey LAB

* Evolve set: a fixed 120-task split of Harvey LAB, with a 40-task in-distribution held-out split (`data/split_workspace.json`, regenerated deterministically by `split_workspace.py`), `k = 2`.
* Starting harness H_0: the react_toolbelt ReAct agent from archipelago, `third_party/archipelago/harness_workspace/`, run by the vendored archipelago runner with `RRSI_HARNESS_MODULE=harness_workspace.main`.
* Score: fraction of rubric criteria passed over all tasks (each task carries 20 to 100 independently judged criteria); a missing deliverable fails every criterion of its task.
* Out-of-distribution tests: JobBench, GDPval and APEX-Agents through the wrappers in `ood/`.

## Prerequisites

* A Harvey LAB checkout at `HARVEY_LAB_ROOT` (tasks and the benchmark's judge code) and its judge environment at `HARVEY_PY`.
* `RRSI_AGENT_PYTHON`: a Python (3.11 or newer) with the archipelago runner dependencies (`pip install .[agentic]`).
* `VERTEX_PROJECT` for the policy model and the Gemini judge.

## Commands

```bash
python3 rrsi.py --domain workspace smoke
python3 rrsi.py --domain workspace baseline
python3 rrsi.py --domain workspace run
python3 rrsi.py --domain workspace heldout --label champ          # 40-task held-out split

# OOD: export the harness at a git ref into a benchmark runner checkout that vendors the
# archipelago react_toolbelt engine, then run that checkout's own driver and judge.
#   <BENCH>_REPO      the checkout;  <BENCH>_AGENT_DIR  its react_toolbelt agent directory
#   <BENCH>_RUN_CMD   the driver command, run in the checkout with MODEL_LABEL set
ood/run_jobbench.sh evolve/workspace champ --limit 3
ood/run_gdpval.sh   evolve/workspace champ --limit 3
ood/run_apex.sh     evolve/workspace champ
```

The wrappers pick a free MCP gateway port unless `GATEWAY_PORT` is set: a foreign gateway on a benchmark's default port silently replaces the tool surface.
