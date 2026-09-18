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
"""FastMCP streamable-http gateway: the tool ENVIRONMENT for the EngDesign
meta-harness. Frozen — this is the benchmark's world, not the agent's scaffold,
and the proposer never edits it.

Surface:
  - filesystem reads (read_text_file, list_files, get_directory_tree, search_files)
  - write_file
  - read_pdf (pymupdf)
  - code_exec (shell + the scientific python stack, run inside a bubblewrap jail)

No web_search: EngDesign ships every input it needs inside the workspace.

TWO CONTAINMENT RULES, both load-bearing:

1. Every path argument must resolve inside WORKSPACE_BASE. The agent's workspace
   holds only the files the task's own `copy_files.txt` exposes (prompt, response
   schema, images, the payload stub). The pristine benchmark tree — which holds
   `evaluate.py`, `solution.txt`, `rubrics.txt` and `reference.txt` right next to
   the prompt — lives outside it and is never reachable.

2. `code_exec` is a SHELL, so a path check on its arguments is worth nothing: the
   command itself can cat, grep or glob anywhere the process can reach. It
   therefore runs under `bwrap` with the workspace as the only writable mount,
   the system runtime read-only, and no network at all. Without this the whole
   evolution is unsound — a scaffold that learns to grep the filesystem for
   `solution.txt` would climb the metric while learning nothing, and would be
   invisible in the score.

Stateless with respect to task, so one gateway safely serves all concurrent
trials.
"""

import os
import shutil
import subprocess
from pathlib import Path

from fastmcp import FastMCP

GATEWAY_HOST = os.environ.get("GATEWAY_HOST", "127.0.0.1")
GATEWAY_PORT = int(os.environ.get("GATEWAY_PORT", "8994"))
CODE_EXEC_PYTHON = os.environ.get("CODE_EXEC_PYTHON", "/opt/conda/bin/python3")
CODE_EXEC_TIMEOUT = int(os.environ.get("CODE_EXEC_TIMEOUT", "600"))
WORKSPACE_BASE = os.path.realpath(
    os.environ.get("WORKSPACE_BASE",
                   os.path.join(os.path.dirname(os.path.dirname(
                       os.path.abspath(__file__))), "workspaces")))
SANDBOX = os.environ.get("ENGD_SANDBOX", "1") not in ("0", "false", "")

MAX_READ_CHARS = 200_000

mcp = FastMCP(
    name="engdesign-gateway",
    instructions=(
        "Tools for engineering design tasks: read the problem statement and "
        "response schema, run shell + scientific python to compute and check a "
        "design, and write the submission file."
    ),
)


# ---------------------------------------------------------------------------- #
# Containment
# ---------------------------------------------------------------------------- #
def _inside(path: str) -> Path | None:
    """Resolve `path` and return it only if it lands inside WORKSPACE_BASE."""
    try:
        p = Path(os.path.realpath(path))
    except Exception:  # noqa: BLE001
        return None
    try:
        p.relative_to(WORKSPACE_BASE)
    except ValueError:
        return None
    return p


def _denied(path: str) -> str:
    return (f"Error: path is outside the workspace and cannot be accessed: "
            f"{path}. Everything this task provides is inside your workspace.")


# ---------------------------------------------------------------------------- #
# Filesystem
# ---------------------------------------------------------------------------- #
@mcp.tool
def read_text_file(path: str) -> str:
    """Read a UTF-8 text file and return its contents. `path` is an absolute
    path inside your workspace. Use for .txt/.md/.json/.csv/.py and other
    plaintext; for PDFs use read_pdf, for binary formats use code_exec."""
    p = _inside(path)
    if p is None:
        return _denied(path)
    if not p.is_file():
        return f"Error: not a file: {path}"
    data = p.read_text(encoding="utf-8", errors="replace")
    if len(data) > MAX_READ_CHARS:
        return data[:MAX_READ_CHARS] + f"\n[truncated at {MAX_READ_CHARS} chars]"
    return data


@mcp.tool
def list_files(path: str = ".") -> str:
    """List the entries of a directory inside your workspace, one per line,
    with a trailing '/' on directories and a byte size on files."""
    p = _inside(path)
    if p is None:
        return _denied(path)
    if not p.is_dir():
        return f"Error: not a directory: {path}"
    out = []
    for e in sorted(p.iterdir()):
        out.append(f"{e.name}/" if e.is_dir() else f"{e.name}  {e.stat().st_size}")
    return "\n".join(out) or "(empty)"


@mcp.tool
def get_directory_tree(path: str = ".", max_entries: int = 500) -> str:
    """Recursively list a directory inside your workspace as an indented tree,
    stopping after `max_entries` entries."""
    root = _inside(path)
    if root is None:
        return _denied(path)
    if not root.is_dir():
        return f"Error: not a directory: {path}"
    lines, n = [], 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in sorted(dirnames) if d not in ("__pycache__", ".git")]
        depth = len(Path(dirpath).relative_to(root).parts)
        lines.append("  " * depth + Path(dirpath).name + "/")
        for f in sorted(filenames):
            n += 1
            if n > max_entries:
                lines.append("  " * (depth + 1) + f"[truncated at {max_entries} entries]")
                return "\n".join(lines)
            lines.append("  " * (depth + 1) + f)
    return "\n".join(lines)


@mcp.tool
def search_files(path: str, name_contains: str) -> str:
    """Find files under a workspace directory whose NAME contains a substring."""
    root = _inside(path)
    if root is None:
        return _denied(path)
    if not root.is_dir():
        return f"Error: not a directory: {path}"
    hits = [str(p) for p in root.rglob("*")
            if p.is_file() and name_contains.lower() in p.name.lower()]
    return "\n".join(sorted(hits)[:500]) or "(no match)"


@mcp.tool
def write_file(path: str, content: str) -> str:
    """Write UTF-8 text to an absolute path inside your workspace, creating
    parent directories as needed. Overwrites an existing file."""
    p = _inside(path)
    if p is None:
        return _denied(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        return f"Error: {e}"
    return f"Wrote {len(content)} chars to {p}"


@mcp.tool
def read_pdf(path: str, max_pages: int = 100) -> str:
    """Extract text from a PDF inside your workspace, page by page."""
    p = _inside(path)
    if p is None:
        return _denied(path)
    if not p.is_file():
        return f"Error: not a file: {path}"
    try:
        import fitz  # pymupdf
    except Exception as e:  # noqa: BLE001
        return f"Error: pymupdf unavailable ({e})"
    try:
        doc = fitz.open(str(p))
    except Exception as e:  # noqa: BLE001
        return f"Error: cannot open pdf: {e}"
    parts = []
    for i, page in enumerate(doc):
        if i >= max_pages:
            parts.append(f"[stopped after {max_pages} pages]")
            break
        parts.append(f"--- page {i + 1} ---\n{page.get_text()}")
    doc.close()
    text = "\n".join(parts)
    if len(text) > MAX_READ_CHARS:
        text = text[:MAX_READ_CHARS] + f"\n[truncated at {MAX_READ_CHARS} chars]"
    return text


# ---------------------------------------------------------------------------- #
# Code execution (jailed)
# ---------------------------------------------------------------------------- #
_RO_BINDS = ("/usr", "/bin", "/sbin", "/lib", "/lib64", "/lib32", "/etc",
             "/opt/conda")


def _bwrap_argv(cwd: str) -> list[str]:
    """bwrap invocation: system runtime read-only, the WHOLE workspace base
    read-write, private /tmp, no network, dies with the gateway.

    The workspace base rather than the single task workspace is bound because
    one gateway serves every concurrent trial and each call names its own cwd;
    binding per-call keeps the mount set identical across trials, which matters
    because a champion and a candidate must run in the same environment.
    """
    argv = ["bwrap"]
    for b in _RO_BINDS:
        if os.path.exists(b):
            argv += ["--ro-bind", b, b]
    argv += [
        "--bind", WORKSPACE_BASE, WORKSPACE_BASE,
        "--proc", "/proc",
        "--dev", "/dev",
        "--tmpfs", "/tmp",
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--chdir", cwd,
    ]
    return argv


@mcp.tool
def code_exec(code: str, workdir: str | None = None) -> dict:
    """Execute a SHELL command. `code` is a shell command string, e.g.
    'ls -la', 'python script.py', or 'python -c "..."'. To run Python, write a
    script with write_file and run it, or use 'python -c'. Do NOT pass raw
    Python source directly as `code`.

    A scientific Python is on PATH (numpy, scipy, matplotlib, pandas, sympy,
    scikit-learn, scikit-image, opencv, control, cvxopt), along with iverilog,
    vvp, octave and ffmpeg. There is no network access. `workdir` sets the cwd
    and must be inside your workspace. Returns {success, output, returncode}."""
    if not code:
        return {"success": False, "output": "Error: required parameter 'code'"}
    cwd_p = _inside(workdir) if workdir else None
    if workdir and cwd_p is None:
        return {"success": False, "output": _denied(workdir)}
    if cwd_p is None or not cwd_p.is_dir():
        return {"success": False,
                "output": "Error: `workdir` must be an existing directory "
                          "inside your workspace."}
    cwd = str(cwd_p)

    env = {
        "PATH": os.path.dirname(CODE_EXEC_PYTHON) + os.pathsep
                + "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": cwd,
        "TMPDIR": "/tmp",
        "LANG": "C.UTF-8",
        "MPLBACKEND": "Agg",
        "OMP_NUM_THREADS": "2",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    argv = (_bwrap_argv(cwd) if (SANDBOX and shutil.which("bwrap")) else [])
    argv += ["/bin/bash", "-lc", code]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True,
                              timeout=CODE_EXEC_TIMEOUT, env=env,
                              cwd=(None if argv[0] == "bwrap" else cwd))
    except subprocess.TimeoutExpired:
        return {"success": False,
                "output": f"Error: timed out after {CODE_EXEC_TIMEOUT}s"}
    except Exception as e:  # noqa: BLE001
        return {"success": False, "output": f"Error: {e}"}
    out = proc.stdout + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
    if len(out) > MAX_READ_CHARS:
        out = out[:MAX_READ_CHARS] + f"\n[output truncated at {MAX_READ_CHARS} chars]"
    return {"success": proc.returncode == 0, "output": out,
            "returncode": proc.returncode}


if __name__ == "__main__":
    os.makedirs(WORKSPACE_BASE, exist_ok=True)
    print(f"[gateway] workspace base {WORKSPACE_BASE}  sandbox={SANDBOX}", flush=True)
    mcp.run(transport="http", host=GATEWAY_HOST, port=GATEWAY_PORT)
