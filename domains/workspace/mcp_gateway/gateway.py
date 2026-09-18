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
"""Single FastMCP streamable-http gateway exposing the tool surface the
react_toolbelt agent needs for job-bench:

  - filesystem read tools (read_text_file, list_files, get_directory_tree, search_files)
  - write_file
  - read_pdf (pymupdf)
  - code_exec (subprocess shell; conda python on PATH for the data stack)
  - web_search (google genai + google_search grounding, Vertex mode)

Deliberately excludes the archipelago mail/calendar/chat app tools and the
buggy spreadsheets/documents meta-tools — office formats go through code_exec
(pandas / openpyxl / python-docx / python-pptx), which is more reliable.

Tools take absolute paths (job-bench's native contract: the wrapper prompt hands
the agent absolute task_folder / output paths, mirroring `claude --add-dir`).
Stateless w.r.t. task, so one gateway safely serves all concurrent tasks.
"""

import os
import subprocess
import sys
from pathlib import Path

from fastmcp import FastMCP

# ---- config (env-driven; see config.py for defaults) -------------------------
GATEWAY_HOST = os.environ.get("GATEWAY_HOST", "127.0.0.1")
GATEWAY_PORT = int(os.environ.get("GATEWAY_PORT", "8990"))
CODE_EXEC_PYTHON = os.environ.get("CODE_EXEC_PYTHON", "/opt/conda/bin/python3")
CODE_EXEC_TIMEOUT = int(os.environ.get("CODE_EXEC_TIMEOUT", "300"))
VERTEX_PROJECT = os.environ.get("VERTEX_PROJECT", "")
VERTEX_LOCATION = os.environ.get("VERTEX_LOCATION", "global")
WEB_SEARCH_MODEL = os.environ.get("WEB_SEARCH_MODEL", "gemini-3.5-flash")

MAX_READ_CHARS = 200_000

mcp = FastMCP(
    name="jobbench-gateway",
    instructions=(
        "Tools for completing white-collar file tasks: read/write files, run "
        "shell+python code for data processing and producing deliverables "
        "(xlsx/docx/pptx/pdf/csv), read PDFs, and search the live web."
    ),
)


# ---------------------------------------------------------------------------- #
# Filesystem tools (absolute paths)
# ---------------------------------------------------------------------------- #
@mcp.tool
def read_text_file(path: str) -> str:
    """Read a UTF-8 text file and return its contents. `path` is an absolute
    path. Use for .txt/.csv/.md/.json/.py and other plaintext. For binary
    office/pdf formats use read_pdf or code_exec (pandas/openpyxl/docx)."""
    p = Path(path)
    if not p.is_file():
        return f"Error: not a file: {path}"
    data = p.read_text(encoding="utf-8", errors="replace")
    if len(data) > MAX_READ_CHARS:
        return data[:MAX_READ_CHARS] + f"\n[truncated at {MAX_READ_CHARS} chars]"
    return data


@mcp.tool
def list_files(path: str = ".") -> str:
    """List files and folders (one per line, dirs suffixed with '/') in the
    given absolute directory path."""
    p = Path(path)
    if not p.is_dir():
        return f"Error: not a directory: {path}"
    out = []
    for entry in sorted(p.iterdir()):
        out.append(entry.name + ("/" if entry.is_dir() else ""))
    return "\n".join(out) if out else "(empty)"


@mcp.tool
def get_directory_tree(path: str = ".", max_entries: int = 500) -> str:
    """Return a recursive directory tree (relative paths) rooted at the given
    absolute path, capped at max_entries lines."""
    root = Path(path)
    if not root.is_dir():
        return f"Error: not a directory: {path}"
    lines = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        rel = os.path.relpath(dirpath, root)
        prefix = "" if rel == "." else rel + "/"
        for fn in sorted(filenames):
            lines.append(prefix + fn)
            if len(lines) >= max_entries:
                lines.append(f"[truncated at {max_entries} entries]")
                return "\n".join(lines)
    return "\n".join(lines) if lines else "(empty)"


@mcp.tool
def search_files(path: str, name_contains: str) -> str:
    """Recursively find files whose name contains the substring `name_contains`
    under the given absolute directory. Returns absolute paths, one per line."""
    root = Path(path)
    if not root.is_dir():
        return f"Error: not a directory: {path}"
    hits = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            if name_contains.lower() in fn.lower():
                hits.append(os.path.join(dirpath, fn))
    return "\n".join(sorted(hits)) if hits else "(no matches)"


@mcp.tool
def write_file(path: str, content: str) -> str:
    """Write `content` (UTF-8 text) to the absolute `path`, creating parent
    directories as needed. Overwrites. For binary deliverables (xlsx/docx/pptx/
    pdf/images) generate them with code_exec instead."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} chars to {path}"


@mcp.tool
def read_pdf(path: str, max_pages: int = 100) -> str:
    """Extract text from a PDF file at the absolute `path` (up to max_pages).
    Returns page-delimited text. For scanned/image PDFs text may be empty; fall
    back to code_exec with pdfplumber/pymupdf if needed."""
    import fitz  # pymupdf

    p = Path(path)
    if not p.is_file():
        return f"Error: not a file: {path}"
    try:
        doc = fitz.open(path)
    except Exception as e:  # noqa: BLE001
        return f"Error opening PDF: {e!r}"
    parts = []
    for i, page in enumerate(doc):
        if i >= max_pages:
            parts.append(f"[truncated at {max_pages} pages of {doc.page_count}]")
            break
        parts.append(f"----- page {i + 1} -----\n{page.get_text()}")
    text = "\n".join(parts)
    if len(text) > MAX_READ_CHARS:
        text = text[:MAX_READ_CHARS] + f"\n[truncated at {MAX_READ_CHARS} chars]"
    return text


# ---------------------------------------------------------------------------- #
# Code execution
# ---------------------------------------------------------------------------- #
@mcp.tool
def code_exec(code: str, workdir: str | None = None) -> dict:
    """Execute a SHELL command in a persistent working directory. `code` is a
    shell command string, e.g. 'ls -la', 'pip install pandas', or
    'python script.py'. To run Python, write a script and run it, or use
    'python -c "..."'. Do NOT pass raw Python source directly as `code`.

    A rich scientific Python is on PATH (pandas, numpy, scipy, scikit-learn,
    matplotlib, openpyxl, python-docx, python-pptx, pdfplumber, reportlab, fitz).
    Use absolute paths. `workdir` optionally sets the cwd (default: /tmp).
    Returns {success, output} where output combines stdout+stderr."""
    if code is None:
        return {"success": False, "output": "Error: required parameter 'code'"}
    cwd = workdir if (workdir and os.path.isdir(workdir)) else "/tmp"
    env = dict(os.environ)
    conda_bin = os.path.dirname(CODE_EXEC_PYTHON)
    env["PATH"] = conda_bin + os.pathsep + env.get("PATH", "")
    try:
        proc = subprocess.run(
            code,
            shell=True,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=CODE_EXEC_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return {"success": False, "output": f"Error: timed out after {CODE_EXEC_TIMEOUT}s"}
    out = proc.stdout + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
    if len(out) > MAX_READ_CHARS:
        out = out[:MAX_READ_CHARS] + f"\n[output truncated at {MAX_READ_CHARS} chars]"
    return {"success": proc.returncode == 0, "output": out, "returncode": proc.returncode}


# ---------------------------------------------------------------------------- #
# Web search (google genai + google_search grounding, Vertex mode)
# ---------------------------------------------------------------------------- #
_genai_client = None


def _get_genai_client():
    global _genai_client
    if _genai_client is None:
        from google import genai

        _genai_client = genai.Client(
            vertexai=True, project=VERTEX_PROJECT, location=VERTEX_LOCATION
        )
    return _genai_client


def _register_web_search():
    pass


if os.environ.get("ENABLE_WEB_SEARCH", "0") == "1":
    _ws_decorator = mcp.tool
else:
    _ws_decorator = lambda f: f  # not registered: closed-workspace benchmark


@_ws_decorator
def web_search(query: str, k: int = 5) -> str:
    """Search the live web and return a grounded answer plus the top-k source
    snippets (title, url, text). Use this to find external reference data that
    is NOT in the task folder (the task may require datasets/specs you must
    discover online)."""
    from google.genai import types

    client = _get_genai_client()
    try:
        resp = client.models.generate_content(
            model=WEB_SEARCH_MODEL,
            contents=query,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())]
            ),
        )
    except Exception as e:  # noqa: BLE001
        return f"web_search error: {e!r}"

    answer = resp.text or ""
    try:
        gm = resp.candidates[0].grounding_metadata
        chunks = list(gm.grounding_chunks or [])
        supports = list(gm.grounding_supports or [])
    except Exception:  # noqa: BLE001
        chunks, supports = [], []

    text_by_chunk: dict[int, str] = {}
    for sup in supports:
        seg_text = ""
        try:
            seg_text = sup.segment.text or ""
        except Exception:  # noqa: BLE001
            seg_text = ""
        for idx in (sup.grounding_chunk_indices or []):
            text_by_chunk.setdefault(idx, seg_text)

    lines = [f"ANSWER:\n{answer}\n", "SOURCES:"]
    for i, ch in enumerate(chunks[:k]):
        try:
            title = ch.web.title or ""
            url = ch.web.uri or ""
        except Exception:  # noqa: BLE001
            title, url = "", ""
        snippet = text_by_chunk.get(i, "")
        lines.append(f"[{i + 1}] {title}\n    {url}\n    {snippet}")
    if not chunks:
        lines.append("(no grounded sources returned)")
    return "\n".join(lines)


if __name__ == "__main__":
    print(
        f"[gateway] starting on http://{GATEWAY_HOST}:{GATEWAY_PORT}/mcp/  "
        f"(code_exec python={CODE_EXEC_PYTHON})",
        file=sys.stderr,
        flush=True,
    )
    mcp.run(transport="http", host=GATEWAY_HOST, port=GATEWAY_PORT)
