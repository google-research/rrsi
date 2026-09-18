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
"""Git plumbing for candidate isolation.

Each domain evolves on its own branch `evolve/<domain>`. Every candidate
harness of a round is drafted and evaluated in its own git worktree checked out
on a branch `<domain>/r<t><variant>` from the incumbent, so variants cannot see
each other's edits and can be evaluated concurrently (Alg. 2, "in parallel").
Accepting a candidate fast-forwards `evolve/<domain>` to its commit. The
harness tree hash (`git rev-parse <commit>:<harness path>`) is what the
frontier records, so commits outside the harness never look like a change.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

AUTHOR = ["-c", "user.email=rrsi@localhost", "-c", "user.name=rrsi"]


def sh(cmd: list[str], cwd: Path | str, check: bool = False) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed in {cwd}: {r.stderr[-800:]}")
    return r


def git(cwd: Path | str, *args: str, check: bool = False) -> subprocess.CompletedProcess:
    return sh(["git", *AUTHOR, *args], cwd, check=check)


def head(cwd) -> str:
    return git(cwd, "rev-parse", "--short", "HEAD").stdout.strip()


def rev(cwd, ref: str) -> str:
    return git(cwd, "rev-parse", "--short", ref).stdout.strip()


def tree_hash(cwd, ref: str, path: str) -> str:
    return git(cwd, "rev-parse", f"{ref}:{path}").stdout.strip()[:12]


def branch_exists(cwd, name: str) -> bool:
    return git(cwd, "rev-parse", "--verify", "--quiet", f"refs/heads/{name}").returncode == 0


def ensure_branch(cwd, name: str, start: str = "HEAD") -> None:
    if not branch_exists(cwd, name):
        git(cwd, "branch", name, start, check=True)


def worktree_add(repo: Path, path: Path, branch: str, start: str) -> Path:
    """Fresh worktree on a NEW branch `branch` at `start` (old one removed)."""
    worktree_remove(repo, path, branch)
    path.parent.mkdir(parents=True, exist_ok=True)
    git(repo, "worktree", "add", "-b", branch, str(path), start, check=True)
    return path


def worktree_remove(repo: Path, path: Path, branch: str | None = None) -> None:
    if path.exists():
        git(repo, "worktree", "remove", "--force", str(path))
        shutil.rmtree(path, ignore_errors=True)
    git(repo, "worktree", "prune")
    if branch and branch_exists(repo, branch):
        git(repo, "branch", "-D", branch)


def is_clean(cwd, path: str) -> bool:
    return git(cwd, "status", "--porcelain", "--", path).stdout.strip() == ""


def revert_path(cwd, path: str) -> None:
    git(cwd, "reset", "-q", "--", path)
    git(cwd, "checkout", "--", path)
    git(cwd, "clean", "-fdq", "--", path)


def diff_with_new_files(cwd: Path, path: str) -> str:
    """Working-tree diff of `path` plus the full text of untracked files, so a
    module the proposer just created is reviewed by the critic in full."""
    git(cwd, "add", "-N", "--", path)
    d = git(cwd, "diff", "--", path).stdout
    return d


def commit_path(cwd, path: str, msg: str) -> str:
    git(cwd, "add", "-A", "--", path)
    git(cwd, "commit", "-q", "--allow-empty", "-m", msg, check=True)
    return head(cwd)


def fast_forward(repo: Path, branch: str, commit: str) -> None:
    """Move refs/heads/<branch> to <commit> if it is a fast-forward."""
    r = git(repo, "merge-base", "--is-ancestor", branch, commit)
    if r.returncode != 0:
        raise RuntimeError(f"{commit} is not a fast-forward of {branch}")
    git(repo, "update-ref", f"refs/heads/{branch}", commit, check=True)
