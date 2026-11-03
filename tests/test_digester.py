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
"""The digester's read jail: the tools may not leave the round's traces directory.

    python3 -m pytest -q tests/test_digester.py    or    python3 tests/test_digester.py

These are pure path tests: no model call, no trace content.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rrsi import digester                                  # noqa: E402


def _jail(td: str) -> Path:
    """`<td>/analysis/rendered` plus a sibling that shares its name prefix."""
    base = Path(td) / "analysis"
    root = base / "rendered"
    root.mkdir(parents=True)
    (root / "task-1.txt").write_text("trace line\n", encoding="utf-8")
    sibling = base / "rendered_private"
    sibling.mkdir()
    (sibling / "secret.txt").write_text("not for the digester\n", encoding="utf-8")
    return root


def test_safe_allows_a_path_inside_the_jail():
    with tempfile.TemporaryDirectory() as td:
        root = _jail(td)
        assert digester._safe(root, "task-1.txt") == (root / "task-1.txt").resolve()
        nested = root / "nested" / "deeper"
        nested.mkdir(parents=True)
        (nested / "x.txt").write_text("x", encoding="utf-8")
        assert digester._safe(root, "nested/deeper/x.txt").is_file()


def test_safe_refuses_a_sibling_that_shares_the_root_prefix():
    # `str(p).startswith(str(root))` calls `.../rendered_private` contained in
    # `.../rendered`; the jail must be judged by path component instead.
    with tempfile.TemporaryDirectory() as td:
        root = _jail(td)
        try:
            digester._safe(root, "../rendered_private/secret.txt")
        except ValueError as e:
            assert "escapes" in str(e)
        else:
            raise AssertionError("a prefix-sharing sibling must not resolve inside the jail")


def test_read_file_refuses_the_prefix_sibling():
    with tempfile.TemporaryDirectory() as td:
        root = _jail(td)
        try:
            digester._read_file(root, "../rendered_private/secret.txt", None, None)
        except ValueError:
            pass
        else:
            raise AssertionError("read_file must not read outside the traces dir")
        assert "trace line" in digester._read_file(root, "task-1.txt", 1, 10)


def test_glob_refuses_a_pattern_that_leaves_the_jail():
    with tempfile.TemporaryDirectory() as td:
        root = _jail(td)
        for pattern in ("../*", "../../*", str(Path(td) / "*")):
            out = digester._glob(root, pattern)
            assert out.startswith("ERROR:"), pattern


def test_glob_lists_only_inside_names():
    with tempfile.TemporaryDirectory() as td:
        root = _jail(td)
        assert digester._glob(root, "*.txt") == "task-1.txt"
        assert digester._glob(root, "").splitlines() == ["task-1.txt"]


if __name__ == "__main__":
    import inspect
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and inspect.isfunction(fn):
            try:
                fn()
                print(f"ok   {name}")
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f"FAIL {name}: {e!r}")
    sys.exit(1 if fails else 0)
