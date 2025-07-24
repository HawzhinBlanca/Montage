#!/usr/bin/env python3
"""Fail if subprocess calls are made without timeout or check=True.
Usage: python scripts/no_unchecked_subprocess.py <path1> [<path2> ...]
The script parses each .py file under the provided paths with ast and walks it to
find subprocess.run / subprocess.call / subprocess.Popen.  A violation is
reported when:
  * `timeout` kwarg is missing, OR
  * when the call is `subprocess.run` and `check` kwarg is missing or False.
It exits non-zero if any violations are found, so it can be wired into
pre-commit or CI pipelines.
"""
import ast
import pathlib
import sys
from typing import List, Tuple

SUBPROCESS_FUNCS = {"run", "call", "Popen"}


class SubprocessVisitor(ast.NodeVisitor):
    def __init__(self, filename: str):
        self.filename = filename
        self.violations: List[Tuple[int, str]] = []

    def visit_Call(self, node: ast.Call):  # noqa: N802
        func_name = self._get_func_name(node.func)
        if func_name and func_name.split(".")[-1] in SUBPROCESS_FUNCS:
            self._check_call(node, func_name)
        self.generic_visit(node)

    def _get_func_name(self, func) -> str | None:  # type: ignore[override]
        if isinstance(func, ast.Attribute):
            parts = []
            while isinstance(func, ast.Attribute):
                parts.append(func.attr)
                func = func.value
            if isinstance(func, ast.Name):
                parts.append(func.id)
            return ".".join(reversed(parts))
        elif isinstance(func, ast.Name):
            return func.id
        return None

    def _check_call(self, node: ast.Call, qualname: str):
        # Check for timeout kwarg
        has_timeout = any(
            isinstance(kw.arg, str) and kw.arg == "timeout" for kw in node.keywords
        )
        # Check for check kwarg (only for subprocess.run)
        has_check_true = any(
            isinstance(kw.arg, str)
            and kw.arg == "check"
            and (isinstance(kw.value, ast.Constant) and kw.value.value is True)
            for kw in node.keywords
        )

        if qualname.endswith("run"):
            if not has_timeout or not has_check_true:
                self.violations.append(
                    (
                        node.lineno,
                        "subprocess.run without timeout and check=True",
                    )
                )
        else:  # Popen or call
            if not has_timeout:
                self.violations.append((node.lineno, f"{qualname} without timeout"))


def scan_file(fp: pathlib.Path) -> List[Tuple[int, str]]:
    try:
        tree = ast.parse(fp.read_text("utf8"))
    except SyntaxError:
        return []  # skip invalid files
    visitor = SubprocessVisitor(str(fp))
    visitor.visit(tree)
    return visitor.violations


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python no_unchecked_subprocess.py <path1> [...]", file=sys.stderr)
        sys.exit(1)

    base_paths = [pathlib.Path(p) for p in sys.argv[1:]]
    any_violation = False

    for base in base_paths:
        for py_file in base.rglob("*.py"):
            violations = scan_file(py_file)
            for lineno, msg in violations:
                any_violation = True
                rel = py_file.relative_to(pathlib.Path.cwd())
                print(f"{rel}:{lineno}: {msg}")

    if any_violation:
        sys.exit(1)


if __name__ == "__main__":
    main() 