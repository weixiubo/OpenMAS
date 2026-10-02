"""Check maintained code size and branch complexity; frozen source is hash-checked separately."""

from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("code/core/src", "code/experiments/openagora_experiments", "scripts", "examples")


def branch_count(node: ast.AST, root: ast.AST) -> int:
    if node is not root and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        return 0
    count = int(isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.IfExp, ast.ExceptHandler)))
    if isinstance(node, ast.BoolOp):
        count += len(node.values) - 1
    if isinstance(node, ast.comprehension):
        count += 1 + len(node.ifs)
    if isinstance(node, ast.Match):
        count += max(0, len(node.cases) - 1)
    return count + sum(branch_count(child, root) for child in ast.iter_child_nodes(node))


def inspect(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source)
    name = str(path.relative_to(ROOT))
    failures = []
    if len(source.splitlines()) > 600:
        failures.append(f"{name}: module exceeds 600 lines")
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        length = node.end_lineno - node.lineno + 1
        complexity = 1 + branch_count(node, node)
        if length > 120 or complexity > 15:
            failures.append(f"{name}:{node.lineno} {node.name}: {length} lines, complexity {complexity}")
    return failures


def main() -> int:
    failures = [
        failure
        for directory in SOURCES
        for path in sorted((ROOT / directory).rglob("*.py"))
        for failure in inspect(path)
    ]
    print(json.dumps({"status": "failed" if failures else "ok", "exceptions": [], "failures": failures}, indent=2))
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
