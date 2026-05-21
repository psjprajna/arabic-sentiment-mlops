"""Fitness function: ADR-0002 domain boundary enforcement.

The domain layer (sentiment/domain/) must not import any infrastructure package.
Must run in <1 second with no network access.
"""

from __future__ import annotations

import ast
from pathlib import Path

DOMAIN_DIR = Path(__file__).parent.parent / "src" / "sentiment" / "domain"
FORBIDDEN_IMPORTS: frozenset[str] = frozenset(
    {
        "fastapi",
        "mlflow",
        "transformers",
        "torch",
        "sklearn",
        "catboost",
        "httpx",
        "requests",
    }
)


def _get_top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and not node.module.startswith("sentiment.domain"):
                imports.add(node.module.split(".")[0])
    return imports


def test_domain_does_not_import_infrastructure() -> None:
    """Domain modules must not import any infrastructure package.

    Enforces ADR-0002: Hexagonal / Ports & Adapters architecture boundary.
    """
    violations: list[str] = []
    for py_file in DOMAIN_DIR.rglob("*.py"):
        if py_file.name.startswith("_"):
            continue
        found = _get_top_level_imports(py_file) & FORBIDDEN_IMPORTS
        if found:
            rel = py_file.relative_to(DOMAIN_DIR.parent.parent)
            violations.append(f"  {rel}: imports {sorted(found)}")

    assert not violations, (
        "ADR-0002 VIOLATION — domain layer imported infrastructure:\n" + "\n".join(violations)
    )
