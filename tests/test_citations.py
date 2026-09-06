"""Keep method-level attribution complete as the public API grows."""

import ast
import importlib
import inspect
import re
from pathlib import Path

import mavats

ROOT = Path(__file__).resolve().parents[1]


def _public_definitions():
    definitions = {}
    for path in sorted((ROOT / "mavats").glob("*.py")):
        if path.name.startswith("_"):
            continue
        module = f"mavats.{path.stem}"
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not node.name.startswith("_"):
                    definitions[f"{module}.{node.name}"] = node
    # Scientific procedures outside the public package, rather than experiment
    # orchestration. Private names must not hide separately implemented methods.
    for module, name in (
        ("structured_ar", "_fit_vector_vecm"),
        ("monitoring", "_wilson"),
        ("inference", "summary"),
        ("marma", "_TrueMARMA"),
        ("constrained_extensions", "_KnownLoadingProjection"),
        ("constrained_extensions", "_IndependentSum"),
        ("advanced_matrix", "_gaussian_score"),
    ):
        path = ROOT / f"benchmarks/{module}.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        node = next(
            n
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name
        )
        definitions[f"benchmarks.{module}.{name}"] = node
    return definitions


def _citation_rows():
    rows = {}
    document = (ROOT / "docs/citations.md").read_text(encoding="utf-8")
    for line in document.splitlines():
        match = re.match(
            r"\| `((?:mavats|benchmarks)\.[\w.]+)` \| (.*?) \| (.*?) \|$", line
        )
        if match:
            name, references, scope = match.groups()
            assert name not in rows, f"Duplicate citation index row: {name}"
            keys = re.findall(r"\[([\w]+)\]\(#([\w]+)\)", references)
            assert keys and all(label == anchor for label, anchor in keys), name
            assert scope.strip(), f"Missing attribution scope: {name}"
            rows[name] = [key for key, _ in keys]
    return document, rows


def test_every_public_function_and_class_has_a_citation_index_entry():
    _, rows = _citation_rows()
    definitions = _public_definitions()
    assert set(rows) == set(definitions), {
        "missing": sorted(set(definitions) - set(rows)),
        "stale": sorted(set(rows) - set(definitions)),
    }


def test_every_public_scientific_function_cites_a_paper_in_its_docstring():
    for name, node in _public_definitions().items():
        if isinstance(node, ast.ClassDef):
            continue  # Result/exception classes inherit the indexed model paper.
        doc = ast.get_docstring(node) or ""
        assert "References\n" in doc, f"Missing References section: {name}"
        references = doc.split("References\n", 1)[1]
        assert re.search(
            r"https://(?:doi\.org|arxiv\.org|academic\.oup\.com|"
            r"yuefenghan\.github\.io|link\.springer\.com)/",
            references,
        ), name


def test_index_references_have_bibtex_metadata_and_readable_primary_links():
    document, rows = _citation_rows()
    bibliography = (ROOT / "docs/references.bib").read_text(encoding="utf-8")
    matches = list(re.finditer(r"^@\w+\{([\w]+),", bibliography, flags=re.MULTILINE))
    assert len(matches) == len({match[1] for match in matches}), "Duplicate BibTeX key"
    entries = {}
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(bibliography)
        entries[match[1]] = bibliography[match.end() : end]
    for key in {key for keys in rows.values() for key in keys}:
        assert key in entries, f"Missing BibTeX entry: {key}"
        for field in ("author", "title", "year", "url"):
            assert re.search(rf"\b{field}\s*=\s*\{{[^}}]+\}}", entries[key]), (
                key,
                field,
            )
        section = re.search(
            rf"^### {key}\n\n(.*?)(?=\n### |\Z)", document, re.MULTILINE | re.DOTALL
        )
        assert section and re.search(r"\[[^\]]+\]\(https://[^)]+\)", section[1]), key


def test_top_level_reexports_resolve_to_indexed_canonical_definitions():
    _, rows = _citation_rows()
    for name in mavats.__all__:
        value = getattr(mavats, name)
        if inspect.isfunction(value) or inspect.isclass(value):
            assert f"{value.__module__}.{value.__name__}" in rows, name
    for canonical in rows:
        module, name = canonical.rsplit(".", 1)
        value = getattr(importlib.import_module(module), name)
        assert inspect.isfunction(value) or inspect.isclass(value), canonical
