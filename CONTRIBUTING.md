# Contributing scientific methods

Install `python -m pip install -e '.[dev,docs]'`, then run:

```bash
python -m pytest
python -m benchmarks.run --quick --repeats 1
python -m examples.quickstart
python -m black --check mavats tests benchmarks examples
python -m isort --check-only mavats tests benchmarks examples
python -m build
```

For a method contribution, cite a primary paper and identify the exact algorithm,
normalizations, assumptions, loss, initialization, stopping criterion and any
departures. Add an independent numerical oracle, an executable usage example,
and a seeded comparison with known truth. Shape-only tests do not establish
scientific correctness. Statistical inference needs size/coverage experiments
in addition to algebraic checks.

Keep time on the first axis, use local random generators, preserve caller inputs,
and expose iterative failure/convergence. Compare factor subspaces or signals
rather than unidentified coordinates. Validate degenerate cases deliberately.
Prefer clear NumPy/SciPy implementations before adding a compiled backend; justify
optimizations with reproducible profiling and equivalence tests.

Update `docs/methods.md`, `docs/references.bib`, and the benchmark inventory when
adding coverage. An implementation of one estimator from a paper must not be
listed as covering its other estimators, rank procedures, or inferential theory.

The old Poetry lock file was removed when the rebuild adopted standard Python
project metadata and NumPy 1.26+/2.x compatibility. Runtime bounds live in
`pyproject.toml`; minimum-dependency and current-dependency CI are separate.
The previous lock remains recoverable from Git history.

## Citation requirement

Every public scientific procedure must include a `References` section in its
docstring with author(s), paper title and primary DOI/manuscript link. Add its
canonical name and attribution scope to [docs/citations.md](docs/citations.md),
and complete BibTeX metadata to [docs/references.bib](docs/references.bib).
Include all distinct paper variants behind a shared entry point. Result helpers
inherit the creating model's references; generic utilities and package extensions
must be identified honestly, without inventing a dedicated paper implementation.
`tests/test_citations.py` checks public API/index coverage and citation metadata.
