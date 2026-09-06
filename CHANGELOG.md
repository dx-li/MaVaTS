# Release notes

## 0.2.0a2 — 2026-09-06

Documentation and reproducibility alpha release; numerical estimators are
unchanged from 0.2.0a1. APIs remain experimental.

- Forty-nine executable real-data method/variant walkthroughs with computed
  figures, paper links, diagnostics and downloadable numerical arrays.
- Bundled Beijing air-quality teaching extract with CC BY 4.0 attribution,
  source/data hashes, retained missingness and a fixed chronological split.
- Training-only preprocessing and observed-target scoring; forecasting is
  explicitly separated from contemporaneous factor reconstruction. Unverified
  assumptions and unconverged fits remain visible, not treated as validation.
- Optional `examples` extra for Matplotlib. Guides, figures, data and example
  scripts ship in the source distribution; the wheel contains the library.
- Release validation checks nested documentation/data assets and executes the
  complete observational gallery against an independently installed wheel.

Install with `python -m pip install "mavats==0.2.0a2"`.
For runnable examples use the
[matching release source](https://github.com/dx-li/MaVaTS/tree/v0.2.0a2),
[data protocol](https://github.com/dx-li/MaVaTS/blob/v0.2.0a2/docs/real-world-examples.md)
and [visual gallery](https://github.com/dx-li/MaVaTS/blob/v0.2.0a2/docs/gallery/index.md).
One observational dataset does not establish every model's assumptions,
inference validity, or suitability for a particular application.

## 0.2.0a1 — 2026-09-06

First alpha release of the MaVaTS development rebuild. This is not a claim of
complete literature coverage, stable APIs or reproduction of every cited paper.

### Included

- Matrix autoregression: projection, ALS, separable Gaussian likelihood,
  multiple lags, reduced-rank LS, sparse posterior-mode estimation, envelope
  likelihood, conditional matrix ARMA and cointegration.
- Matrix/tensor factors: alpha-PCA, projected and lagged factors, TOPUP/TIPUP
  and iterative variants, published factor-rank criteria and explicit stability
  paths, constrained and CP factors, robust and threshold estimators.
- Additional scoped methods for two-way dynamics, decorrelation, matrix GARCH,
  sequential monitoring, MAR inference and multi-term tensor autoregression.
- Method-level paper attribution enforced by tests: 114 public API/benchmark
  entries across 29 papers, with explicit algorithm and inference boundaries.
- Seventeen executable examples and retained synthetic benchmarks. Results
  preserve failures, uncertainty, source fingerprints and initialization audits.

### Compatibility and limitations

Requires Python 3.10+, NumPy 1.26+ and SciPy 1.13+. Validation covers Python
3.10–3.13 on Linux, macOS and Windows, including the minimum dependencies.
New APIs accept time-first arrays. Legacy submodules remain, but users of 0.1.x
should check each function's documented shapes and options: this is a substantial
rebuild, not a blanket compatibility guarantee. No mandatory compiled extension
or Rust runtime is introduced.

Local optimizer convergence does not prove global optimality, model
identification, stability or good prediction. Paper-specific standard errors,
selection rules and robust/missing-data extensions are not implied by the
presence of a point estimator. Broader literature coverage, real-data studies
and a unified documentation portal remain unfinished. API docs are published
online; detailed guides, citations and benchmark reports remain in the source
repository. Historical benchmark versions and hashes describe their original
development snapshots and are deliberately not rewritten for this release.

Install this prerelease explicitly with `python -m pip install mavats==0.2.0a1`.
See [the complete method inventory](https://github.com/dx-li/MaVaTS/blob/v0.2.0a1/docs/methods.md)
and [citation index](https://github.com/dx-li/MaVaTS/blob/v0.2.0a1/docs/citations.md)
before interpreting a result.
