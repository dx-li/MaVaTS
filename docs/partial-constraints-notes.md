# Partial and multi-term constrained matrix factors

The implementation targets Chen, Tsay and Chen (2020), *Constrained Factor
Models for High-Dimensional Matrix-Variate Time Series*, JASA 115(530), 775–793,
[DOI](https://doi.org/10.1080/01621459.2019.1584899), using the accessible
[author manuscript v3](https://arxiv.org/html/1710.06075v3), Sections 3.3–3.4.
The manuscript was revised in October 2022. The former Rutgers journal-PDF
link redirects and publisher full-text access was unavailable during this
implementation; exact journal/manuscript equivalence has not been certified.

## Implemented models

`fit_partial_constrained_factor` fits equation (5), including the cross-factor
blocks. `fit_multiterm_constrained_factor` fits equation (3), with separate
component loading spaces. Neither is a fit of generic PCA to the residuals of
the existing fully constrained estimator. The original
`fit_constrained_factor` API and numerical behavior are unchanged.

For the partial model, orthonormalize each supplied constraint matrix and
construct a complete orthogonal complement. With row bases H_R1,H_R2 and
column bases H_C1,H_C2, form X_lk,t = H_Rl' X_t H_Ck. For any block B, the
moment is the sum, over requested positive lags and all within-block column
pairs i,j, of Omega_ij(h) Omega_ij(h)', where
Omega_ij(h) = sum_t B_t[:,i] B_(t+h)[:,j]' / (T-h).
Row group l uses the sum of the two separate block moments; column group k
uses the analogous sum from transposed blocks. These are equations (12) and
(14). Concatenating blocks before computing the moment adds cross-block
column pairs and produces a different estimator. Independently fitting all
four blocks fails to enforce shared loading spaces.

All blocks use one normalization computed from the original observations.
Block-specific scale normalization would reweight the summed moments. Returned
eigenvalues and `block_moments` use this common working scale; their physical
units are fourth powers of observation units. In each `block_moments[mode]`,
the outer indices remain (row-group, column-group), including for column
moments. Public loadings are orthonormal in original coordinates, and public
scores and signals use original observation units.

`interactions=False` imposes equation (6): only the two diagonal blocks enter
loading estimation and reconstruction. This is a user-supplied scientific
restriction, not automatic model selection. Reconstruction rejects nonzero
cross blocks. With interactions present, shared loading group ranks are
selected jointly from the sums, not separately for each factor block.

## Overlapping multi-term constraints

When every competing term is orthogonal on at least one side, projecting into
the target's two supplied spaces isolates it directly. Otherwise Remark 3
annihilates competing row spans to estimate the target's column loading and
competing column spans to estimate its row loading. Orthogonal complement
coordinates produce the same Gram moments as the full projectors, with less
redundant computation. Competitors already killed by the target-side projection
are excluded from the opposite-side annihilator. For more than two terms,
annihilating the union of competing spans is an explicit algebraic extension
of the paper's displayed two-term procedure.

After estimating mode spaces, construct the joint column design
D = [C_1 kron R_1, ..., C_J kron R_J] in column-major vectorization and solve
min_f ||vec_F(X_t - mean) - D f||² by a retained full-column-rank SVD.
Independent score projections double-count overlapping components. This joint
score solve is a documented reconstruction completion, not an explicit
factor-update formula stated in Remark 3. It does not alter the published
loading moment estimators. `max_design_elements` guards the dense allocation;
no ridge or rank-deficient minimum-norm fallback is implicit.

## Ranks, identification and limits

Partial `row_ranks=(r1,r2)` and `column_ranks=(c1,c2)` separate supplied and
complement loading groups, bounded by their respective dimensions. A None
constraint denotes the entire space (empty complement); a numeric matrix of
shape (dimension,0) denotes an empty supplied space (full complement).
Empty groups force rank zero; explicit zero ranks suppress a group. Multi-term
ranks are a sequence of pairs; (0,0) explicitly omits a term and removes it
from the competing projections. A half-empty multi-term rank is rejected.

Missing ranks use the existing adjacent eigenvalue ratio with its relative
machine-precision floor and half-block-dimension search bound. The paper's
ambient-p search-bound notation is inconsistent with its shorter constraint
spectrum, so that literal bound is not silently implemented. Full block ranks
require explicit specification. Exactly zero partial moments select zero;
this is an algebraic convention, not a calibrated no-factor test. A multi-term
component with no informative projected dynamics must be explicitly omitted.
`rank_diagnostics` records the search bound and numerical moment rank.

The model assumes temporally white measurement noise independent of the
factor process and sufficient nonzero lag information. Only loading spaces
are identified; signs and rotations of latent coordinates remain arbitrary.
The multi-term estimator requires surviving projected loading ranks and a
full-rank joint score design. Numerical guards reject exhausted complements,
lost estimated loading directions and unidentified score decompositions.
They cannot certify unknown population ranks or correct constraints. In
particular, intersecting constraint spans may erase genuine factors even when
a smaller empirical rank appears estimable. Close principal angles amplify
noise: inspect `identification_diagnostics` and `score_condition`.

The procedures use direct linear algebra: `converged=True, n_iter=0` does not
mean successful statistical recovery. No standard errors, confidence intervals,
constraint tests, calibrated rank tests, or factor forecasting dynamics are
implemented. Optional ordinary mean removal/restoration is explicit; there
is no entrywise standardization. `transform` uses each current observation,
the fitted loadings and training mean only, including per-observation numerical
scales. It performs contemporaneous denoising, not forecasting.

## Verification

`tests/test_partial_constraints.py` includes nested-loop lag-moment oracles,
full-projector versus complement-coordinate checks, shared loading recovery,
four-block Pythagorean identities, diagonal/full-space reductions, unequal
group and term ranks, overlapping dense Kronecker score solves, information-loss
failures, basis/observation scaling, empty groups and held-out causality.
The example runs as `python -m examples.partial_constraints` from the repository
root. Benchmark regimes and their noise levels are separate integration designs,
not claimed replications of the paper's simulation experiments.
