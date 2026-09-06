# Conditional ARMA and constrained-factor checkpoint

These synthetic integration experiments target the explicitly scoped methods in
[the protocol](../README.md), not paper-table replications or general superiority.
The independent replicate is a simulated series, not an individual time point.
Reported uncertainty is the replicate standard error, `sample SD / sqrt(R)`.
Finite unconverged fits are included, not dropped. Raw records retain options,
errors, timings, diagnostics, random seeds, dependencies and source fingerprints.

Sources are frozen at `9ed6ebd756bb7a285e18e085e57308c3bb4599ed`. The earlier
eight studies retain their own historical implementation fingerprints; these
new results do not relabel those older artifacts. Single-thread BLAS settings
are recorded, but other verification processes shared the local machine during
part of execution. Timing is descriptive, not an isolated hardware comparison.

## Conditional matrix ARMA forecasts

[Raw records](marma.json) contain 160 runs, five paired series per cell,
with no execution errors. Seven selected fits remain unconverged: all five
near-cancellation LS fits and the MLE fits at seeds 1973 and 1974 reached their
300-iteration budget. All 40 selected MARMA fits are feasible under the requested
full-polynomial bounds; none certifies structural identification. All 40 MLE
LS warm-up solves converged under their separately documented looser tolerance.
One near-cancellation LS record retains an earlier feasible iterate. Selection
uses the smallest retained objective, so a selected unconverged start can have
a converged alternative; convergence flags are not overwritten.

Models use 500 training and 50 test matrices, with identical 46 forecast origins
for horizons one and five. Forecast-error MSE is averaged over matrix entries
and origins within each series, then summarized across series. The true-parameter
comparator also observes latent past innovations, unlike the fitted conditional
zero-prefix filter. Sampling variation means even this comparator need not have
the lowest realized test error in every cell.

One-step MSE, mean ± series SE:

| Method | Isotropic | Separable | Near cancellation | Nonseparable |
| --- | ---: | ---: | ---: | ---: |
| MARMA LS, known orders | .9952 ± .0507 | .9591 ± .0632 | 1.0206 ± .0972 | .9868 ± .0877 |
| MARMA MLE, known orders | .9953 ± .0505 | .9581 ± .0603 | .9708 ± .0607 | .9836 ± .0865 |
| MAR ALS(1) | 1.0768 ± .0703 | 1.0209 ± .0857 | .9652 ± .0611 | 1.0616 ± .1184 |
| MAR MLE(1) | 1.0768 ± .0702 | 1.0198 ± .0831 | .9610 ± .0599 | 1.0588 ± .1159 |
| VAR(1) | 1.0921 ± .0721 | 1.0339 ± .0866 | .9736 ± .0611 | 1.0783 ± .1219 |
| VAR(5) | 1.0732 ± .0662 | 1.0323 ± .0739 | 1.0307 ± .0724 | 1.0610 ± .0921 |
| Zero | 1.5277 ± .0688 | 1.5381 ± .0974 | .9563 ± .0590 | 1.5372 ± .1116 |
| True parameters and latent innovations | .9814 ± .0511 | .9519 ± .0607 | .9519 ± .0607 | .9781 ± .0813 |

Five-step MSE at those same origins:

| Method | Isotropic | Separable | Near cancellation | Nonseparable |
| --- | ---: | ---: | ---: | ---: |
| MARMA LS, known orders | 1.5213 ± .0659 | 1.5225 ± .0814 | .9959 ± .0591 | 1.5076 ± .1152 |
| MARMA MLE, known orders | 1.5213 ± .0659 | 1.5235 ± .0818 | .9601 ± .0449 | 1.5090 ± .1155 |
| MAR ALS(1) | 1.5328 ± .0686 | 1.5318 ± .0829 | .9541 ± .0439 | 1.5153 ± .1166 |
| MAR MLE(1) | 1.5329 ± .0686 | 1.5320 ± .0828 | .9541 ± .0439 | 1.5181 ± .1177 |
| VAR(1) | 1.5278 ± .0664 | 1.5297 ± .0787 | .9541 ± .0439 | 1.5142 ± .1135 |
| VAR(5) | 1.5817 ± .0557 | 1.5993 ± .0851 | .9788 ± .0433 | 1.5972 ± .1193 |
| Zero | 1.5261 ± .0608 | 1.5347 ± .0791 | .9600 ± .0421 | 1.5216 ± .1103 |
| True parameters and latent innovations | 1.5226 ± .0689 | 1.5320 ± .0893 | .9557 ± .0440 | 1.5195 ± .1199 |

Outside cancellation, MARMA LS reduces paired one-step MSE relative to MAR ALS
by .0816 ± .0311, .0618 ± .0340 and .0748 ± .0428 in the isotropic, separable
and nonseparable regimes. At five steps most methods approach the unconditional
variance, so one-step gains do not imply long-horizon gains. The near-cancelling
process is almost white: simple MAR/zero forecasts are competitive or better,
and the local optimization is harder. Mean relative AR/MA operator errors there
are 9.74/10.27 for LS and 5.11/5.41 for MLE. Admissible roots therefore do not
establish coefficient identification or an accurate structural decomposition.

Covariance misspecification matters even when forecast means look similar.
Mean innovation-covariance relative errors for LS/MLE are .1195/.0649 under
isotropic noise, .1055/.0724 under separable noise, and .0956/.4495 under
nonseparable noise. In the last regime the mean one-step negative Gaussian
log score is 7.4111 for LS's full residual covariance versus 7.7875 for the
separable MLE (lower is better). Raw records also retain five-step scores and
individual covariance errors. No parameter-estimation uncertainty is included.

Whole-fit median times, including both starts and MLE warm-ups, are roughly
2.5–3.7 seconds outside cancellation and 13.1/15.9 seconds for LS/MLE under
cancellation. MAR and VAR are much cheaper in this small-dimensional design.
Five repetitions and a shared machine do not support precise speed rankings.

## Partial constrained factors

[Raw records](constrained-extensions.json) include 280 partial-factor and 150
multi-term-factor runs, with no execution errors or unconverged outcomes.
The new partial and multi-term estimators report zero optimization iterations;
the projected-PCA comparator performs one projection sweep. Successful
termination does not establish correct ranks, constraints or signal recovery.
There are ten series per cell, 400 training and 30 held-out 8×10 matrices.

Held-out relative Frobenius signal error, mean ± series SE; lower is better.
Each current noisy held-out matrix is observed: this is denoising, not forecasting.

| Method | Full interactions | Diagonal only | Weak complement | Rotated/wrong priors |
| --- | ---: | ---: | ---: | ---: |
| Partial, fixed generating group ranks | .0986 ± .0025 | .1533 ± .0056 | .1949 ± .0089 | .6010 ± .0098 |
| Partial, ratio-selected group ranks | .0986 ± .0025 | .1533 ± .0056 | .1949 ± .0089 | .5597 ± .0411 |
| Partial, diagonal restriction | .7645 ± .0109 | .1024 ± .0036 | .2600 ± .0099 | .8724 ± .0061 |
| Fully constrained primary block | .8672 ± .0087 | .6350 ± .0225 | .2442 ± .0092 | .9612 ± .0021 |
| Unconstrained lagged factors | .0995 ± .0025 | .1546 ± .0056 | .2407 ± .0104 | .0995 ± .0025 |
| Projected PCA | .0983 ± .0025 | .1528 ± .0056 | .1943 ± .0089 | .0983 ± .0025 |
| Known-loading projection | .0978 ± .0025 | .1521 ± .0056 | .1933 ± .0089 | .0978 ± .0025 |

With correct priors, the automatic selector recovered the generating group
ranks in all 30 series across the first three regimes. This is a finite design
result, not a calibrated rank guarantee. Fixed-rank partial fits do not beat
projected PCA here: paired error differences (partial minus PCA) are
.00036 ± .00012, .00053 ± .00018 and .00065 ± .00048 respectively.
Partial fits substantially outperform a fully constrained primary block when
the omitted groups matter. The diagonal restriction helps only when its zero
cross-block assumption is correct. It can beat the known-loading comparator
because that comparator knows the full Tucker loading spaces, **not** which
cross blocks vanish; the comparator is not a universal lower bound.

Wrong priors change the constraints while keeping the full-interaction data
exactly paired. Actual supplied/complement ranks become (1,3)/(2,2), whereas
fixed fits keep generating ranks (1,2)/(2,1). The automatic row-complement
search is capped at two and cannot recover the actual rank three. Thus this
is a combined constraint/rank-misspecification experiment, not an isolated test
of constraint validity. Automatic fits select row (1,2) six times and (1,1)
four times, with column (2,2) throughout. Unconstrained methods retain their
paired accuracy; nominal fixed partial fits incur an extra .50275 ± .00894
held-out error relative to PCA. Priors need scientific justification.

## Overlapping multi-term factors

The two terms have rank (1,1). Constraint-span cosines are 0, .7 and .98,
and the actual loading directions share that overlap. All fitted joint score
designs have rank two. Held-out total-signal errors are:

| Method | Orthogonal | Overlapping | Near overlap |
| --- | ---: | ---: | ---: |
| Competing-span estimator, joint scores | .0961 ± .0048 | .0948 ± .0037 | .1089 ± .0039 |
| Unadjusted sum of single-term fits | .0961 ± .0048 | .5009 ± .0042 | .8775 ± .0662 |
| Unconstrained lagged factors | .1373 ± .0058 | .1380 ± .0051 | .1603 ± .0055 |
| Projected PCA | .1341 ± .0059 | .1336 ± .0054 | .1344 ± .0048 |
| Known-loading joint projection | .0949 ± .0049 | .0930 ± .0038 | .0916 ± .0033 |

The independent sum is a deliberately unadjusted comparator, not the paper's
multi-term estimator. It agrees under orthogonality and double-counts overlapping
components. Mean fitted score-design condition numbers increase from 1.00 to
1.71 to 6.68. Near overlap makes individual components harder to recover than
their total signal; inspect raw `component_signal_error` separately from held-out
total-signal errors. Component errors here concern training reconstructions,
not held-out components. These results do not establish performance as the
constraint complements become exhausted or the joint design becomes singular.

Median constrained-estimator call times were approximately 1–2 ms on this run,
without per-estimator warm-up. Millisecond wall times on a shared machine do not
justify precise speed rankings.
