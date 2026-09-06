# Envelope MAR: fixed dimensions and initialization sensitivity

Retained on 2026-09-05. The [final raw study](envelope.json) fingerprints
source `dbbd0b0af215ba99bce6df2e7d2e7f973caf17e5`; the
[initial-start audit](envelope-initial-starts.json) fingerprints
`6b3f4d87fdd8add8d352dd437d7f11044ba865f1`. All 34 source-file hashes in
each artifact match its checkpoint, and neither run detected source edits.
The initial artifact was renamed after execution without changing its bytes.

The [protocol](../README.md#envelope-mar) specifies data, options and scoring.
The estimator follows [Samadi & De Alwis (2026)](https://doi.org/10.1080/07350015.2025.2537404),
with [explicit numerical and source conventions](../../docs/envelope-notes.md).
These are controlled development experiments, not replications of paper tables.

Each artifact contains 720 fit records: nine methods on 80 series, ten seeds
per order/regime cell. The rerun uses the **same** series as the initialization
audit; the combined 1,440 records are not independent datasets. All 720 records
in each artifact completed and converged, with no captured method warnings.
All EMAR warmups converged too. Successful convergence is not a certificate
of the best likelihood solution or good statistical performance.

## One-step conditional-mean forecast error

Mean entrywise MSE ± Monte Carlo SE across ten independent series per cell;
SE is sample standard deviation divided by sqrt(10). Forecast origins and
methods within a series are dependent, not additional independent repetitions.
EMAR nominal/under/over/full use dimensions `(2,2)/(1,2)/(3,4)/(4,5)`.
All methods receive the true lag order. Nominal dimensions and RR rank bounds
are supplied extra information, not data-selected estimates. The known-parameter
oracle has identically zero conditional-mean error and is omitted from this table.

| Regime / order | EMAR nominal | EMAR under | EMAR over | EMAR full | MAR MLE | MAR ALS | RR MAR ALS | VAR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| high-immaterial / 1 | 0.0027 ± 0.0002 | 0.0047 ± 0.0005 | 0.0039 ± 0.0002 | 0.0041 ± 0.0002 | 0.0041 ± 0.0002 | 0.0195 ± 0.0019 | 0.0171 ± 0.0023 | 0.0611 ± 0.0010 |
| high-material / 1 | 0.0098 ± 0.0007 | 0.0456 ± 0.0159 | 0.0112 ± 0.0009 | 0.0115 ± 0.0010 | 0.0115 ± 0.0010 | 0.0148 ± 0.0012 | 0.0134 ± 0.0011 | 0.0640 ± 0.0027 |
| isotropic / 1 | 0.0080 ± 0.0008 | 0.0124 ± 0.0015 | 0.0089 ± 0.0007 | 0.0092 ± 0.0007 | 0.0092 ± 0.0007 | 0.0093 ± 0.0007 | 0.0077 ± 0.0007 | 0.0633 ± 0.0025 |
| covariance-coupled / 1 | 0.0029 ± 0.0002 | 0.0049 ± 0.0005 | 0.0037 ± 0.0002 | 0.0040 ± 0.0003 | 0.0040 ± 0.0003 | 0.0195 ± 0.0021 | 0.0213 ± 0.0019 | 0.0610 ± 0.0015 |
| high-immaterial / 2 | 0.0026 ± 0.0003 | 0.0040 ± 0.0004 | 0.0055 ± 0.0005 | 0.0077 ± 0.0009 | 0.0077 ± 0.0009 | 0.0409 ± 0.0021 | 0.0375 ± 0.0023 | 0.1460 ± 0.0069 |
| high-material / 2 | 0.0243 ± 0.0027 | 0.0486 ± 0.0051 | 0.0260 ± 0.0024 | 0.0270 ± 0.0024 | 0.0270 ± 0.0024 | 0.0356 ± 0.0036 | 0.0337 ± 0.0034 | 0.1482 ± 0.0086 |
| isotropic / 2 | 0.0145 ± 0.0012 | 0.0178 ± 0.0015 | 0.0202 ± 0.0012 | 0.0216 ± 0.0012 | 0.0216 ± 0.0012 | 0.0216 ± 0.0011 | 0.0178 ± 0.0013 | 0.1402 ± 0.0040 |
| covariance-coupled / 2 | 0.0027 ± 0.0003 | 0.0040 ± 0.0005 | 0.0055 ± 0.0004 | 0.0086 ± 0.0020 | 0.0086 ± 0.0020 | 0.0416 ± 0.0028 | 0.0381 ± 0.0026 | 0.1449 ± 0.0077 |

Underfitting remains harmful, notably with high material variance. Nominal EMAR
has smaller mean one-step error than MAR-MLE in these cells, but not uniformly
than every comparator: first-order isotropic RR ALS is slightly better here.
The covariance-coupled design violates the nominal reducing-space assumption;
favorable errors there do not verify that assumption. VAR pays for a much larger
unrestricted coefficient model at these sample sizes.

## Paired differences from MAR-MLE

EMAR nominal minus MAR-MLE, mean ± paired-series Monte Carlo SE. Negative is
better for EMAR. The first three columns compare true conditional means; the
last compares noisy held-out observations. Every score uses the same 31 origins
within a series. The raw artifact retains observed errors and analytic forecast
noise floors at all three horizons for every method.

| Regime / order | Conditional h=1 | Conditional h=3 | Conditional h=5 | Observed h=1 |
| --- | ---: | ---: | ---: | ---: |
| high-immaterial / 1 | -0.00144 ± 0.00011 | -2.19e-05 ± 5.2e-06 | -1.24e-06 ± 1.4e-06 | -0.000657 ± 0.00085 |
| high-material / 1 | -0.00169 ± 0.00051 | -2.51e-05 ± 2.9e-05 | 4.74e-06 ± 9.9e-06 | -0.00128 ± 0.00082 |
| isotropic / 1 | -0.00127 ± 0.00035 | -1.7e-05 ± 2.3e-05 | -9.19e-06 ± 7.4e-06 | -0.00128 ± 0.00095 |
| covariance-coupled / 1 | -0.00107 ± 0.00016 | -1.21e-05 ± 1.4e-05 | 1.77e-07 ± 3.3e-06 | -0.00131 ± 0.00054 |
| high-immaterial / 2 | -0.00504 ± 0.00091 | -0.000368 ± 0.00012 | -1.34e-05 ± 6.8e-06 | -0.00335 ± 0.0022 |
| high-material / 2 | -0.00277 ± 0.0027 | -0.000232 ± 0.00025 | -1.68e-05 ± 2.5e-05 | -0.000553 ± 0.0037 |
| isotropic / 2 | -0.00707 ± 0.00058 | -0.00019 ± 0.0001 | -3.34e-06 ± 1.1e-05 | -0.00289 ± 0.0025 |
| covariance-coupled / 2 | -0.00589 ± 0.002 | -0.000411 ± 0.0002 | -1.54e-05 ± 8.2e-06 | -0.00529 ± 0.0035 |

Several observed-error differences and long-horizon differences are small
relative to Monte Carlo uncertainty. These ten-seed comparisons do not establish
universal superiority, asymptotic efficiency or reliable inference after tuning.

## Identified operator and covariance errors

Relative Frobenius errors, mean ± Monte Carlo SE. Operator error pools all lag
Kronecker matrices; covariance error uses the full Kronecker covariance. Raw
factor differences would depend on arbitrary scales and are not used.

| Regime / order | Operator EMAR nominal | Operator MAR-MLE | Covariance EMAR nominal | Covariance MAR-MLE |
| --- | ---: | ---: | ---: | ---: |
| high-immaterial / 1 | 0.1733 ± 0.0219 | 0.4622 ± 0.0313 | 0.0781 ± 0.0055 | 0.0785 ± 0.0054 |
| high-material / 1 | 0.6026 ± 0.0751 | 0.6588 ± 0.0635 | 0.0659 ± 0.0079 | 0.0671 ± 0.0081 |
| isotropic / 1 | 0.3737 ± 0.0302 | 0.4307 ± 0.0271 | 0.0730 ± 0.0044 | 0.0869 ± 0.0040 |
| covariance-coupled / 1 | 0.2541 ± 0.0167 | 0.4312 ± 0.0374 | 0.0774 ± 0.0054 | 0.0779 ± 0.0055 |
| high-immaterial / 2 | 0.3538 ± 0.0328 | 1.1365 ± 0.1255 | 0.0699 ± 0.0029 | 0.0689 ± 0.0034 |
| high-material / 2 | 1.3577 ± 0.1239 | 1.6503 ± 0.1348 | 0.0551 ± 0.0048 | 0.0550 ± 0.0048 |
| isotropic / 2 | 0.7085 ± 0.0331 | 0.9190 ± 0.0254 | 0.0668 ± 0.0029 | 0.0816 ± 0.0022 |
| covariance-coupled / 2 | 0.3706 ± 0.0178 | 1.1925 ± 0.1952 | 0.0700 ± 0.0029 | 0.0694 ± 0.0037 |

## Initialization audit: keep the unfavorable result

The first implementation used covariance eigenvector/random starts but did not
also seed the envelope with the unrestricted warmup's coefficient spaces.
Despite all fits converging, high-material-variance nominal EMAR forecasts were
often worse than MAR-MLE. Six training-only follow-up fits using coefficient
spaces found substantially higher likelihoods in five cases, motivating the
additional default start. No held-out outcomes enter optimization or choose a
start; starts are compared by the training profile likelihood.

The rerun is nevertheless a **development sensitivity audit on reused seeds**,
not an untouched validation set. Both full artifacts are retained. In the
high-material regime:

| Order / method | Initial h=1 conditional MSE | Updated h=1 conditional MSE | Updated minus initial log likelihood |
| --- | ---: | ---: | ---: |
| 1 / EMAR nominal | 0.0476 ± 0.0117 | 0.0098 ± 0.0007 | 41.5 ± 11 |
| 1 / EMAR under | 0.0834 ± 0.0186 | 0.0456 ± 0.0159 | 69 ± 27 |
| 1 / EMAR over | 0.0125 ± 0.0018 | 0.0112 ± 0.0009 | 3.41 ± 3.6 |
| 2 / EMAR nominal | 0.0433 ± 0.0085 | 0.0243 ± 0.0027 | 37.6 ± 25 |
| 2 / EMAR under | 0.0580 ± 0.0074 | 0.0486 ± 0.0051 | 24.5 ± 7.6 |
| 2 / EMAR over | 0.0255 ± 0.0025 | 0.0260 ± 0.0024 | 0.836 ± 0.54 |

Extra starts do not certify a global solution or uniformly improve forecasts:
the second-order overfit case illustrates the latter. Nominal second-order
training likelihood gains range from -2.4e-5 (numerical stopping/path variation)
to 261.5. Other regimes' nominal solutions are nearly unchanged. More starts,
larger dimension/sample grids and genuinely new validation seeds remain useful.

## Runtime and reproducibility

Final-study fit seconds, median and maximum across 80 runs per method. Timing
includes EMAR warmup and all optimization, but excludes simulation and scoring.
All methods completed/converged 80/80. The known-parameter oracle is not fitted,
so its constructor timing is not a meaningful speed comparator.

| Method | Median seconds | Maximum seconds |
| --- | ---: | ---: |
| EMAR nominal | 0.5162 | 3.2278 |
| EMAR under | 0.4643 | 2.6067 |
| EMAR over | 0.6907 | 11.1010 |
| EMAR full | 0.0171 | 0.0962 |
| MAR MLE | 0.0137 | 0.0924 |
| MAR ALS | 0.0119 | 0.0672 |
| RR MAR ALS | 0.0167 | 0.2056 |
| VAR | 0.0005 | 0.0015 |

Both final selected gradient maxima are below 1e-6. Runs use Python 3.13.5,
NumPy 2.5.2 and SciPy 1.18.1 on the recorded macOS host, with all four BLAS/OpenMP
thread environment variables set to one. Full platform/BLAS details are in each
JSON artifact. These are development-host timings, not isolated hardware
microbenchmarks, large-dimension scaling evidence or a claim that EMAR is faster
than unrestricted methods on these small matrices.
