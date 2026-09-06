# Published factor ranks and explicit penalty stability

Retained on 2026-09-05 from source checkpoint
`10d8b0500eff1f1c8064cf48740aef1f8c8c280e`:
[rank comparisons](factor-ranks.json) and [stability paths](rank-stability.json).
Both artifacts' complete source fingerprints match that checkpoint, with no
source edits detected during execution. The [protocol](../README.md#published-factor-rank-criteria-and-penalty-stability)
specifies all data, options and comparisons. The implementation targets
[Han, Chen and Zhang (2022)](https://doi.org/10.1214/22-EJS1991), using the
accessible author manuscript v3 and documented finite-grid extensions.

There are 3,260 rank-comparison records and 80 stability-path records. The
latter contain 6,000 fitted subsample cells, 80 fixed-c=1 comparisons and 52
joint refits; these nested fits are not additional independent experiments.
Each method/design uses ten independent series, seeds 2573–2582, with paired
observations across methods. There are no execution errors or captured method
warnings. The rank study retains 137 unconverged fits; stability retains 47
unconverged cells and 28 incomplete modewise choices. All 52 selected joint
refits and all 80 fixed-c comparisons converged. No unavailable choice was
replaced by a fixed constant or a grid endpoint.

## Exact rank recovery

Entries are correct raw mode-rank tuples out of ten, including unconverged
returned fits. `i` means ranks and spaces are updated sequentially; `adj-iTIPUP`
initializes by the old adjacent ratio then holds ranks fixed. IC2 uses nu=0,c=1;
ER1 uses c=1,c0=.1. All five penalties for each criterion are retained in the
raw artifact; this table selects representative variants, not the best result
per design.

| Design / h0 | IC2 TOPUP | IC2 iTOPUP | IC2 TIPUP | IC2 iTIPUP | ER1 TOPUP | ER1 iTOPUP | ER1 TIPUP | ER1 iTIPUP | adj-iTIPUP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Strong / 1 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 |
| Weak / 1 | 0 | 0 | 4 | 4 | 10 | 10 | 10 | 10 | 10 |
| Cancellation / 1 | 10 | 10 | 0 | 0 | 10 | 10 | 5 | 7 | 5 |
| Cancellation / 2 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 |
| White factors / 1 | 0 | 0 | 0 | 0 | 10 | 10 | 4 | 4 | 4 |
| No factors / 1 | 10 | 10 | 10 | 10 | 0 | 0 | 0 | 0 | 0 |
| Serial noise / 1 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 | 10 |

Ten replications give coarse uncertainty: a 10/10 rate has a 95% Wilson
interval of approximately [0.722,1], and 0/10 has [0,0.278]. These are
independent-series intervals, not pooled across dependent methods or modes;
see [Wilson (1927)](https://doi.org/10.1080/01621459.1927.10502953).
The table is not evidence of universal consistency. In particular, white
factors have zero population positive-lag signal moments and serial noise
violates the white-error premise; favorable finite-sample rank counts there
do not establish the paper's assumptions. ER's positive search cannot return
zero, while IC zero selection is not a calibrated null test.

## Held-out clean-signal error

Entrywise MSE, mean ± independent-series Monte Carlo SE, computed as sample
standard deviation divided by sqrt(10). All finite returned fits are included,
whether converged or not. Current held-out noisy observations are projected
using training spaces and mean: this is denoising, not forecasting.

| Design / h0 | IC2 iTOPUP | IC2 iTIPUP | ER1 iTIPUP | Known-loading oracle | Training mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| Strong / 1 | 0.0398 ± 0.0016 | 0.0397 ± 0.0016 | 0.0397 ± 0.0016 | 0.0367 ± 0.0015 | 4.2803 ± 0.3661 |
| Weak / 1 | 0.6339 ± 0.0610 | 0.4011 ± 0.1087 | 0.0397 ± 0.0016 | 0.0367 ± 0.0015 | 1.6613 ± 0.1255 |
| Cancellation / 1 | 0.0389 ± 0.0015 | 3.6252 ± 0.3444 | 1.1441 ± 0.4003 | 0.0367 ± 0.0015 | 3.8883 ± 0.2833 |
| Cancellation / 2 | 0.0394 ± 0.0015 | 0.0427 ± 0.0016 | 0.0427 ± 0.0016 | 0.0367 ± 0.0015 | 3.8883 ± 0.2833 |
| White factors / 1 | 4.1258 ± 0.1995 | 4.1258 ± 0.1995 | 1.5266 ± 0.4188 | 0.0367 ± 0.0015 | 4.1258 ± 0.1995 |
| No factors / 1 | 0.0033 ± 0.0002 | 0.0033 ± 0.0002 | 0.0477 ± 0.0123 | 0.0033 ± 0.0002 | 0.0033 ± 0.0002 |
| Serial noise / 1 | 0.0472 ± 0.0016 | 0.0470 ± 0.0015 | 0.0470 ± 0.0015 | 0.0412 ± 0.0016 | 4.2855 ± 0.3657 |

The oracle receives true loading spaces, not the true mean. Thus even its
no-factor fit has nonzero error from the estimated training mean. It is an
extra-information comparator, not a universal lower bound. Relative error is
omitted for zero clean signals. Matching rank alone need not give accurate
spaces, as the cancellation errors illustrate.

Supplying the true weak-factor exponent nu=.5 to IC2 changes its penalty and
gives additional information. TOPUP/iTOPUP still recover the exact ranks in
0/10 runs; their MSEs are 0.1410 ± 0.0024 and 0.1340 ± 0.0064. TIPUP/iTIPUP
recover 10/10, with MSEs 0.0401 ± 0.0016 and 0.0397 ± 0.0016. Two oracle-strength
iTOPUP fits remain unconverged. This experiment does not implement strength
estimation or prove that the theoretically motivated exponent alone supplies
good finite-sample tuning for every moment family.

The 137 unconverged rank-comparison fits comprise 107 no-factor, 16 lag-one
cancellation, 12 white-factor and two weak oracle-strength fits. None occur
in strong, lag-two cancellation or serial-noise comparisons. These iteration
flags are distinct from rank-recovery errors and are preserved in raw results.

## Stability tuning and missing choices

Each path has 25 constants × three nested subsamples. The explicit rule requires
zero empirical rank variance over at least two gridpoints, excludes cap/zero
plateaus and unconverged cells, then uses the first admissible interval's
lower-middle point per mode. The rule assumes positive ranks and may fail or
select an incorrect rank. It is not the global minimum of variance, which can
be attained by degenerate plateaus.

| Design | Method | Complete choices / 10 | Correct joint refits / complete | Fixed-c correct / 10 | Unconverged cells / 750 |
| --- | --- | ---: | ---: | ---: | ---: |
| Strong | TIPUP | 10 | 10/10 | 10 | 0 |
| Strong | iTIPUP | 10 | 10/10 | 10 | 0 |
| Weak | TIPUP | 10 | 10/10 | 4 | 0 |
| Weak | iTIPUP | 10 | 10/10 | 4 | 0 |
| Cancellation | TIPUP | 9 | 8/9 | 0 | 0 |
| Cancellation | iTIPUP | 3 | 3/3 | 0 | 47 |
| No factors | TIPUP | 0 | — | 10 | 0 |
| No factors | iTIPUP | 0 | — | 10 | 0 |

All selected joint refits match their modewise path ranks in these seeds;
the API does not promise this for coupled iterative updates. On weak data,
tuning-minus-fixed-c paired MSE changes are −0.2968 ± 0.0965 for TIPUP and
−0.3614 ± 0.1076 for iTIPUP. All ten series have choices here. Cancellation
has missing choices and unstable loading information, so reporting only its
selected-fit error would hide non-selection: correct selected ranks are only
8/10 and 3/10 of all attempted paths, respectively. No-factor absence is the
expected boundary of this positive-rank convenience rule, not a path execution
error; the fixed-c IC procedure selects zero in all those runs.

## Scope and reproducibility limits

Environment: Python 3.13.5, NumPy 2.5.2, SciPy 1.18.1; all reported BLAS thread
controls are one. Exact machine/BLAS metadata and per-fit/path times are in
the artifacts. Runs overlapped other local verification jobs and were not
warm-up-isolated; timings support execution/profiling, not a speed ranking.
Path and refit times are separate, and end-to-end tuning comparisons must
include both. Full API paths retain fitted arrays and are not a streaming
solution for arbitrarily dense grids.

These are small synthetic integration studies at one non-quick size, not the
paper's complete experiments. Broader sample/dimension/strength grids,
independent subset/grid sensitivity, nonzero means, real data, loading inference
and calibrated ER rank-zero extensions remain open. Earlier 10,410 records
retain their historical fingerprints unchanged. Including these 3,340 new
outer records gives 13,750 retained outer records; neither that count nor the
6,000 nested path cells measures literature coverage or independent sample size.
