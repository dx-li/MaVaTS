# Matrix extensions and inference checkpoint

Generated on 2026-09-05. Raw artifacts record seeds, source hashes, software,
BLAS and requested thread settings. These are specified synthetic designs,
not full paper replications or general performance rankings.

## Paired matrix-method experiments

[matrix-extensions.json](matrix-extensions.json) retains 20 repetitions per
method, 200 runs total, with no errors or unconverged fits. Every method within
a family receives the same observations. Values below are mean ± replication
standard error; timing is retained in the raw data but is not a warmed-up
speed comparison.

| Design and estimator | Held-out score | Additional recovery evidence |
| --- | --- | --- |
| Sparse diagonal MAR, EMVS | MSE 1.0268 ± 0.0181 | Operator relative error 0.0491 ± 0.0019; exact support 20/20 |
| Same, ALS | MSE 1.0273 ± 0.0182 | Operator error 0.0593 ± 0.0019 |
| Same, separable MLE | MSE 1.0273 ± 0.0182 | Operator error 0.0591 ± 0.0019 |
| Same, unrestricted VAR | MSE 1.0313 ± 0.0179 | Operator error 0.1083 ± 0.0024 |
| Threshold factors, estimated threshold/ranks | Signal error 0.0458 ± 0.0108 | Correct unequal ranks 20/20; threshold absolute error 0.00374 ± 0.00068 |
| Same, known threshold/ranks oracle | Signal error 0.0244 ± 0.0007 | Receives extra information |
| Same, global lag factors, fixed union ranks | Signal error 0.0507 ± 0.0016 | No regime adaptation |
| Decorrelation, fixed correlation threshold + block VAR | MSE 0.9484 ± 0.0077 | Nine scalar blocks recovered 20/20 |
| Same, adjacent-ratio grouping + block VAR | MSE 0.9485 ± 0.0076 | Cannot select all singleton groups by construction |
| Same, unrestricted VAR | MSE 0.9491 ± 0.0077 | Full transition model |

The sparse design uses 1,100 training observations and 50 chronological
one-step test forecasts, with fixed spike variance .001 and other documented
defaults. Priors were not tuned on the test set. Support scores refer to
conditional inclusion masks; the fitted coefficient arrays remain continuous.
Its paired MSE difference from VAR is -0.00451 ± 0.00099, but most of that
gain is also obtained by ordinary bilinear ALS (-0.00396 ± 0.00094).

The threshold design uses 400 training and 50 held-out observations, threshold
zero and ranks (1,2)/(2,1). Its held-out score projects noisy observations,
so it is denoising, **not forecasting**. The estimated regime classification
error is 0.0050 ± 0.0025. A wrong threshold can change a held-out projection
substantially, explaining its larger replication uncertainty.

The decorrelation design uses 10,000 training observations and 100 chronological
one-step forecasts, with separate VARs fitted once to the transformed blocks.
The fixed correlation threshold is .15. Its paired MSE difference from full
VAR is only -0.00069 ± 0.00049; the ratio version's is -0.00052 ± 0.00044.
This long-training, small-dimension design establishes working reconstruction
and forecasting, not a convincing universal accuracy advantage.

## MAR interval coverage and specification size/power

[inference.json](inference.json) records the configuration, source hashes and
group summaries; [inference.jsonl](inference.jsonl) retains all 4,400 individual
experiments. There are 200 independent series for each method/design/sample
size combination and zero failed fits. Reusing each replication's seed across
methods, regimes and sample sizes makes those comparisons paired, not
independent of one another.

The following are marginal 95% operator-interval coverage rates, averaged over
the 16 entries **within each replication first**, then over replications.
Uncertainty is the Monte Carlo SE across series, not across operator entries.
Raw summaries also retain each entry's coverage and mean interval widths.

| T | Innovation covariance | Projection | ALS | Separable MLE |
| --- | --- | --- | --- | --- |
| 200 | Isotropic | .9263 ± .0064 | .9319 ± .0062 | .9313 ± .0062 |
| 200 | Separable | .9244 ± .0069 | .9363 ± .0061 | .9300 ± .0064 |
| 200 | Nonseparable | .9278 ± .0063 | .9328 ± .0058 | .9288 ± .0063* |
| 800 | Isotropic | .9450 ± .0054 | .9478 ± .0056 | .9484 ± .0055 |
| 800 | Separable | .9372 ± .0065 | .9406 ± .0061 | .9400 ± .0067 |
| 800 | Nonseparable | .9444 ± .0055 | .9403 ± .0063 | .9491 ± .0056* |

\* The MLE covariance assumption is violated in this row. A favorable realized
coverage value does not validate the procedure under misspecification.

There is finite-sample undercoverage, especially at T=200. Larger-sample rates
are generally nearer .95, but individual entries can differ materially from
the average: T=800 separable MLE ranges from .880 to .985 in these 200 series.
These experiments do not justify exact 95% calibration or simultaneous bands.

At nominal 5%, the Kronecker specification test rejects 14/200 true-null
series at each sample size: estimated size .070, Monte Carlo SE .0181,
Wilson 95% interval [.0422,.1141]. The chosen non-Kronecker diagonal alternative
is rejected in 200/200 series at both sizes: estimated power 1, Wilson interval
[.9812,1]. This is a strong fixed alternative, not a local-power curve. The
two sample-size results are paired and should not be pooled as 400 independent
replications.

Remaining evidence includes larger spatial dimensions, weak/zero coefficients,
near instability, heavier tails, dependent innovations, prior sensitivity,
multi-step forecasts, real datasets, and paper-specific simulation replications.
