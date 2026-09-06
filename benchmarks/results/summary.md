# Benchmark integration results

Scores are held-out one-step MSE for forecasts and relative in-sample signal
error for factor/CP tasks. SE is the sample standard deviation divided by
the square root of successful replications. This describes Monte Carlo
variation for these fixed synthetic scenarios, not uncertainty about
performance across arbitrary datasets. Successful unconverged runs remain
in the summaries; failed runs remain in the counts. Oracle methods receive
extra information. Fit times include full calls without dedicated warm-up.

| Task | Scenario | Method | Successful / total | Unconverged | Mean score | SE | Median fit ms |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| forecast | correlated | last-value | 5 / 5 | 0 | 1.3867 | 0.0314 | 0 |
| forecast | correlated | mar-als | 5 / 5 | 0 | 1.0535 | 0.0362 | 11.8 |
| forecast | correlated | mar-mle | 5 / 5 | 0 | 1.0499 | 0.0356 | 23.2 |
| forecast | correlated | mar-projection | 5 / 5 | 0 | 1.0664 | 0.0376 | 3.43 |
| forecast | correlated | oracle-mar | 5 / 5 | 0 | 1.0437 | 0.0348 | 0 |
| forecast | correlated | training-mean | 5 / 5 | 0 | 1.4313 | 0.0743 | 0.0575 |
| forecast | correlated | var | 5 / 5 | 0 | 1.3153 | 0.0553 | 2.19 |
| forecast | correlated | var-ridge | 5 / 5 | 0 | 1.3118 | 0.0552 | 2.75 |
| forecast | correlated | zero | 5 / 5 | 0 | 1.4224 | 0.0721 | 0.0133 |
| forecast | isotropic | last-value | 5 / 5 | 0 | 1.3527 | 0.0268 | 0 |
| forecast | isotropic | mar-als | 5 / 5 | 0 | 1.0045 | 0.0172 | 9.65 |
| forecast | isotropic | mar-mle | 5 / 5 | 0 | 1.0045 | 0.0172 | 21.4 |
| forecast | isotropic | mar-projection | 5 / 5 | 0 | 1.0166 | 0.0179 | 3.54 |
| forecast | isotropic | oracle-mar | 5 / 5 | 0 | 0.9991 | 0.0177 | 0 |
| forecast | isotropic | training-mean | 5 / 5 | 0 | 1.3819 | 0.0399 | 0.0572 |
| forecast | isotropic | var | 5 / 5 | 0 | 1.2436 | 0.027 | 2.3 |
| forecast | isotropic | var-ridge | 5 / 5 | 0 | 1.2428 | 0.027 | 2.39 |
| forecast | isotropic | zero | 5 / 5 | 0 | 1.3703 | 0.0359 | 0.0128 |
| forecast | reduced-rank | last-value | 5 / 5 | 0 | 1.5091 | 0.0258 | 0 |
| forecast | reduced-rank | mar-als | 5 / 5 | 0 | 1.0027 | 0.0178 | 9.02 |
| forecast | reduced-rank | mar-mle | 5 / 5 | 0 | 1.0026 | 0.0178 | 18.4 |
| forecast | reduced-rank | mar-projection | 5 / 5 | 0 | 1.0091 | 0.018 | 3.25 |
| forecast | reduced-rank | oracle-mar | 5 / 5 | 0 | 0.9991 | 0.0177 | 0 |
| forecast | reduced-rank | rr-mar-oracle-ranks | 5 / 5 | 0 | 1.0025 | 0.0177 | 11.6 |
| forecast | reduced-rank | training-mean | 5 / 5 | 0 | 1.2424 | 0.0319 | 0.0561 |
| forecast | reduced-rank | var | 5 / 5 | 0 | 1.2246 | 0.025 | 3.55 |
| forecast | reduced-rank | var-ridge | 5 / 5 | 0 | 1.2236 | 0.025 | 2.5 |
| forecast | reduced-rank | zero | 5 / 5 | 0 | 1.2348 | 0.0276 | 0.0129 |
| matrix-cp | distinct-ar | cp-refined | 5 / 5 | 0 | 0.030474 | 0.0124 | 1.61 |
| matrix-factor | elliptical-t3 | alpha-pca | 5 / 5 | 0 | 0.057939 | 0.00522 | 3.57 |
| matrix-factor | elliptical-t3 | constrained-oracle-spans | 5 / 5 | 0 | 0.044651 | 0.00134 | 1.91 |
| matrix-factor | elliptical-t3 | huber-factor | 5 / 5 | 0 | 0.046367 | 0.00118 | 12.7 |
| matrix-factor | elliptical-t3 | itipup | 5 / 5 | 0 | 0.051276 | 0.00138 | 18.8 |
| matrix-factor | elliptical-t3 | itopup | 5 / 5 | 0 | 0.051685 | 0.0016 | 22.9 |
| matrix-factor | elliptical-t3 | lagged-factor | 5 / 5 | 0 | 0.051628 | 0.00162 | 23.4 |
| matrix-factor | elliptical-t3 | matrix-kendall | 5 / 5 | 0 | 0.045822 | 0.00128 | 761 |
| matrix-factor | elliptical-t3 | projected-pca | 5 / 5 | 0 | 0.057772 | 0.00514 | 6.56 |
| matrix-factor | elliptical-t3 | tipup | 5 / 5 | 0 | 0.051437 | 0.00141 | 4.88 |
| matrix-factor | gaussian | alpha-pca | 5 / 5 | 0 | 0.045883 | 0.000298 | 6.42 |
| matrix-factor | gaussian | constrained-oracle-spans | 5 / 5 | 0 | 0.044986 | 0.000281 | 1.76 |
| matrix-factor | gaussian | huber-factor | 5 / 5 | 0 | 0.045865 | 0.000294 | 9.56 |
| matrix-factor | gaussian | itipup | 5 / 5 | 0 | 0.048127 | 0.000368 | 17.3 |
| matrix-factor | gaussian | itopup | 5 / 5 | 0 | 0.048184 | 0.000358 | 26.9 |
| matrix-factor | gaussian | lagged-factor | 5 / 5 | 0 | 0.048146 | 0.000356 | 16.1 |
| matrix-factor | gaussian | matrix-kendall | 5 / 5 | 0 | 0.045937 | 0.00029 | 726 |
| matrix-factor | gaussian | projected-pca | 5 / 5 | 0 | 0.045861 | 0.000295 | 4.48 |
| matrix-factor | gaussian | tipup | 5 / 5 | 0 | 0.048227 | 0.000389 | 5.89 |
| matrix-factor | outliers | alpha-pca | 5 / 5 | 0 | 0.42563 | 0.0156 | 3.34 |
| matrix-factor | outliers | constrained-oracle-spans | 5 / 5 | 0 | 0.29041 | 0.00957 | 1.73 |
| matrix-factor | outliers | huber-factor | 5 / 5 | 0 | 0.29132 | 0.00957 | 10.4 |
| matrix-factor | outliers | itipup | 5 / 5 | 0 | 0.31425 | 0.00972 | 21.8 |
| matrix-factor | outliers | itopup | 5 / 5 | 0 | 0.31679 | 0.0101 | 25.7 |
| matrix-factor | outliers | lagged-factor | 5 / 5 | 0 | 0.46355 | 0.0664 | 18.4 |
| matrix-factor | outliers | matrix-kendall | 5 / 5 | 0 | 0.29072 | 0.00959 | 834 |
| matrix-factor | outliers | projected-pca | 5 / 5 | 0 | 0.3348 | 0.00938 | 4.53 |
| matrix-factor | outliers | tipup | 5 / 5 | 0 | 0.32811 | 0.0128 | 4.78 |
| tensor-factor | gaussian | itipup | 5 / 5 | 0 | 0.033068 | 0.000474 | 43.3 |
| tensor-factor | gaussian | itopup | 5 / 5 | 0 | 0.033062 | 0.000461 | 69.6 |
| tensor-factor | gaussian | tipup | 5 / 5 | 0 | 0.033077 | 0.000473 | 20.7 |
| tensor-factor | gaussian | topup | 5 / 5 | 0 | 0.033056 | 0.000461 | 61.9 |
