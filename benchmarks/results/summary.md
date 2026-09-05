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
| forecast | correlated | mar-als | 5 / 5 | 0 | 1.0535 | 0.0362 | 10.6 |
| forecast | correlated | mar-mle | 5 / 5 | 0 | 1.0499 | 0.0356 | 19.4 |
| forecast | correlated | mar-projection | 5 / 5 | 0 | 1.0664 | 0.0376 | 3.3 |
| forecast | correlated | oracle-mar | 5 / 5 | 0 | 1.0437 | 0.0348 | 0 |
| forecast | correlated | training-mean | 5 / 5 | 0 | 1.4313 | 0.0743 | 0.0561 |
| forecast | correlated | var | 5 / 5 | 0 | 1.3153 | 0.0553 | 2.25 |
| forecast | correlated | var-ridge | 5 / 5 | 0 | 1.3118 | 0.0552 | 2.32 |
| forecast | correlated | zero | 5 / 5 | 0 | 1.4224 | 0.0721 | 0.0123 |
| forecast | isotropic | last-value | 5 / 5 | 0 | 1.3527 | 0.0268 | 0 |
| forecast | isotropic | mar-als | 5 / 5 | 0 | 1.0045 | 0.0172 | 8.77 |
| forecast | isotropic | mar-mle | 5 / 5 | 0 | 1.0045 | 0.0172 | 16.2 |
| forecast | isotropic | mar-projection | 5 / 5 | 0 | 1.0166 | 0.0179 | 3.39 |
| forecast | isotropic | oracle-mar | 5 / 5 | 0 | 0.9991 | 0.0177 | 0 |
| forecast | isotropic | training-mean | 5 / 5 | 0 | 1.3819 | 0.0399 | 0.0563 |
| forecast | isotropic | var | 5 / 5 | 0 | 1.2436 | 0.027 | 2.26 |
| forecast | isotropic | var-ridge | 5 / 5 | 0 | 1.2428 | 0.027 | 2.4 |
| forecast | isotropic | zero | 5 / 5 | 0 | 1.3703 | 0.0359 | 0.013 |
| forecast | reduced-rank | last-value | 5 / 5 | 0 | 1.5091 | 0.0258 | 0 |
| forecast | reduced-rank | mar-als | 5 / 5 | 0 | 1.0027 | 0.0178 | 8.74 |
| forecast | reduced-rank | mar-mle | 5 / 5 | 0 | 1.0026 | 0.0178 | 16.3 |
| forecast | reduced-rank | mar-projection | 5 / 5 | 0 | 1.0091 | 0.018 | 3.4 |
| forecast | reduced-rank | oracle-mar | 5 / 5 | 0 | 0.9991 | 0.0177 | 0 |
| forecast | reduced-rank | rr-mar-oracle-ranks | 5 / 5 | 0 | 1.0025 | 0.0177 | 12.6 |
| forecast | reduced-rank | training-mean | 5 / 5 | 0 | 1.2424 | 0.0319 | 0.0565 |
| forecast | reduced-rank | var | 5 / 5 | 0 | 1.2246 | 0.025 | 2.26 |
| forecast | reduced-rank | var-ridge | 5 / 5 | 0 | 1.2236 | 0.025 | 2.55 |
| forecast | reduced-rank | zero | 5 / 5 | 0 | 1.2348 | 0.0276 | 0.013 |
| matrix-cp | distinct-ar | cp-refined | 5 / 5 | 0 | 0.030474 | 0.0124 | 1.56 |
| matrix-factor | elliptical-t3 | alpha-pca | 5 / 5 | 0 | 0.057939 | 0.00522 | 3.32 |
| matrix-factor | elliptical-t3 | constrained-oracle-spans | 5 / 5 | 0 | 0.044651 | 0.00134 | 1.67 |
| matrix-factor | elliptical-t3 | huber-factor | 5 / 5 | 0 | 0.046367 | 0.00118 | 8.65 |
| matrix-factor | elliptical-t3 | itipup | 5 / 5 | 0 | 0.051276 | 0.00138 | 14.8 |
| matrix-factor | elliptical-t3 | itopup | 5 / 5 | 0 | 0.051685 | 0.0016 | 15.8 |
| matrix-factor | elliptical-t3 | lagged-factor | 5 / 5 | 0 | 0.051628 | 0.00162 | 13.2 |
| matrix-factor | elliptical-t3 | matrix-kendall | 5 / 5 | 0 | 0.045822 | 0.00128 | 629 |
| matrix-factor | elliptical-t3 | projected-pca | 5 / 5 | 0 | 0.057772 | 0.00514 | 4.41 |
| matrix-factor | elliptical-t3 | tipup | 5 / 5 | 0 | 0.051437 | 0.00141 | 4.54 |
| matrix-factor | gaussian | alpha-pca | 5 / 5 | 0 | 0.045883 | 0.000298 | 3.45 |
| matrix-factor | gaussian | constrained-oracle-spans | 5 / 5 | 0 | 0.044986 | 0.000281 | 1.67 |
| matrix-factor | gaussian | huber-factor | 5 / 5 | 0 | 0.045865 | 0.000294 | 7.5 |
| matrix-factor | gaussian | itipup | 5 / 5 | 0 | 0.048127 | 0.000368 | 12.6 |
| matrix-factor | gaussian | itopup | 5 / 5 | 0 | 0.048184 | 0.000358 | 16.4 |
| matrix-factor | gaussian | lagged-factor | 5 / 5 | 0 | 0.048146 | 0.000356 | 13.1 |
| matrix-factor | gaussian | matrix-kendall | 5 / 5 | 0 | 0.045937 | 0.00029 | 616 |
| matrix-factor | gaussian | projected-pca | 5 / 5 | 0 | 0.045861 | 0.000295 | 4.42 |
| matrix-factor | gaussian | tipup | 5 / 5 | 0 | 0.048227 | 0.000389 | 4.55 |
| matrix-factor | outliers | alpha-pca | 5 / 5 | 0 | 0.42563 | 0.0156 | 3.33 |
| matrix-factor | outliers | constrained-oracle-spans | 5 / 5 | 0 | 0.29041 | 0.00957 | 1.74 |
| matrix-factor | outliers | huber-factor | 5 / 5 | 0 | 0.29132 | 0.00957 | 8.8 |
| matrix-factor | outliers | itipup | 5 / 5 | 0 | 0.31425 | 0.00972 | 20.9 |
| matrix-factor | outliers | itopup | 5 / 5 | 0 | 0.31679 | 0.0101 | 17.7 |
| matrix-factor | outliers | lagged-factor | 5 / 5 | 0 | 0.46355 | 0.0664 | 13.1 |
| matrix-factor | outliers | matrix-kendall | 5 / 5 | 0 | 0.29072 | 0.00959 | 621 |
| matrix-factor | outliers | projected-pca | 5 / 5 | 0 | 0.3348 | 0.00938 | 4.34 |
| matrix-factor | outliers | tipup | 5 / 5 | 0 | 0.32811 | 0.0128 | 4.52 |
| tensor-factor | gaussian | itipup | 5 / 5 | 0 | 0.033068 | 0.000474 | 37.3 |
| tensor-factor | gaussian | itopup | 5 / 5 | 0 | 0.033062 | 0.000461 | 48.6 |
| tensor-factor | gaussian | tipup | 5 / 5 | 0 | 0.033077 | 0.000473 | 16.3 |
| tensor-factor | gaussian | topup | 5 / 5 | 0 | 0.033056 | 0.000461 | 38.7 |
