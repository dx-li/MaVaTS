# Method-to-paper citation index

Every public scientific procedure has an explicit paper citation in its API
docstring. This index also covers legacy functions, baselines, evaluation metrics,
simulation utilities, result classes and their methods. Top-level package
re-exports use the canonical module names below.

A citation is attribution, not a claim that every option reproduces a paper
exactly. The scope column distinguishes published estimators, extensions and
conventional utilities. Pure result containers, exceptions, properties, state
serialization and forecast/transform methods inherit their model's references;
they are not attributed as independent published methods. A shared factor result
must additionally be cited through the estimator that created it. Private
validation, algebra and benchmark orchestration are implementation infrastructure,
not additional scientific estimators.

For method variants sharing an entry point, cite the applicable paper(s) stated
in the scope column and docstring. Exact assumptions, manuscript versions,
algorithms, deviations and missing inference are in [methods.md](methods.md)
and the linked method notes. [references.bib](references.bib) supplies BibTeX.
Tests check that every public package function/class is indexed, every index key
has a bibliographic entry, and every public scientific function has a References
section with a primary-paper link. New APIs must update this index.

## Public API coverage

| Canonical API | Paper key(s) | Citation scope |
| --- | --- | --- |
| `mavats.MAR.estimate_mar1` | [chen2021mar](#chen2021mar) | Legacy projection/LS/MLE MAR wrapper; normalization and option compatibility are documented. |
| `mavats.MAR.estimate_residual_cov` | [chen2021mar](#chen2021mar) | Full centered sample covariance of vectorized MAR residuals, with ddof=1; not separable covariance MLE. |
| `mavats.autoregression.nearest_kronecker_product` | [chen2021mar](#chen2021mar) | Rearranged rank-one SVD/Kronecker projection, Section 3.1; no reduced-rank regression or ridge option. |
| `mavats.autoregression.MARResult` | [chen2021mar](#chen2021mar), [xiao2022rrmar](#xiao2022rrmar) | Projection/LS/MLE; reduced-rank LS uses the second reference. Multi-lag/intercept/ridge options and safeguards are documented extensions. |
| `mavats.autoregression.fit_mar` | [chen2021mar](#chen2021mar), [xiao2022rrmar](#xiao2022rrmar) | Projection/LS/MLE; reduced-rank LS uses the second reference. Multi-lag/intercept/ridge options and safeguards are documented extensions. |
| `mavats.autoregression.MARRankSelection` | [xiao2022rrmar](#xiao2022rrmar) | Joint RR.LS EBIC; not the distinct RR.CC estimator. |
| `mavats.autoregression.select_mar_rank` | [xiao2022rrmar](#xiao2022rrmar) | Joint RR.LS EBIC; not the distinct RR.CC estimator. |
| `mavats.baselines.VARResult` | [chen2021mar](#chen2021mar), [hoerl1970ridge](#hoerl1970ridge) | Conventional vector baseline and optional quadratic ridge penalty; not a new matrix estimator. |
| `mavats.baselines.fit_var` | [chen2021mar](#chen2021mar), [hoerl1970ridge](#hoerl1970ridge) | Conventional vector baseline and optional quadratic ridge penalty; not a new matrix estimator. |
| `mavats.baselines.NaiveResult` | [hyndman2006accuracy](#hyndman2006accuracy) | Random-walk and historical-mean benchmarks; tensor entrywise use and zero forecast are elementary comparison conventions. |
| `mavats.baselines.fit_naive` | [hyndman2006accuracy](#hyndman2006accuracy) | Random-walk and historical-mean benchmarks; tensor entrywise use and zero forecast are elementary comparison conventions. |
| `mavats.calibration.MonitorCalibration` | [he2024breaks](#he2024breaks) | Paper monitoring statistic; finite-horizon Gaussian-reference calibration is a package extension, not the paper's exact finite-sample guarantee. |
| `mavats.calibration.calibrate_monitor` | [he2024breaks](#he2024breaks) | Paper monitoring statistic; finite-horizon Gaussian-reference calibration is a package extension, not the paper's exact finite-sample guarantee. |
| `mavats.cointegration.CMAROptimizationRun` | [li2024cointegration](#li2024cointegration) | 2024 manuscript; fixed ranks, full-complement numerical I(1) checks, explicit manuscript corrections; no rank test. |
| `mavats.cointegration.CMARI1Diagnostics` | [li2024cointegration](#li2024cointegration) | 2024 manuscript; fixed ranks, full-complement numerical I(1) checks, explicit manuscript corrections; no rank test. |
| `mavats.cointegration.cmar_i1_diagnostics` | [li2024cointegration](#li2024cointegration) | 2024 manuscript; fixed ranks, full-complement numerical I(1) checks, explicit manuscript corrections; no rank test. |
| `mavats.cointegration.CMARResult` | [li2024cointegration](#li2024cointegration) | 2024 manuscript; fixed ranks, full-complement numerical I(1) checks, explicit manuscript corrections; no rank test. |
| `mavats.cointegration.fit_cmar` | [li2024cointegration](#li2024cointegration) | 2024 manuscript; fixed ranks, full-complement numerical I(1) checks, explicit manuscript corrections; no rank test. |
| `mavats.constrained.fit_constrained_factor` | [chen2020constrained](#chen2020constrained) | Single-term fully constrained branch only. |
| `mavats.constrained.PartialConstrainedFactorResult` | [chen2020constrained](#chen2020constrained) | Shared four-block partial factor model and optional zero cross blocks; accessible manuscript v3 Sections 3.3–3.4, with journal-version boundary documented. |
| `mavats.constrained.fit_partial_constrained_factor` | [chen2020constrained](#chen2020constrained) | Separate block lag moments summed for shared groups; explicit/heuristic ranks, not calibrated no-factor tests. |
| `mavats.constrained.MultiTermConstrainedFactorResult` | [chen2020constrained](#chen2020constrained) | Orthogonal/overlapping term spaces and joint score reconstruction; joint LS and more-than-two-term annihilation are explicit extensions. |
| `mavats.constrained.fit_multiterm_constrained_factor` | [chen2020constrained](#chen2020constrained) | Section 3.3/Remark 3 direct or competing-complement loading estimation, with projection-survival and joint-design diagnostics. |
| `mavats.marma.MARMAParameters` | [tsay2024marma](#tsay2024marma) | Rank-one-per-lag minus-MA model, normalization and covariance conventions. |
| `mavats.marma.MARMADiagnostics` | [tsay2024marma](#tsay2024marma) | Full-polynomial stability/invertibility; does not certify left coprimeness or minimal orders. |
| `mavats.marma.marma_diagnostics` | [tsay2024marma](#tsay2024marma) | Numerical companion-root checks for the complete AR and MA polynomials. |
| `mavats.marma.MARMAFilterResult` | [tsay2024marma](#tsay2024marma) | Conditional innovations, fixed-parameter forecasts and impulse-response covariance; no parameter/presample uncertainty. |
| `mavats.marma.filter_marma` | [tsay2024marma](#tsay2024marma) | First max(p,q) observations conditioned on, their innovations fixed zero; not exact stationary/Kalman filtering. |
| `mavats.marma.MARMAOptimizationRun` | [tsay2024marma](#tsay2024marma) | Local numerical solve and explicit initialization diagnostics, not a global optimization certificate. |
| `mavats.marma.MARMAResult` | [tsay2024marma](#tsay2024marma) | Fitted conditional model and result operations inherit its paper and implementation scope. |
| `mavats.marma.fit_marma` | [tsay2024marma](#tsay2024marma) | Rank-one conditional LS/Gaussian MLE, complete innovation recursion and explicit numerical optimizer; corrected Gaussian one-half factor. |
| `mavats.marma.MARMASimulation` | [tsay2024marma](#tsay2024marma) | Model simulation outputs; burn-in is not an exact stationary draw. |
| `mavats.marma.simulate_marma` | [tsay2024marma](#tsay2024marma) | Supplied-parameter MARMA simulation, not an exact paper experiment. |
| `mavats.cp.CPIdentificationError` | [chang2023cp](#chang2023cp) | Unthresholded refined CP estimator and identification diagnostics. |
| `mavats.cp.CPResult` | [chang2023cp](#chang2023cp) | Unthresholded refined CP estimator and identification diagnostics. |
| `mavats.cp.fit_cp_factor` | [chang2023cp](#chang2023cp) | Unthresholded refined CP estimator and identification diagnostics. |
| `mavats.decorrelation.MatrixDecorrelationResult` | [han2024decorrelation](#han2024decorrelation) | Threshold/ratio grouping; supplement boundary choices documented. |
| `mavats.decorrelation.fit_matrix_decorrelation` | [han2024decorrelation](#han2024decorrelation) | Threshold/ratio grouping; supplement boundary choices documented. |
| `mavats.dynamic.TwoWayDynamicResult` | [yuan2023dynamic](#yuan2023dynamic) | Additive two-way model, residual rank iteration and scoped latent-factor forecasting. |
| `mavats.dynamic.fit_two_way_dynamic` | [yuan2023dynamic](#yuan2023dynamic) | Additive two-way model, residual rank iteration and scoped latent-factor forecasting. |
| `mavats.envelope.EnvelopeMARResult` | [samadi2026envelope](#samadi2026envelope) | Fixed-dimension Gaussian EMAR(p), shared response reducing spaces; local Grassmann likelihood optimization, explicit source corrections; no SEMAR, selection or inference. |
| `mavats.envelope.fit_envelope_mar` | [samadi2026envelope](#samadi2026envelope) | Equations (13), (15)–(16), Algorithm 1; fixed positive envelope dimensions, joint-lag likelihood and free intercept; numerical choices and source discrepancies documented. |
| `benchmarks.envelope._KnownEMAR` | [samadi2026envelope](#samadi2026envelope) | Known-parameter conditional mean and impulse-response forecast noise floor for equation (13); extra oracle information, not estimation. |
| `mavats.alphaPCA.estimate_alpha_PCA` | [chen2021alphapca](#chen2021alphapca) | Legacy loading/scoring and covariance APIs; normalizations documented. |
| `mavats.alphaPCA.estimate_cov_Ri` | [chen2021alphapca](#chen2021alphapca) | Legacy loading/scoring and covariance APIs; normalizations documented. |
| `mavats.alphaPCA.estimate_cov_Cj` | [chen2021alphapca](#chen2021alphapca) | Legacy loading/scoring and covariance APIs; normalizations documented. |
| `mavats.factormodel.estimate_factor_model` | [wang2019matrixfactor](#wang2019matrixfactor) | Legacy lagged matrix factor API. |
| `mavats.factors.eigenvalue_ratio` | [wang2019matrixfactor](#wang2019matrixfactor) | All column-pair lag covariances; generic ratio helper explicitly defines search/tie/zero conventions. |
| `mavats.factors.fit_lagged_factor` | [wang2019matrixfactor](#wang2019matrixfactor) | All column-pair lag covariances; generic ratio helper explicitly defines search/tie/zero conventions. |
| `mavats.factors.FactorResult` | [wang2019matrixfactor](#wang2019matrixfactor), [chen2021alphapca](#chen2021alphapca), [chen2022tensorfactor](#chen2022tensorfactor) | Shared orthonormal Tucker result geometry, not an additional estimator; use the originating estimator's references. |
| `mavats.factors.fit_alpha_pca` | [chen2021alphapca](#chen2021alphapca) | Alpha-PCA loading/scoring branch. |
| `mavats.factors.fit_projected_pca` | [yu2022projection](#yu2022projection) | Projected matrix-factor estimator; iteration options documented. |
| `mavats.huber.HuberFactorResult` | [he2024huber](#he2024huber) | Matrixwise Huber loss; fixed threshold and sequential descent safeguard are explicit choices. |
| `mavats.huber.fit_huber_factor` | [he2024huber](#he2024huber) | Matrixwise Huber loss; fixed threshold and sequential descent safeguard are explicit choices. |
| `mavats.ihr.IHRRegressionDiagnostics` | [he2023ihr](#he2023ihr) | 2023 v1 entrywise IHR, fixed pilot threshold and rank rules; accepted renamed successor not reconciled. |
| `mavats.ihr.IHRFactorResult` | [he2023ihr](#he2023ihr) | 2023 v1 entrywise IHR, fixed pilot threshold and rank rules; accepted renamed successor not reconciled. |
| `mavats.ihr.fit_ihr_factor` | [he2023ihr](#he2023ihr) | 2023 v1 entrywise IHR, fixed pilot threshold and rank rules; accepted renamed successor not reconciled. |
| `mavats.ihr.IHRRankResult` | [he2023ihr](#he2023ihr) | 2023 v1 entrywise IHR, fixed pilot threshold and rank rules; accepted renamed successor not reconciled. |
| `mavats.ihr.select_ihr_ranks` | [he2023ihr](#he2023ihr) | 2023 v1 entrywise IHR, fixed pilot threshold and rank rules; accepted renamed successor not reconciled. |
| `mavats.inference.MARInferenceResult` | [chen2021mar](#chen2021mar) | Specified stationary MAR(1) asymptotic covariance/Wald and Kronecker specification procedures; no general HAC/post-selection inference. |
| `mavats.inference.mar_inference` | [chen2021mar](#chen2021mar) | Specified stationary MAR(1) asymptotic covariance/Wald and Kronecker specification procedures; no general HAC/post-selection inference. |
| `mavats.inference.MARSpecificationResult` | [chen2021mar](#chen2021mar) | Specified stationary MAR(1) asymptotic covariance/Wald and Kronecker specification procedures; no general HAC/post-selection inference. |
| `mavats.inference.mar_specification_test` | [chen2021mar](#chen2021mar) | Specified stationary MAR(1) asymptotic covariance/Wald and Kronecker specification procedures; no general HAC/post-selection inference. |
| `mavats.metrics.mean_squared_error` | [hyndman2006accuracy](#hyndman2006accuracy) | Conventional squared-error utility, pooling scalar entries; not MASE. |
| `mavats.metrics.relative_frobenius_error` | [chen2022tensorfactor](#chen2022tensorfactor) | Reconstruction setting reference; conventional relative norm, with explicit zero-truth behavior, not a new paper estimator. |
| `mavats.metrics.subspace_distance` | [wang2019matrixfactor](#wang2019matrixfactor) | Loading-space comparison; symmetric normalized projector distance extends equal-rank convention to unequal ranks. |
| `mavats.model_selection.ForecastEvaluation` | [tashman2000evaluation](#tashman2000evaluation) | Rolling-origin design; matrix/tensor callable factory is a software extension. |
| `mavats.model_selection.rolling_forecast` | [tashman2000evaluation](#tashman2000evaluation) | Rolling-origin design; matrix/tensor callable factory is a software extension. |
| `mavats.monitoring.MonitorStep` | [he2024breaks](#he2024breaks) | Accepted power transformation and four boundary families; result/state helpers inherit this reference. |
| `mavats.monitoring.MatrixFactorMonitor` | [he2024breaks](#he2024breaks) | Accepted power transformation and four boundary families; result/state helpers inherit this reference. |
| `mavats.robust.matrix_kendall` | [he2025kendall](#he2025kendall) | Matrix Kendall pair kernel; optional sampled-pair approximation explicitly labeled. |
| `mavats.robust.fit_matrix_kendall` | [he2025kendall](#he2025kendall) | Matrix Kendall pair kernel; optional sampled-pair approximation explicitly labeled. |
| `mavats.simulation.matrix_normal` | [dawid1981matrix](#dawid1981matrix) | Matrix-normal distribution; singular covariance support is an explicit extension. |
| `mavats.simulation.mar_spectral_radius` | [chen2021mar](#chen2021mar) | MAR model simulation/stability; selected lag/intercept/burn-in designs are not exact paper replications. |
| `mavats.simulation.simulate_mar` | [chen2021mar](#chen2021mar) | MAR model simulation/stability; selected lag/intercept/burn-in designs are not exact paper replications. |
| `mavats.simulation.FactorSimulation` | [chen2022tensorfactor](#chen2022tensorfactor) | Tucker model simulation with library-selected Gaussian AR cores/loadings/noise. |
| `mavats.simulation.simulate_factor` | [chen2022tensorfactor](#chen2022tensorfactor) | Tucker model simulation with library-selected Gaussian AR cores/loadings/noise. |
| `mavats.sparse.SparseMARResult` | [celani2024sparse](#celani2024sparse) | Continuous spike-and-slab EMVS posterior mode; prior-preserving normalization extension, not MCMC. |
| `mavats.sparse.fit_sparse_mar` | [celani2024sparse](#celani2024sparse) | Continuous spike-and-slab EMVS posterior mode; prior-preserving normalization extension, not MCMC. |
| `mavats.tensor.fit_tensor_factor` | [chen2022tensorfactor](#chen2022tensorfactor), [han2024iterative](#han2024iterative) | TOPUP/TIPUP cite the first paper; iTOPUP/iTIPUP cite both, including the dedicated iterative estimator. |
| `mavats.tensor_autoregression.TensorARProjection` | [li2021tenar](#li2021tenar) | 2021 manuscript projection/LS/MLE; local higher-order CP, explicit terms/lags, no automatic selection. |
| `mavats.tensor_autoregression.TensorARRun` | [li2021tenar](#li2021tenar) | 2021 manuscript projection/LS/MLE; local higher-order CP, explicit terms/lags, no automatic selection. |
| `mavats.tensor_autoregression.TensorARResult` | [li2021tenar](#li2021tenar) | 2021 manuscript projection/LS/MLE; local higher-order CP, explicit terms/lags, no automatic selection. |
| `mavats.tensor_autoregression.fit_tensor_ar` | [li2021tenar](#li2021tenar) | 2021 manuscript projection/LS/MLE; local higher-order CP, explicit terms/lags, no automatic selection. |
| `mavats.threshold.ThresholdFactorResult` | [liu2022threshold](#liu2022threshold) | Two-regime threshold factors, including unequal selected ranks; not multiple thresholds. |
| `mavats.threshold.fit_threshold_factors` | [liu2022threshold](#liu2022threshold) | Two-regime threshold factors, including unequal selected ranks; not multiple thresholds. |
| `mavats.volatility.MatrixGARCHParameters` | [yu2024garch](#yu2024garch) | Accepted matrix GARCH model, filtering/simulation/QMLE/next-step forecasts; optimizer and constraint choices documented. |
| `mavats.volatility.MatrixGARCHState` | [yu2024garch](#yu2024garch) | Accepted matrix GARCH model, filtering/simulation/QMLE/next-step forecasts; optimizer and constraint choices documented. |
| `mavats.volatility.MatrixGARCHForecast` | [yu2024garch](#yu2024garch) | Accepted matrix GARCH model, filtering/simulation/QMLE/next-step forecasts; optimizer and constraint choices documented. |
| `mavats.volatility.MatrixGARCHFilterResult` | [yu2024garch](#yu2024garch) | Accepted matrix GARCH model, filtering/simulation/QMLE/next-step forecasts; optimizer and constraint choices documented. |
| `mavats.volatility.filter_matrix_garch` | [yu2024garch](#yu2024garch) | Accepted matrix GARCH model, filtering/simulation/QMLE/next-step forecasts; optimizer and constraint choices documented. |
| `mavats.volatility.MatrixGARCHSimulation` | [yu2024garch](#yu2024garch) | Accepted matrix GARCH model, filtering/simulation/QMLE/next-step forecasts; optimizer and constraint choices documented. |
| `mavats.volatility.simulate_matrix_garch` | [yu2024garch](#yu2024garch) | Accepted matrix GARCH model, filtering/simulation/QMLE/next-step forecasts; optimizer and constraint choices documented. |
| `mavats.volatility.MatrixGARCHOptimizationRun` | [yu2024garch](#yu2024garch) | Accepted matrix GARCH model, filtering/simulation/QMLE/next-step forecasts; optimizer and constraint choices documented. |
| `mavats.volatility.MatrixGARCHResult` | [yu2024garch](#yu2024garch) | Accepted matrix GARCH model, filtering/simulation/QMLE/next-step forecasts; optimizer and constraint choices documented. |
| `mavats.volatility.fit_matrix_garch` | [yu2024garch](#yu2024garch) | Accepted matrix GARCH model, filtering/simulation/QMLE/next-step forecasts; optimizer and constraint choices documented. |

## Factor-rank selection and stability

| Canonical API | Paper key(s) | Citation scope |
| --- | --- | --- |
| `mavats.rank_selection.RankSelectionStep` | [han2022rank](#han2022rank) | Physical-unit IC/ER spectra and criterion diagnostics, not an inferential result. |
| `mavats.rank_selection.TensorRankSelectionResult` | [han2022rank](#han2022rank) | Equations (6)–(8), sequential rank/space updates and fixed-training-mean transforms; zero IC ranks allowed, ER searches positive ranks. |
| `mavats.rank_selection.select_tensor_rank` | [han2022rank](#han2022rank) | All five IC and five ER penalties for TOPUP/TIPUP and iterative variants; supplied strength exponent and multipliers, no automatic strength estimate or factor-loading inference. |
| `mavats.rank_stability.RankStabilityCell` | [han2022rank](#han2022rank) | Subsample IC result or retained numerical failure. |
| `mavats.rank_stability.RankStabilityInterval` | [han2022rank](#han2022rank) | Explicit finite-grid stability interval; not a confidence interval. |
| `mavats.rank_stability.RankStabilityChoice` | [han2022rank](#han2022rank) | Modewise finite-grid interpretation of Section 5.4; no choice is returned when no admissible interval exists. |
| `mavats.rank_stability.TensorRankStabilityResult` | [han2022rank](#han2022rank) | Remark 8 empirical variance over explicit nested spatial/time subsamples, retaining every fitted cell. |
| `mavats.rank_stability.tensor_rank_stability` | [han2022rank](#han2022rank) | Published empirical variance; supplied grids, variance tolerance and plateau conventions are documented software extensions, not a tuning optimality guarantee. |

## Benchmark-only scientific procedures

Most benchmark code chooses data designs, invokes the cited package methods,
or evaluates known-parameter oracles; those are not additional estimators.
The independently implemented fixed-rank vector VECM and binomial Wilson
intervals have their own citations below. Benchmark simulation designs inherit their model-family
references, with exact coefficients and contamination documented in
[the benchmark protocol](../benchmarks/README.md); they are not paper replications.

| Canonical API | Paper key(s) | Citation scope |
| --- | --- | --- |
| `benchmarks.structured_ar._fit_vector_vecm` | [johansen1991vecm](#johansen1991vecm) | Conditional Gaussian reduced-rank VECM, one difference lag and unrestricted intercept; fixed supplied rank, no rank test or inferential API. |
| `benchmarks.monitoring._wilson` | [wilson1927interval](#wilson1927interval) | Wilson score intervals for independent-series alarm/detection counts; not dependent monitoring time points. |
| `benchmarks.inference.summary` | [wilson1927interval](#wilson1927interval) | Wilson score intervals for specification rejection counts; coverage averages use replicate SEs, not binomial pooling of coefficients. |
| `benchmarks.marma._TrueMARMA` | [tsay2024marma](#tsay2024marma) | Known-parameter conditional mean and impulse-response covariance under the paper's minus-MA model; extra latent past innovations, not an estimated procedure. |
| `benchmarks.constrained_extensions._KnownLoadingProjection` | [chen2020constrained](#chen2020constrained) | Known-loading joint LS comparison in the paper's signal spaces; extra loading information, not a published loading estimator or universal error lower bound. |
| `benchmarks.constrained_extensions._IndependentSum` | [chen2020constrained](#chen2020constrained) | Deliberately unadjusted sum of single-term fits illustrating overlap bias; not the paper's multi-term estimator. |
| `benchmarks.advanced_matrix._gaussian_score` | [tsay2024marma](#tsay2024marma) | Conventional negative Gaussian density score with full constants and positive-definite covariance validation; applicable beyond MARMA, not a new estimator. |
| `benchmarks.factor_ranks._Projection` | [han2022rank](#han2022rank), [hyndman2006accuracy](#hyndman2006accuracy) | Known-loading Tucker projection or empty-loading training mean; extra information or imposed null model, not automatic rank selection. |
| `benchmarks.factor_ranks._adjacent_fit` | [wang2019matrixfactor](#wang2019matrixfactor), [han2022rank](#han2022rank) | Matched-cap adjacent-ratio initialization followed by fixed-rank loading iteration; not the published iterative rank-reselection criterion. |
| `benchmarks.rank_scenarios.population_factor_lag_moment` | [han2022rank](#han2022rank) | Independent analytic signal-only lag-moment oracle for the benchmark's stationary AR cores; designs are not paper replications. |

## References

### han2022rank

Han, Yuefeng, Chen, Rong, and Zhang, Cun-Hui (2022). Rank Determination in Tensor Factor Model. *Electronic Journal of Statistics*, 16(1), 1726–1803. [Journal DOI](https://doi.org/10.1214/22-EJS1991); implementation equations and version scope follow the [accessible author manuscript v3](https://arxiv.org/html/2011.07131v3). See [rank-selection notes](rank-selection-notes.md) and [stability notes](rank-stability-notes.md).

### chen2021mar

[Chen, Xiao & Yang (2021). Autoregressive Models for Matrix-Valued Time Series.](https://doi.org/10.1016/j.jeconom.2020.07.015)

### wang2019matrixfactor

[Wang, Liu & Chen (2019). Factor Models for Matrix-Valued High-Dimensional Time Series.](https://doi.org/10.1016/j.jeconom.2018.09.013)

### chen2021alphapca

[Chen & Fan (2021, online). Statistical Inference for High-Dimensional Matrix-Variate Factor Models.](https://doi.org/10.1080/01621459.2021.1970569)

### yu2022projection

[Yu, He, Kong & Zhang (2022). Projected Estimation for Large-Dimensional Matrix Factor Models.](https://doi.org/10.1016/j.jeconom.2021.04.001)

### chen2022tensorfactor

[Chen, Yang & Zhang (2022). Factor Models for High-Dimensional Tensor Time Series.](https://doi.org/10.1080/01621459.2021.1912757)

### han2024iterative

[Han, Chen, Yang & Zhang (2024). Tensor Factor Model Estimation by Iterative Projection.](https://doi.org/10.1214/24-AOS2412)

### he2025kendall

[He, Wang, Yu, Zhou & Zhou (2025). A New Non-Parametric Kendall's Tau for Matrix-Valued Elliptical Observations.](https://arxiv.org/abs/2207.09633)

### he2024huber

[He, Kong, Yu, Zhang & Zhao (2024; online 2023). Matrix Factor Analysis: From Least Squares to Iterative Projection.](https://doi.org/10.1080/07350015.2023.2191676)

### he2023ihr

[He, Kong, Liu & Zhao (2023 preprint v1). Robust Statistical Inference for Large-Dimensional Matrix-Valued Time Series via Iterative Huber Regression.](https://arxiv.org/abs/2306.03317v1)

### xiao2022rrmar

[Xiao, Han, Chen & Liu (2022 author manuscript). Reduced Rank Autoregressive Models for Matrix Time Series.](https://yuefenghan.github.io/papers/Reduced_Rank_MAR.pdf)

### celani2024sparse

[Celani, Pagnottoni & Jones (2024). Bayesian Variable Selection for Matrix Autoregressive Models.](https://doi.org/10.1007/s11222-024-10402-y)

### chang2023cp

[Chang, He, Yang & Yao (2023). Modelling Matrix Time Series via a Tensor CP-Decomposition.](https://academic.oup.com/jrsssb/article/85/1/127/7008470)

### chen2020constrained

[Chen, Tsay & Chen (2020). Constrained Factor Models for High-Dimensional Matrix-Variate Time Series.](https://doi.org/10.1080/01621459.2019.1584899)

### yuan2023dynamic

[Yuan, Gao, He, Huang & Guo (2023). Two-Way Dynamic Factor Models for High-Dimensional Matrix-Valued Time Series.](https://doi.org/10.1093/jrsssb/qkad077)

### han2024decorrelation

[Han, Chen, Zhang & Yao (2024). Simultaneous Decorrelation of Matrix Time Series.](https://doi.org/10.1080/01621459.2022.2151448)

### liu2022threshold

[Liu & Chen (2022). Identification and Estimation of Threshold Matrix-Variate Factor Models.](https://doi.org/10.1111/sjos.12576)

### yu2024garch

[Yu, Li, Jiang & Zhu (2025 issue; online 2024). Matrix GARCH Model: Inference and Application.](https://doi.org/10.1080/01621459.2024.2415719)

### he2024breaks

[He, Kong, Trapani & Yu (2024). Online Change-Point Detection for Matrix-Valued Time Series with Latent Two-Way Factor Structure.](https://doi.org/10.1214/24-AOS2410)

### li2021tenar

[Li & Xiao (2021 manuscript). Multi-Linear Tensor Autoregressive Models.](https://arxiv.org/abs/2110.00928v1)

### li2024cointegration

[Li & Xiao (2024 manuscript). Cointegrated Matrix Autoregression Models.](https://arxiv.org/abs/2409.10860v1)

### hoerl1970ridge

[Hoerl & Kennard (1970). Ridge Regression: Biased Estimation for Nonorthogonal Problems.](https://doi.org/10.1080/00401706.1970.10488634)

### hyndman2006accuracy

[Hyndman & Koehler (2006). Another Look at Measures of Forecast Accuracy.](https://doi.org/10.1016/j.ijforecast.2006.03.001)

### tashman2000evaluation

[Tashman (2000). Out-of-Sample Tests of Forecasting Accuracy: An Analysis and Review.](https://doi.org/10.1016/S0169-2070(00)00065-0)

### dawid1981matrix

[Dawid (1981). Some Matrix-Variate Distribution Theory: Notational Considerations and a Bayesian Application.](https://doi.org/10.1093/biomet/68.1.265)

### johansen1991vecm

[Johansen (1991). Estimation and Hypothesis Testing of Cointegration Vectors in Gaussian Vector Autoregressive Models.](https://doi.org/10.2307/2938278)

### wilson1927interval

[Wilson (1927). Probable Inference, the Law of Succession, and Statistical Inference.](https://doi.org/10.1080/01621459.1927.10502953)

### tsay2024marma

[Tsay (2024; online 2023). Matrix-Variate Time Series Analysis: A Brief Review and Some New Developments.](https://doi.org/10.1111/insr.12558)

### samadi2026envelope

[Samadi & De Alwis (2026; online 2025). Envelope Matrix Autoregressive Models.](https://doi.org/10.1080/07350015.2025.2537404)
