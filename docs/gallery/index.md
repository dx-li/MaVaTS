# Real-data method gallery

Read the [data protocol and interpretation guide](../real-world-examples.md) first.
All applications use the same fixed observational split; none establishes model validity.

| Walkthrough | API | Output type | Execution status |
| --- | --- | --- | --- |
| [MAR: PROJECTION](mar-projection.md) | `fit_mar` | forecast | completed |
| [MAR: ALS](mar-als.md) | `fit_mar` | forecast | completed |
| [MAR: MLE](mar-mle.md) | `fit_mar` | forecast | completed |
| [Two-lag MAR](mar-multilag.md) | `fit_mar` | forecast | completed |
| [Reduced-rank MAR](mar-reduced-rank.md) | `fit_mar` | forecast | completed |
| [MAR rank selection](mar-rank-selection.md) | `select_mar_rank` | forecast | completed |
| [Envelope MAR](envelope-mar.md) | `fit_envelope_mar` | forecast | completed |
| [Matrix ARMA: LS](marma-ls.md) | `fit_marma` | forecast | completed |
| [Cointegrated MAR: LS](cmar-ls.md) | `fit_cmar` | forecast | completed |
| [Matrix ARMA: MLE](marma-mle.md) | `fit_marma` | forecast | not_converged |
| [Cointegrated MAR: MLE](cmar-mle.md) | `fit_cmar` | forecast | completed |
| [Tensor AR: PROJECTION](tensor-ar-projection.md) | `fit_tensor_ar` | forecast | completed |
| [Tensor AR: LS](tensor-ar-ls.md) | `fit_tensor_ar` | forecast | completed |
| [Tensor AR: MLE](tensor-ar-mle.md) | `fit_tensor_ar` | forecast | completed |
| [Sparse MAR](sparse-mar.md) | `fit_sparse_mar` | forecast | completed |
| [MAR marginal intervals: PROJECTION](inference-projection.md) | `mar_inference` | inference | completed |
| [MAR marginal intervals: ALS](inference-als.md) | `mar_inference` | inference | completed |
| [MAR marginal intervals: MLE](inference-mle.md) | `mar_inference` | inference | completed |
| [Kronecker specification test](specification.md) | `mar_specification_test` | specification | completed |
| [Alpha-PCA: alpha=-1](alpha-pca-0.md) | `fit_alpha_pca` | factor | completed |
| [Alpha-PCA: alpha=0](alpha-pca-1.md) | `fit_alpha_pca` | factor | completed |
| [Alpha-PCA: alpha=1](alpha-pca-2.md) | `fit_alpha_pca` | factor | completed |
| [Projected PCA](projected-pca.md) | `fit_projected_pca` | factor | completed |
| [Lagged matrix factors](lagged-factor.md) | `fit_lagged_factor` | factor | completed |
| [Matrix Kendall factors](kendall.md) | `fit_matrix_kendall` | factor | completed |
| [Matrixwise Huber factors](huber.md) | `fit_huber_factor` | factor | completed |
| [Entrywise Huber factors](ihr.md) | `fit_ihr_factor` | factor | completed |
| [TOPUP tensor factors](topup.md) | `fit_tensor_factor` | factor | completed |
| [ITOPUP tensor factors](itopup.md) | `fit_tensor_factor` | factor | completed |
| [TIPUP tensor factors](tipup.md) | `fit_tensor_factor` | factor | completed |
| [ITIPUP tensor factors](itipup.md) | `fit_tensor_factor` | factor | completed |
| [Tensor rank selection: IC](tensor-rank-ic.md) | `select_tensor_rank` | rank | completed |
| [Tensor rank selection: ER](tensor-rank-er.md) | `select_tensor_rank` | rank | completed |
| [Rank stability path](rank-stability.md) | `tensor_rank_stability` | stability | completed |
| [IHR rank selection: ratio](ihr-rank-ratio.md) | `select_ihr_ranks` | rank | completed |
| [IHR rank selection: threshold](ihr-rank-threshold.md) | `select_ihr_ranks` | rank | completed |
| [Constrained factors](constrained.md) | `fit_constrained_factor` | factor | completed |
| [Partially constrained factors](partial-constrained.md) | `fit_partial_constrained_factor` | factor | completed |
| [Multi-term constrained factors](multiterm-constrained.md) | `fit_multiterm_constrained_factor` | factor | completed |
| [CP matrix factors](cp.md) | `fit_cp_factor` | factor | completed |
| [Two-way dynamic factors](two-way-dynamic.md) | `fit_two_way_dynamic` | forecast | completed |
| [Threshold factors](threshold.md) | `fit_threshold_factors` | threshold | completed |
| [Matrix decorrelation](decorrelation.md) | `fit_matrix_decorrelation` | decorrelation | completed |
| [Matrix GARCH](matrix-garch.md) | `fit_matrix_garch` | volatility | completed |
| [Sequential factor monitoring](monitor.md) | `MatrixFactorMonitor` | monitor | completed |
| [Gaussian-reference monitor calibration](calibrated-monitor.md) | `calibrate_monitor` | monitor | completed |
| [Naive baseline: last](naive-last.md) | `fit_naive` | forecast | completed |
| [Naive baseline: mean](naive-mean.md) | `fit_naive` | forecast | completed |
| [Unrestricted VAR baseline](var.md) | `fit_var` | forecast | completed |
