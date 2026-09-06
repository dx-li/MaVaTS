"""Scientific methods for matrix- and tensor-valued time series.

Arrays are time-first. Tucker factor results use orthonormal loadings; additive
dynamic factors and legacy submodules have their own documented normalizations.
See the repository's
docs/methods.md for assumptions, references and coverage gaps. This development
release is not a claim of complete literature coverage.
"""

from .autoregression import (
    MARRankSelection,
    MARResult,
    fit_mar,
    nearest_kronecker_product,
    select_mar_rank,
)
from .baselines import VARResult, fit_naive, fit_var
from .calibration import MonitorCalibration, calibrate_monitor
from .cointegration import CMARI1Diagnostics, CMARResult, cmar_i1_diagnostics, fit_cmar
from .constrained import fit_constrained_factor
from .cp import CPIdentificationError, CPResult, fit_cp_factor
from .decorrelation import MatrixDecorrelationResult, fit_matrix_decorrelation
from .dynamic import TwoWayDynamicResult, fit_two_way_dynamic
from .factors import (
    FactorResult,
    eigenvalue_ratio,
    fit_alpha_pca,
    fit_lagged_factor,
    fit_projected_pca,
)
from .huber import HuberFactorResult, fit_huber_factor
from .ihr import IHRFactorResult, IHRRankResult, fit_ihr_factor, select_ihr_ranks
from .inference import (
    MARInferenceResult,
    MARSpecificationResult,
    mar_inference,
    mar_specification_test,
)
from .monitoring import MatrixFactorMonitor, MonitorStep
from .robust import fit_matrix_kendall, matrix_kendall
from .sparse import SparseMARResult, fit_sparse_mar
from .tensor import fit_tensor_factor
from .tensor_autoregression import TensorARResult, fit_tensor_ar
from .threshold import ThresholdFactorResult, fit_threshold_factors
from .volatility import (
    MatrixGARCHFilterResult,
    MatrixGARCHForecast,
    MatrixGARCHParameters,
    MatrixGARCHResult,
    MatrixGARCHSimulation,
    MatrixGARCHState,
    filter_matrix_garch,
    fit_matrix_garch,
    simulate_matrix_garch,
)

__version__ = "0.2.0.dev0"

__all__ = [
    "CMARI1Diagnostics",
    "CMARResult",
    "cmar_i1_diagnostics",
    "fit_cmar",
    "IHRFactorResult",
    "IHRRankResult",
    "fit_ihr_factor",
    "select_ihr_ranks",
    "TensorARResult",
    "fit_tensor_ar",
    "MatrixFactorMonitor",
    "MonitorStep",
    "MonitorCalibration",
    "calibrate_monitor",
    "TwoWayDynamicResult",
    "fit_two_way_dynamic",
    "MatrixGARCHFilterResult",
    "MatrixGARCHForecast",
    "MatrixGARCHParameters",
    "MatrixGARCHResult",
    "MatrixGARCHSimulation",
    "MatrixGARCHState",
    "filter_matrix_garch",
    "fit_matrix_garch",
    "simulate_matrix_garch",
    "MARInferenceResult",
    "MARSpecificationResult",
    "MatrixDecorrelationResult",
    "SparseMARResult",
    "ThresholdFactorResult",
    "fit_matrix_decorrelation",
    "fit_sparse_mar",
    "fit_threshold_factors",
    "mar_inference",
    "mar_specification_test",
    "HuberFactorResult",
    "fit_huber_factor",
    "CPIdentificationError",
    "CPResult",
    "fit_cp_factor",
    "FactorResult",
    "MARResult",
    "MARRankSelection",
    "VARResult",
    "eigenvalue_ratio",
    "fit_alpha_pca",
    "fit_constrained_factor",
    "fit_lagged_factor",
    "fit_mar",
    "fit_matrix_kendall",
    "fit_naive",
    "fit_projected_pca",
    "fit_tensor_factor",
    "fit_var",
    "matrix_kendall",
    "nearest_kronecker_product",
    "select_mar_rank",
]
