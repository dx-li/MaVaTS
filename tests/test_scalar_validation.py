import numpy as np
import pytest

from mavats._validation import finite_scalar
from mavats.calibration import calibrate_monitor


@pytest.mark.parametrize(
    "value",
    [complex(0.5, 1), np.complex128(0.5 + 1j), np.complex64(0.5), np.array(0.5 + 0j)],
)
def test_real_scalar_validation_never_discards_an_imaginary_component(value):
    with pytest.raises(ValueError, match="real scalar"):
        finite_scalar(value, "value")
    with pytest.raises(ValueError, match="real scalar"):
        calibrate_monitor(10, alpha=value)
