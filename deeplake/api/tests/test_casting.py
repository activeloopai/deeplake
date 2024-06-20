import numpy as np
import pytest

from deeplake.util.casting import intelligent_cast
from deeplake.util.exceptions import TensorDtypeMismatchError


def test_intelligent_cast():
    assert intelligent_cast(1, "int64", "generic").dtype == np.int64
    assert intelligent_cast(1, np.int64, "generic").dtype == np.int64
    assert intelligent_cast(1, np.float64, "generic").dtype == np.float64
    assert intelligent_cast(True, np.bool_, "generic").dtype == np.bool_

    with pytest.raises(TensorDtypeMismatchError):
        intelligent_cast(1, "int32", "generic")
    with pytest.raises(TensorDtypeMismatchError):
        intelligent_cast(1, np.float32, "generic")
