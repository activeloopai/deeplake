from hub.util.exceptions import HubAutoUnsupportedFileExtensionError
import pytest
import numpy as np
from hub.api.tests.common import get_dummy_data_path
import hub


def test_jpeg():
    path = get_dummy_data_path("cat.jpeg")
    cat = hub.load(path)
    assert type(cat) == np.ndarray
    assert cat.shape == (900, 900, 3)


@pytest.mark.xfail(raises=HubAutoUnsupportedFileExtensionError, strict=True)
def test_unsupported():
    path = get_dummy_data_path("cat.unsupported_ext")
    hub.load(path)
