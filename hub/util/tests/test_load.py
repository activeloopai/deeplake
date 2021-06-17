import os
from hub.tests.common import get_dummy_data_path
import hub


def test_load():
    # TODO: make test fixtures for these paths
    path = get_dummy_data_path("compressed_images")
    cat_path = os.path.join(path, "cat.jpeg")
    flower_path = os.path.join(path, "flower.png")

    # TODO: hub.load
    cat = hub.load(cat_path)
    flower = hub.load(flower_path)

    assert cat.is_symbolic
    assert flower.is_symbolic

    assert cat.shape == (900, 900, 3)
    assert (
        not cat.is_symbolic
    ), "If any properties are read, this Sample is not symbolic"
    assert cat.compression == "JPEG"
    assert cat.dtype == "uint8"
    assert cat.array.shape == (900, 900, 3)

    assert flower.shape == (513, 464, 4)
    assert (
        not flower.is_symbolic
    ), "If any properties are read, this Sample is not symbolic"
    assert flower.compression == "PNG"
    assert flower.dtype == "uint8"
    assert flower.array.shape == (513, 464, 4)


# TODO: test creating Sample with np.ndarray
