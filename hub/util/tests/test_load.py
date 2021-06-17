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

    assert not cat.was_read
    assert not flower.was_read

    assert cat.shape == (900, 900, 3)
    assert cat.was_read, "`was_read` should be true after reading any properties"
    assert cat.compression == "JPEG"
    assert cat.dtype == "uint8"
    assert cat.array.shape == (900, 900, 3)

    assert flower.shape == (513, 464, 4)
    assert flower.was_read, "`was_read` should be true after reading any properties"
    assert flower.compression == "PNG"
    assert flower.dtype == "uint8"
    assert flower.array.shape == (513, 464, 4)
