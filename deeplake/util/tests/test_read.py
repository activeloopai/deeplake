import numpy as np
import deeplake


def test_read(cat_path, flower_path):
    cat = deeplake.read(cat_path)
    flower = deeplake.read(flower_path)

    assert cat.is_lazy
    assert flower.is_lazy

    assert cat.shape == (900, 900, 3)
    assert cat.is_lazy, "Reading properties should not decompress sample."
    assert cat.compression == "jpeg"
    assert cat.dtype == "uint8"
    assert cat.array.shape == (900, 900, 3)

    assert flower.shape == (513, 464, 4)
    assert flower.is_lazy, "Reading properties should not decompress sample."
    assert flower.compression == "png"
    assert flower.dtype == "uint8"
    assert flower.array.shape == (513, 464, 4)

    arr = np.array(cat, dtype=np.uint32)
    assert arr.shape == (900, 900, 3)
    assert arr.dtype == np.uint32

    pil_img = flower.pil
    assert pil_img.size == (464, 513)


# TODO: test creating Sample with np.ndarray
