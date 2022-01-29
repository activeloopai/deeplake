from hub import read as hub_read


def test_read(cat_path, flower_path):
    cat = hub_read(cat_path)
    flower = hub_read(flower_path)

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


# TODO: test creating Sample with np.ndarray
