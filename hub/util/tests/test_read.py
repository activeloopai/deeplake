import hub


def test_read(cat_path, flower_path):
    cat = hub.read(cat_path)
    flower = hub.read(flower_path)

    assert cat.is_lazy
    assert flower.is_lazy

    assert cat.shape == (900, 900, 3)
    assert not cat.is_lazy, "If any properties are read, this Sample is not lazy"
    assert cat.compression == "jpeg"
    assert cat.dtype == "uint8"
    assert cat.array.shape == (900, 900, 3)

    assert flower.shape == (513, 464, 4)
    assert not flower.is_lazy, "If any properties are read, this Sample is not lazy"
    assert flower.compression == "png"
    assert flower.dtype == "uint8"
    assert flower.array.shape == (513, 464, 4)


# TODO: test creating Sample with np.ndarray
