import os
import hub


def test_load(cat_path, flower_path):
    cat = hub.load(cat_path)
    flower = hub.load(flower_path)

    assert cat.is_symbolic
    assert flower.is_symbolic

    assert cat.shape == (900, 900, 3)
    assert (
        not cat.is_symbolic
    ), "If any properties are read, this Sample is not symbolic"
    assert cat.compression == "jpeg"
    assert cat.dtype == "uint8"
    assert cat.array.shape == (900, 900, 3)

    assert flower.shape == (513, 464, 4)
    assert (
        not flower.is_symbolic
    ), "If any properties are read, this Sample is not symbolic"
    assert flower.compression == "png"
    assert flower.dtype == "uint8"
    assert flower.array.shape == (513, 464, 4)


# TODO: test creating Sample with np.ndarray
