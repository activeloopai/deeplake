import pytest

from deeplake import DeeplakeRandom


@pytest.mark.parametrize("input", [1234, None])
def test_seed(input):
    obj = DeeplakeRandom()
    obj.seed(input)

    assert obj.get_seed() == input


@pytest.mark.parametrize("input", [1.3, "a string"])
def test_invalid_seed(input):
    obj = DeeplakeRandom()
    with pytest.raises(TypeError):
        obj.seed(input)
