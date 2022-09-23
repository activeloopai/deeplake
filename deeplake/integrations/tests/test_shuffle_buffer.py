import pytest
from deeplake.util.check_installation import pytorch_installed

if not pytorch_installed():
    pytest.skip("pytroch is not installed", allow_module_level=True)

from deeplake.integrations.pytorch.shuffle_buffer import ShuffleBuffer
import torch


def test_zero_buffer_size():
    with pytest.raises(ValueError):
        ShuffleBuffer(0)


def test_too_small_buffer():
    buffer = ShuffleBuffer(10)

    tensor = {"val": torch.ones(10)}

    with pytest.warns(UserWarning):
        result = buffer.exchange(tensor)

    assert result == tensor


def test_adding_tensor():
    buffer = ShuffleBuffer(40)
    tensor = {"val": torch.ones(10)}

    result = buffer.exchange(tensor)

    assert result == None


def test_constant_tensor():
    buffer = ShuffleBuffer(8)
    tensor = {"val": torch.tensor(1)}

    result = buffer.exchange(tensor)
    result = buffer.exchange(tensor)

    assert result == tensor
