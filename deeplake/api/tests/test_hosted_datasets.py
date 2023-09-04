import pytest
import deeplake


@pytest.mark.slow
def test_mnist():
    ds = deeplake.load("hub://activeloop/mnist-test")
    sample = ds[0]
    assert sample.images.numpy().shape == (28, 28)
    assert sample.labels.numpy().shape == (1,)
    assert ds.storage.read_only is True
