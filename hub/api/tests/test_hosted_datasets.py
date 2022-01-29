from hub import load as hub_load


def test_mnist():
    ds = hub_load("hub://activeloop/mnist-test")
    sample = ds[0]
    assert sample.images.numpy().shape == (28, 28)
    assert sample.labels.numpy().shape == (1,)
