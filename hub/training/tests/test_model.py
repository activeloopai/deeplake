import numpy as np
import pytest
from hub.utils import pytorch_loaded, tensorflow_loaded
from hub.training.model import Model

import importlib

torch_spec = importlib.util.find_spec("torch")
if torch_spec is not None:
    import torch
tensorflow_spec = importlib.util.find_spec("tensorflow")
if tensorflow_spec is not None:
    import tensorflow as tf


@pytest.mark.skipif(
    not pytorch_loaded(),
    reason="requires pytorch to be loaded",
)
def test_store_load_torch():
    if torch_spec is None:
        raise ModuleNotFoundError("Module 'torch' is not installed")
    model_arch = torch.nn.Sequential(
        torch.nn.Linear(1000, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 10),
    )
    model_init = Model(model_arch)
    model_init.store("./data/")
    loaded_model = Model()
    loaded_model.load(f"./data/{model_init._model.__class__.__name__}.pth")
    assert repr(model_init._model) == repr(loaded_model._model)
    for (p1, p2) in zip(
        model_init._model.named_parameters(), loaded_model._model.named_parameters()
    ):
        assert p1[0] == p2[0]
        assert torch.equal(p1[1].data, p2[1].data)


@pytest.mark.skipif(
    not tensorflow_loaded(),
    reason="requires tensorflow to be loaded",
)
def test_store_load_tf():
    if tensorflow_spec is None:
        raise ModuleNotFoundError("Module 'tensorflow' is not installed")
    ext = ".h5" if int((tf.__version__)[0]) < 2 else ".tf"

    def get_model():
        inputs = tf.keras.Input(shape=(32,))
        outputs = tf.keras.layers.Dense(1)(inputs)
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="mean_squared_error")
        return model

    model_arch = get_model()
    model_init = Model(model_arch)
    test_input = np.random.random((128, 32))
    model_init.store("/tmp/")
    loaded_model = Model()
    loaded_model.load(f"/tmp/{model_init._model.__class__.__name__}{ext}")
    np.testing.assert_allclose(
        model_init._model.predict(test_input), loaded_model._model.predict(test_input)
    )
