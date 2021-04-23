"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import os
import glob
import shutil
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

PYTORCH_MODEL_DIR = "./data/pytorch_test/"
TF_MODEL_DIR = "./data/tensorflow_test/"


def create_pytorch_model(model_dir=None, epoch=None):
    if not model_dir:
        model_dir = PYTORCH_MODEL_DIR
    shutil.rmtree(model_dir, ignore_errors=True)
    os.makedirs(model_dir)
    model_arch = torch.nn.Sequential(
        torch.nn.Linear(1000, 100),
        torch.nn.ReLU(),
        torch.nn.Linear(100, 10),
    )
    model_init = Model(model_arch)
    model_init.store(model_dir, epoch=epoch)
    return model_init


@pytest.mark.skipif(
    not pytorch_loaded(),
    reason="requires pytorch to be loaded",
)
def test_store_load_torch():
    if torch_spec is None:
        raise ModuleNotFoundError("Module 'torch' is not installed")
    model_init = create_pytorch_model()
    loaded_model = Model()
    loaded_model.load(glob.glob(f"{PYTORCH_MODEL_DIR}*.pth")[0])
    assert repr(model_init._model) == repr(loaded_model._model)
    with torch.no_grad():
        for (p1, p2) in zip(
            model_init._model.named_parameters(), loaded_model._model.named_parameters()
        ):
            assert p1[0] == p2[0]
            assert torch.equal(p1[1].data, p2[1].data)


def test_store():
    shutil.rmtree(PYTORCH_MODEL_DIR, ignore_errors=True)
    os.makedirs(PYTORCH_MODEL_DIR)
    model_init = 5
    model_arch = Model(model_init)

    with pytest.raises(ValueError) as exc:
        model_arch.store(PYTORCH_MODEL_DIR)
    assert exc.type == ValueError


@pytest.mark.skipif(
    not tensorflow_loaded(),
    reason="requires tensorflow to be loaded",
)
def test_store_load_tf():
    if tensorflow_spec is None:
        raise ModuleNotFoundError("Module 'tensorflow' is not installed")

    shutil.rmtree(TF_MODEL_DIR, ignore_errors=True)
    os.makedirs(TF_MODEL_DIR)

    def get_model():
        inputs = tf.keras.Input(shape=(32,))
        outputs = tf.keras.layers.Dense(1)(inputs)
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="mean_squared_error")
        return model

    model_arch = get_model()
    model_init = Model(model_arch)
    test_input = np.random.random((128, 32))
    model_init.store(TF_MODEL_DIR)
    loaded_model = Model()
    loaded_model.load(glob.glob(f"{TF_MODEL_DIR}*.h5")[0])
    np.testing.assert_allclose(
        model_init._model.predict(test_input), loaded_model._model.predict(test_input)
    )


def test_pytorch_lightning_import():
    pytorch_lightning_spec = importlib.util.find_spec("pytorch_lightning")
    if pytorch_lightning_spec is not None:
        try:
            import pytorch_lightning as pl

            PYTORCH_LIGHTNING_MODEL_CLASSES = (pl.LightningModule,)
            assert True
        except:
            assert False
    else:
        assert True


@pytest.mark.skipif(
    not pytorch_loaded(),
    reason="requires pytorch to be loaded",
)
def test_epoch():
    create_pytorch_model(epoch=5)
    assert any("_5.pth" in filename for filename in os.listdir(PYTORCH_MODEL_DIR))


@pytest.mark.skipif(
    not pytorch_loaded(),
    reason="requires pytorch to be loaded",
)
def test_token():
    import botocore

    model_dir = "s3://test"
    os.environ["AWS_ACCESS_KEY_ID"] = "AWS_ACCESS_KEY_ID"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "AWS_SECRET_ACCESS_KEY"
    os.environ["AWS_DEFAULT_REGION"] = "us-east1"
    with pytest.raises(botocore.exceptions.EndpointConnectionError):
        create_pytorch_model(model_dir=model_dir)
    with pytest.raises(botocore.exceptions.EndpointConnectionError):
        loaded_model = Model()
        loaded_model.load(os.path.join(model_dir, "model.pth"))
