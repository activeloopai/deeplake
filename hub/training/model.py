import os
import sys
from typing import Union, Dict
import json
import tempfile
import io

from hub.store.store import get_fs_and_path

import importlib

torch_spec = importlib.util.find_spec("torch")
if torch_spec is not None:
    import torch
tensorflow_spec = importlib.util.find_spec("tensorflow")
if tensorflow_spec is not None:
    import tensorflow as tf
    import h5py

if torch_spec is not None:
    PYTORCH_MODEL_CLASSES = (torch.nn.Module, torch.nn.Sequential)
if tensorflow_spec is not None:
    TENSORFLOW_MODEL_CLASSES = (tf.keras.Model, tf.keras.Sequential)


class Model:
    def __init__(self, model=None, metainfo: Dict = dict()):
        """Creates model with given metainfo.

        Parameters
        ----------
        model: PyTorch, Tensorflow or Pytorch Lightning model object
        metainfo: dict
            Dictionary of model meta information
        """
        self._metainfo = metainfo
        self._model = model

    @property
    def description(self) -> str:
        """Model description"""
        return self._metainfo.get("description") if self._metainfo else None

    def load(self, model_path: str, token: str = None):
        """Loads a Pytorch or Tensorflow model

        Usage
        ----------

        >>> loaded_model = load('path/to/model/file')

        Parameters
        ----------
        model_path: str
            Path(local or s3) to model file. Should be of type '.h5'
                    for Tensorflow models and of type '.pth' or '.pt' for PyTorch models.
        token: str
            Path to aws credentials if `model_path` is aws s3 path.
            default: os.environ['AWS_CONFIG_FILE']

        Returns
        ----------
        Pytorch or tf.keras(compiled if saved model was compiled) models
        """
        if model_path.startswith("s3://"):
            if not token:
                token = os.environ["AWS_CONFIG_FILE"]
            fs, url = get_fs_and_path(model_path, token=token)
            url = os.path.join("s3://", url)
        else:
            fs, url = get_fs_and_path(model_path)
        if model_path.endswith(".pth") or model_path.endswith(".pt"):
            if "torch" not in sys.modules:
                raise ModuleNotFoundError(
                    "Unable to load a model. \
                                        Module 'torch' is not installed"
                )
            with fs.open(model_path, "rb") as opened_file:
                self._model = torch.load(opened_file)
        elif model_path.endswith(".h5"):
            if "tensorflow" not in sys.modules:
                raise ModuleNotFoundError(
                    "Unable to load a model. \
                                        Module 'tensorflow' is not installed"
                )
            with fs.open(model_path, "rb") as opened_file:
                f = h5py.File(opened_file, "r")
                self._model = tf.keras.models.load_model(f)
        elif model_path.endswith(".tf"):
            if "tensorflow" not in sys.modules:
                raise ModuleNotFoundError(
                    "Unable to load a model. \
                                        Module 'tensorflow' is not installed"
                )
            self._model = tf.keras.models.load_model(model_path)
        else:
            raise ValueError("Not supported model type")

    def store(self, model_dir: str, token: str = None):
        """Saves an object to a file.

        Usage
        ----------

        >>> store(/dir/to/save/model/, model)

        Parameters
        ----------
        model_dir: str
            Path(local or s3) to folder where model will be saved.
        model: PyTorch or tf.Keras model
        token: str
            Path to aws credentials if `model_dir` is aws s3 path.
            default: os.environ['AWS_CONFIG_FILE']

        Raises
        ----------
        ValueError: If model type is not supported(supported types:
                torch.nn.Module, tf.keras.Model, tf.keras.Sequential)
        """
        if model_dir.startswith("s3://"):
            if not token:
                token = os.environ["AWS_CONFIG_FILE"]
            fs, url = get_fs_and_path(model_dir, token=token)
            url = os.path.join("s3://", url)
        else:
            fs, url = get_fs_and_path(model_dir)
        model_class = self._model.__class__
        if "PYTORCH_MODEL_CLASSES" in globals() and issubclass(
            model_class, PYTORCH_MODEL_CLASSES
        ):
            model_full_path = os.path.join(url, model_class.__name__ + ".pth")
            with fs.open(model_full_path, "wb") as opened_file:
                torch.save(self._model, opened_file)
        elif "TENSORFLOW_MODEL_CLASSES" in globals() and issubclass(
            model_class, TENSORFLOW_MODEL_CLASSES
        ):
            try:
                model_full_path = os.path.join(url, model_class.__name__ + ".h5")
                io_h5 = io.BytesIO()
                self._model.save(io_h5)
                with fs.open(model_full_path, "wb") as opened_file:
                    opened_file.write(io_h5.getbuffer())
            except TypeError:
                model_full_path = os.path.join(url, model_class.__name__ + ".tf")
                self._model.save(model_full_path)
        else:
            raise ValueError(f"Unable to store a model of type {type(self._model)}")
