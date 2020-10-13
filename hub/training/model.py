import os
from typing import Union
import json
import tempfile
import io

import torch
import tensorflow as tf
import h5py

from model_example import Identity
from hub.utils import get_fs_and_path


TENSORFLOW_MODEL_CLASSES = (tf.keras.Model, tf.keras.Sequential)
PYTORCH_MODEL_CLASSES = (torch.nn.Module,)


def load(model_path: str, token: str = None):
    """Loads a Pytorch or Tensorflow model
    Usage:
    >>> loaded_model = load('path/to/model/file')
    
    Arguments:
    model_path: Path(local or s3) to model file. Should be of type '.h5' for Tensorflow models 
                and of type '.pth' or '.pt' for PyTorch models.
    token: Path to aws credentials if `model_path` is aws s3 path. 
           default: os.environ['AWS_CONFIG_FILE'] 

    Returns:
    Pytorch or tf.keras(compiled if saved model was compiled) models
    """
    if model_path.startswith('s3://'):
        if not token:
            token = os.environ['AWS_CONFIG_FILE']
        fs, url = get_fs_and_path(model_path, token=token)  
        url = os.path.join('s3://', url) 
    else:
        fs, url = get_fs_and_path(model_path)   
    if model_path.endswith('.pth') or model_path.endswith('.pt'): 
        with fs.open(model_path, 'rb') as opened_file:  
            model = torch.load(opened_file)
    elif model_path.endswith('.h5'):    
        with fs.open(model_path, 'rb') as opened_file:
            f = h5py.File(opened_file, 'r')
            model = tf.keras.models.load_model(f)
    else:
        raise ValueError("Not supported model type")
    return model


def store(model_dir: str, model, token: str = None): 
    """Saves an object to a file.
    Usage:
    >>> store(/dir/to/save/model/, model)
    
    Arguments:
    model_dir: Path(local or s3) to folder where model will be saved.
    model: PyTorch or tf.Keras model
    token: Path to aws credentials if `model_dir` is aws s3 path. 
           default: os.environ['AWS_CONFIG_FILE'] 

    Raises: ValueError if model type is not supported(supported types:
            torch.nn.Module, tf.keras.Model, tf.keras.Sequential)
    """
    if model_dir.startswith('s3://'):
        if not token:
            token = os.environ['AWS_CONFIG_FILE']
        fs, url = get_fs_and_path(model_dir, token=token)  
        url = os.path.join('s3://', url) 
    else:
        fs, url = get_fs_and_path(model_dir)
    model_class = model.__class__
    if issubclass(model_class, PYTORCH_MODEL_CLASSES):              
        model_full_path = os.path.join(url, model.__class__.__name__ + '.pth')
        with fs.open(model_full_path, 'wb') as opened_file:  
            torch.save(model, opened_file)
    elif issubclass(model_class, TENSORFLOW_MODEL_CLASSES):
        model_full_path = os.path.join(url, model.__class__.__name__ + '.h5')
        io_h5 = io.BytesIO()
        model.save(io_h5)
        with fs.open(model_full_path, 'wb') as opened_file: 
            opened_file.write(io_h5.getbuffer()) 
    else:
        raise ValueError(f"Unable to store a model of type {type(model)}")
