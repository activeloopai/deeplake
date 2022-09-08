from hub.core.dataset import Dataset
from typing import List
from hub.core.transform.transform import ComputeFunction 
from torch.utils.data.dataloader import DataLoader
import torch
import numpy as np
import time

class Augmenter():    
  def __init__(self, transformation_dict=None):
    """
    Creates an Augmenter object. 
    
    Args:
      pipe_dict: (Optional) Give the transformation dictionary as input.
        Expected structure of transformation_dict -> keys: Tensors
                                                     values: List of transformation sequences
    """
    if transformation_dict!=None:
      self.pipe_dict = transformation_dict
    else:
      self.pipe_dict = {}

  def add_step(self, input_tensors , step_transform, tensor_condition=None):
    """
    Adds a transformation_pipeline to each of the tensors in input_tensors.

    Args:
      input_tensors: List of tensors
      step_transform: The transformation sequence to be used to transform tensors.
    """
    for tensor in input_tensors:
      if tensor not in self.pipe_dict.keys():
        self.pipe_dict[tensor] = [(step_transform, tensor_condition)]
      else:
        self.pipe_dict[tensor].append((step_transform, tensor_condition))

  def augment(self, ds, num_workers=1, batch_size=1):
    """
    Returns a Dataloader. Each sample in the dataloader contains a transformed tensor according to the defined steps.

    Args: 
      ds: Takes in a Hub dataset. 
      num_workers: The number of workers to use. 
    """
    #Todo - Add other arguments from dataset.pytorch
    pipe_dict = self.pipe_dict.copy()
    return ds.pytorch(transform=pipe_dict, multiple_transforms=True, num_workers=num_workers, batch_size=batch_size)




