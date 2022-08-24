from hub.core.dataset import Dataset
from typing import List
from hub.core.transform.transform import ComputeFunction 
from torch.utils.data.dataloader import DataLoader
import torch
from hub.core.augment.utils import *
# import cv2 as cv
# from PIL import Image
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

  def add_step(self, input_tensors , step_transform):
    """
    Adds a transformation_pipeline to each of the tensors in input_tensors.

    Args:
      input_tensors: List of tensors
      step_transform: The transformation sequence to be used to transform tensors.
    """
    for tensor in input_tensors:
      if tensor not in self.pipe_dict.keys():
        self.pipe_dict[tensor] = [step_transform]
      else:
        self.pipe_dict[tensor].append(step_transform)

  def augment(self, ds, num_workers=1):
    """
    Returns a Dataloader. Each sample in the dataloader contains a transformed tensor according to the defined steps.

    Args: 
      ds: Takes in a Hub dataset. 
      num_workers: The number of workers to use. 
    """
    #Todo - Add other arguments from dataset.pytorch
    pipe_dict = self.pipe_dict.copy()
    return ds.pytorch(transform=pipe_dict, multiple_transforms=True, num_workers=num_workers)

class Policy():
  def __init__(self, policy_input="random_policy_1"):
    """
    Initializes an Auto-Augment like policy.

    Args:
      policy_input: Can be a string or a list of tuples. Each tuple might have multiple tuples inside which represents
                    transformations to be applied sequentially according to a probability and magnitude. 
   
    Example:
        ``` 
        policy_input = [
          (("transform1", probability1, magnitude1), ("transform2", probability2, magnitude2)),
          (("transform3", probability3, magnitude3), ("transform4", probability4, magnitude4)),
        ]
        policy_obj = Policy(policy_input)
        ```
        For each sample(image) we choose one of the elements of the list and iterate through the tuple to perform all
        the transformations according to their probability and magnitude.
                    
    """
    if isinstance(policy_input,str):
      self.policy_set = self.get_policies(policy_input)
    else:
      self.policy_set = policy_input
    self.transforms_lis = None


  def get_policies(self, policy_name):
    """
    Stores pretrained policies. Triggered if the policy_input is a string.

    Args:
      policy_name: Name of a pretrained policy.
    """

    if policy_name == "random_policy_1":
      return [
        (("Posterize", 0.4, 4), ("Rotate", 0.6, 30)),
        (("Equalize", 0.8, None), ("Equalize", 0.6, None)),
        (("Posterize", 0.6, 2), ("Posterize", 0.6, 3)),
        (("Equalize", 0.4, None), ("Solarize", 0.2, 150)),
        (("Equalize", 0.4, None), ("Rotate", 0.8, 60)),
        (("Solarize", 0.6, 200), ("Equalize", 0.6, None)),
        (("Posterize", 0.8, 1), ("Equalize", 1.0, None)),
        (("Rotate", 0.2, 90), ("Solarize", 0.6, 100)),
        (("Equalize", 0.6, None), ("Posterize", 0.4, 6)),
        (("Rotate", 0.8, 70), ("Color", 0.4, 0.5)),
        (("Rotate", 0.4, 30), ("Equalize", 0.6, None)),
        (("Equalize", 0.0, None), ("Equalize", 0.8, None)),
        (("Invert", 0.6, None), ("Equalize", 1.0, None)),
        (("Color", 0.6, 0.4), ("Contrast", 1.0, 1)),
        (("Rotate", 0.8, 8), ("Color", 1.0, 0.1)),
        (("Color", 0.8, 1), ("Solarize", 0.8, 200)),
        (("ShearX", 0.6, 30), ("Equalize", 1.0, None)),
        (("Color", 0.4, 0.4), ("Equalize", 0.6, None)),
        (("Equalize", 0.4, None), ("Solarize", 0.2, 100)),
        (("Invert", 0.6, None), ("Equalize", 1.0, None)),
        (("Color", 0.6, 0.1), ("Contrast", 1.0, 3)),
        (("Equalize", 0.8, None), ("Equalize", 0.6, None)),
        ]


  def run_policy(self, image):
    """
    Runs a policy on a given image.shape

    Args:
      image: A numpy array
    """
    num_policies = len(self.policy_set)
    policy_id = np.random.randint(num_policies)
    policy = self.policy_set[policy_id]
    probs = np.random.rand(3)
    for i in range(len(policy)):
      if probs[i] < policy[i][1]:
        image = run_transform(image, policy[i][0], policy[i][2])
    return image
  

  def run_policy_sample(self, sample):
    """
    Runs a policy on a sample. Used in return_dataloader. 
    """
    image = sample[self.tensor]
    num_policies = len(self.policy_set)
    policy_id = np.random.randint(num_policies)
    policy = self.policy_set[policy_id]
    probs = np.random.rand(3)
    for i in range(len(policy)):
      if probs[i] < policy[i][1]:
        image = run_transform(image, policy[i][0], policy[i][2])
    sample[self.tensor] = image
    return sample

  def augment(self, ds, tensor="images", batch_size = 1, num_workers = None):#used to be return_dataloader
    """
    Returns a dataloader. 

    Args:
      ds: A hub dataset.
      tensor: tensor_name that has to be transformed.
      batch_size: The batch size of the dataloader. 
      num_workers: The number of workers to be used.
    """
    #TODO add pytorch arguments
    self.tensor = tensor
    if num_workers is not None:
      return ds.pytorch(batch_size=batch_size, transform=self.run_policy_sample, num_workers = num_workers)
    return ds.pytorch(batch_size=batch_size, transform=self.run_policy_sample)

