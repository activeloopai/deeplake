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

class Augmenter():    #used to be Pipeline
  def __init__(self):
    self.pipe_dict = {}
    pass

  def add_step(self, input_tensors , step_transform):
    for tensor in input_tensors:
      if tensor not in self.pipe_dict.keys():
        self.pipe_dict[tensor] = [step_transform]
      else:
        self.pipe_dict[tensor].append(step_transform)

  def augment(self, ds, num_workers=1):
    pipe_dict = self.pipe_dict.copy()
    return ds.pytorch(transform=pipe_dict, multiple_transforms=True, num_workers=num_workers)

class Policy():
  def __init__(self, policy_input="image net"):#policy_name="image net"
    if isinstance(policy_input,str):
      self.policy_set = self.get_policies(policy_input)
    else:
      self.policy_set = policy_input
    self.transforms_lis = None


  def get_policies(self, policy_name):

    if policy_name == "image net":
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
    num_policies = len(self.policy_set)
    policy_id = np.random.randint(num_policies)
    policy = self.policy_set[policy_id]
    probs = np.random.rand(3)
    for i in range(len(policy)):
      if probs[i] < policy[i][1]:
        image = run_transform(image, policy[i][0], policy[i][2])
    return image
  

  def run_policy_sample(self, sample):
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

  def return_dataloader(self, ds, tensor="images", batch_size = 1, num_workers = None):
    self.tensor = tensor
    if num_workers is not None:
      return ds.pytorch(batch_size=batch_size, transform=self.run_policy_sample, num_workers = num_workers)
    return ds.pytorch(batch_size=batch_size, transform=self.run_policy_sample)

