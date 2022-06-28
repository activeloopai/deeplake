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

def pipeline_image(image, pipe):
  updated_image = image.numpy()
  if pipe is None:
    return image
  for fun in pipe:
    if len(fun.args)!=0:
      arr = [*fun.args]
      arr.insert(0, updated_image)
      args = tuple(arr)
      updated_image = fun.func(*args)
    else:
      updated_image = fun.func(updated_image)
  return updated_image


# #old hub_iterator
# def hub_iterator(dataloader, tensor_name, pipe):
#   #first decide the kind of pipeline then send to the kind of hub iterator.
#   for _, sample  in enumerate(dataloader):
#     image_batch = sample[tensor_name]#.cpu().detach().numpy()##to move to pipeline_images
#     num_images = image_batch.shape[0]
#     image_arr = []
#     for i in range(num_images):
#       image = image_batch[i]
#       image = pipeline_image(image, pipe)
#       image_arr.append(image)
    
#     sample[tensor_name] = torch.from_numpy(np.array(image_arr))
#     yield sample



#hub_iterator_multitensor parallel
def hub_iterator_multitensor_parallel(dataloader, pipes):
  for _, sample in enumerate(dataloader):
    for tensor in pipes[tensor]:
      pipe = pipes[tensor]
      image_batch = sample[tensor]
      num_images = image_batch.shape[0]
      image_arr = []
      for i in range(num_images):
        image = pipeline_image(image_batch[i], pipe)
        image_arr.append(image)
      sample[tensor] = torch.from_numpy(np.array(image_arr))
    yield sample
      


#hub_iterator_multitensor sequential 
def hub_iterator_multitensor_sequential(dataloader, pipes):   #pute the loader loop first, would inc speed by num_tensor*num_pipe
  for tensor in pipes.keys():
    tensor_pipes = pipes[tensor]
    for pipe in tensor_pipes:
      for _, sample in enumerate(dataloader):
        image_batch = sample[tensor]
        num_images = image_batch.shape[0]
        image_arr = []
        for i in range(num_images):
          image = pipeline_image(image_batch[i], pipe)
          image_arr.append(image)
        sample[tensor] = torch.from_numpy(np.array(image_arr))
        yield sample
  


      


class Hubloader():
  def __init__(self, loader: Dataset,  pipeline, batch_size, pipe_type):
    
    if isinstance(loader, Dataset):
      loader = loader.pytorch(batch_size = batch_size)
    
    self.dataloader = loader
    self.pipeline = pipeline
    self.pipe_type = pipe_type
    # self.tensor_name = tensor_name


  def __iter__(self):
    if self.pipe_type == "sequential":
      return hub_iterator_multitensor_sequential(self.dataloader, self.pipeline)
    elif self.pipe_type == "parallel":
      return hub_iterator_multitensor_parallel(self.dataloader, self.pipeline)




class Augment():
  def __init__(self, pipeline: List[ComputeFunction], pipe_type):
    self.pipeline = pipeline
    self.pipe_type = pipe_type
    # self.hub_loader = Hubloader(data_in, pipeline)
  def __call__(self, loader, batch_size = 1):
    return Hubloader(loader, self.pipeline, batch_size, self.pipe_type)
  # def get_policies(self, policy):
  #   if policy == "imagenet":
  #     return



class Augmenter():
  def __init__(self):
    self.pipe_dict = {}
    pass
  def add_step(self, input_tensors , step_transform):
    for tensor in input_tensors:
      if tensor not in self.pipe_dict.keys():
        self.pipe_dict[tensor] = [step_transform]
      else:
        self.pipe_dict[tensor].append(step_transform)
  
  def augment(self, loader, batch_size=1):
    return Hubloader(loader, self.pipe_dict, batch_size, pipe_type="sequential")
  
#make policy a class and decorate one method with hub compute and use that to call onto other methods, use that to interact with hubloader and pipeline_image



class Policy():


  def __init__(self, policy_input="image net", policy_name="image net"):
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
    

  # def __iter__(self):
  #   if self.loader == None:
  #     raise Exception("Loader not initialized. ")
  #   for _, sample in enumerate(self.loader):
  #     images = sample[self.tensor]
  #     transformed_batch = []
  #     for i in range(images.shape[0]):
  #       transformed_batch.append(self.run_policy(images[i].numpy()))
  #     sample[self.tensor] = torch.from_numpy(np.array(transformed_batch))
  #     yield sample



  # def initialize_loader(self, loader, tensor = "images" ):
  #   self.loader = loader  
  #   self.tensor = tensor
  #   if isinstance(loader, Dataset):
  #     self.loader = loader.pytorch()


  def return_dataloader(self, ds, tensor="images", batch_size = 1):
    self.tensor = tensor
    return ds.pytorch(batch_size=batch_size, transform=self.run_policy_sample)
