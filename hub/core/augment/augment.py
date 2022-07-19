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
  if not isinstance(image, np.ndarray):
    updated_image = image.numpy()
  else:
    updated_image = image
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


def pipeline_image_sample(sample, pipe):  #meant for ds.pytorch
  images = sample[images]
  new_sample_images = []
  for image in images:
    new_sample_images.append(pipeline_image(image, pipe))
  new_sample_images = torch.from_numpy(np.array(new_sample_images))
  sample[images] = new_sample_images
  return sample


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
def hub_iterator_multitensor_sequential(dataloader, pipes):  
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

        

  



  if num_workers is not None:
    return ds.pytorch(batch_size=batch_size, transform=self.run_policy_sample, num_workers = num_workers)
  return ds.pytorch(batch_size=batch_size, transform=self.run_policy_sample)


      


class Hubloader():
  def __init__(self, loader: Dataset, pipeline, batch_size, pipe_type):
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




# class Augment():  
#   def __init__(self, pipeline: List[ComputeFunction], pipe_type):
#     self.pipeline = pipeline
#     self.pipe_type = pipe_type
#   def __call__(self, loader, batch_size = 1):
#     return Hubloader(loader, self.pipeline, batch_size, self.pipe_type)




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
  
  def augment(self, loader, batch_size=1):
    return Hubloader(loader, self.pipe_dict, batch_size, pipe_type="sequential")

  def use_pytorch(self, ds):
    pipe_dict = self.pipe_dict.copy()
    return ds.pytorch(transform=pipe_dict, multiple_transforms=True)

  def pipeline_image_sample(self, sample):  #meant for ds.pytorch
    pipe = self.pipe
    images = sample[self.tensor]
    new_sample_images = []
    for image in images:
      new_sample_images.append(pipeline_image(image, pipe))
    new_sample_images = torch.from_numpy(np.array(new_sample_images))
    sample[images] = new_sample_images
    return sample

  def return_generator(self, ds: Dataset, batch_size=1, num_workers=None):
    pipes = self.pipe_dict
    for tensor in pipes:
      self.tensor = tensor
      tensor_pipes = pipes[tensor]
      for pipe in tensor_pipes:
        self.pipe = pipe
        if num_workers is not None:
          dataloader = ds.pytorch(batch_size=batch_size, transform=self.pipeline_image_sample, num_workers = num_workers)
        else:
          dataloader = ds.pytorch(batch_size=batch_size, transform=self.pipeline_image_sample)
        for _, sample in enumerate(dataloader):
          yield sample

#make policy a class and decorate one method with hub compute and use that to call onto other methods, use that to interact with hubloader and pipeline_image




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


  def return_dataloader(self, ds, tensor="images", batch_size = 1, num_workers = None):
    self.tensor = tensor
    if num_workers is not None:
      return ds.pytorch(batch_size=batch_size, transform=self.run_policy_sample, num_workers = num_workers)
    return ds.pytorch(batch_size=batch_size, transform=self.run_policy_sample)

