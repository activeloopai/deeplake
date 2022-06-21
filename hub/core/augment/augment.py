from hub.core.dataset import Dataset
from typing import List
from hub.core.transform.transform import ComputeFunction 
from torch.utils.data.dataloader import DataLoader
import torch
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



class Pipeline():
  def __init__(self):
    self.pipe_dict = {}
    pass
  def add_step(self, input_tensors , pipe):
    for tensor in input_tensors:
      if tensor not in self.pipe_dict.keys():
        self.pipe_dict[tensor] = [pipe]
      else:
        self.pipe_dict[tensor].append(pipe)
  
  def augment(self, loader, batch_size=1):
    return Hubloader(loader, self.pipe_dict, batch_size, pipe_type="sequential")
