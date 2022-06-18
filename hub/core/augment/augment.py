from hub.core.dataset import Dataset
from typing import List
from hub.core.transform.transform import ComputeFunction 
from torch.utils.data.dataloader import DataLoader
import torch
# import cv2 as cv
# from PIL import Image
import numpy as np
import time


def pipeline(image, pipe):
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


def hub_iterator(dataloader,tensor_name, pipe):
  for _, sample  in enumerate(dataloader):
    image_batch = sample[tensor_name].cpu().detach().numpy()
    num_images = image_batch.shape[0]
    image_arr = []
    for i in range(num_images):
      image = image_batch[i]
      image = pipeline(image, pipe)
      image_arr.append(image)
    
    sample[tensor_name] = torch.from_numpy(np.array(image_arr))
    yield sample

class Hubloader():
  def __init__(self, loader: Dataset, tensor_name: str, pipeline, batch_size):
    
    if isinstance(loader, Dataset):
      loader = loader.pytorch(batch_size = batch_size)
      
    
    self.dataloader = loader
    self.pipeline = pipeline
    self.tensor_name = tensor_name


  def __iter__(self):
    return hub_iterator(self.dataloader, self.tensor_name, self.pipeline)



class Augment():
  def __init__(self, pipeline: List[ComputeFunction]):
    self.pipeline = pipeline
    # self.hub_loader = Hubloader(data_in, pipeline)
  def __call__(self, loader, tensor_name="images", batch_size = 1):
    return Hubloader(loader, tensor_name, self.pipeline, batch_size)