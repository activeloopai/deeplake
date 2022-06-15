from hub.core.dataset import Dataset
from typing import List
from hub.core.transform.transform import ComputeFunction 
from torch.utils.data.dataloader import DataLoader
# import torch
# import cv2 as cv
# from PIL import Image
# import numpy as np


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


def hub_iterator(dataloader, pipe):
  for x, y in dataloader:
    image = x.cpu().detach().numpy().squeeze()
    if pipe == None:
      yield (image, y)
    else:
      yield(pipeline(image, pipe), y)

class Hubloader():
  def __init__(self, loader: Dataset, pipeline=None):
    if loader is not DataLoader:
      loader = loader.pytorch()
    self.dataloader = loader
    self.pipeline = pipeline

  def __iter__(self):
    return hub_iterator(self.dataloader, self.pipeline)



class Augment():
  def __init__(self, pipeline: List[ComputeFunction]):
    self.pipeline = pipeline
    # self.hub_loader = Hubloader(data_in, pipeline)
  def __call__(self, loader):
    return Hubloader(loader, self.pipeline)