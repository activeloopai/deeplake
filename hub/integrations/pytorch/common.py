from typing import Callable, Dict, List, Optional
from hub.util.iterable_ordered_dict import IterableOrderedDict
import numpy as np
import torch 


def pipeline_image(image, pipe): #could not import due to circular impoet error
  if not isinstance(image, np.ndarray):
    updated_image = image.numpy()
  else:
    updated_image = image
  if pipe is None:
    return image
  if len(image.shape) == 2:
    updated_image = np.expand_dims(updated_image, axis=2)
  for fun in pipe:
    if len(fun.args)!=0:
      arr = [*fun.args]
      arr.insert(0, updated_image)
      args = tuple(arr)
      updated_image = fun.func(*args)
    else:
      updated_image = fun.func(updated_image)
  updated_image = torch.from_numpy(updated_image).permute(2,0,1)
  return updated_image


def collate_fn(batch):
    import torch

    elem = batch[0]

    if isinstance(elem, IterableOrderedDict):
        return IterableOrderedDict(
            (key, collate_fn([d[key] for d in batch])) for key in elem.keys()
        )

    if isinstance(elem, np.ndarray) and elem.size > 0 and isinstance(elem[0], str):
        batch = [it[0] for it in batch]
    return torch.utils.data._utils.collate.default_collate(batch)

    
def convert_fn(data):
    import torch

    if isinstance(data, IterableOrderedDict):
        return IterableOrderedDict((k, convert_fn(v)) for k, v in data.items())
    if isinstance(data, np.ndarray) and data.size > 0 and isinstance(data[0], str):
        data = data[0]

    return torch.utils.data._utils.collate.default_convert(data)


def transform_sample(data, transform):
    transformed_samples = []
    for tensor in transform.keys():
        tensor_pipes = transform[tensor]
        for transformation in tensor_pipes:
            label_condition = transformation[1]
            if label_condition==None or label_condition(data) == True:
                transformed_sample = data.copy()    
                transformed_sample[tensor] = pipeline_image(data[tensor], transformation[0])
                transformed_samples.append(transformed_sample)
    return transformed_samples
            

class PytorchTransformFunction:
    def __init__(
        self,
        transform_dict: Optional[Dict[str, Optional[Callable]]] = None,
        composite_transform: Optional[Callable] = None,
    ) -> None:
        self.composite_transform = composite_transform
        self.transform_dict = transform_dict

    def __call__(self, data_in: Dict, multiple_transforms=False) -> Dict:
        if self.composite_transform is not None:
            return self.composite_transform(data_in)
        elif self.transform_dict is not None and multiple_transforms==False:
            data_out = {}
            for tensor, fn in self.transform_dict.items():
                value = data_in[tensor]
                data_out[tensor] = value if fn is None else fn(value)
            data_out = IterableOrderedDict(data_out)
            return data_out
        elif self.transform_dict != None and multiple_transforms == True:
            transformed_samples = transform_sample(data_in, self.transform_dict)
            return transformed_samples
        return data_in
        