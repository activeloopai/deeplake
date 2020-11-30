# Tensor

Hub Tensors are scalable NumPy-like arrays stored on the cloud accessible over the internet as if they’re local NumPy arrays. Their chunkified structure makes it super fast to interact with them.

Tensor represents a single array containing homogeneous data type. It could contain a list of text files, audio, images, or video data. The first dimension represents the batch dimension. 

One can specify `dtag` for the element to specify its nature.


## Initialize
You can initialize a tensor like this and get the first element.
```python
from hub import tensor

t = tensor.from_zeros((10, 512, 512), dtype="uint8")
t[0].compute()
```

You can also initialize the tensor object from a numpy array.

```python
import numpy as np
from hub import tensor

t = tensor.from_zeros(np.zeros((10, 512, 512)))
```


## Concat or Stack

Concatenating or stacking tensors works as in other frameworks.

```python
from hub import tensor

t1 = tensor.from_zeros((10, 512, 512), dtype="uint8")
t2 = tensor.from_zeros((20, 512, 512), dtype="uint8")
tensors = [t1, t2]

tensor.concat(tensors, axis=0, chunksize=-1)
tensor.stack(tensors, axis=0, chunksize=-1)
```

## API
```eval_rst
.. autoclass:: hub.dataset.Tensor
   :members:
```
