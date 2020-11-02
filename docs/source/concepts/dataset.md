# Dataset

Hub Datasets are dictionaries containing tensors. You can think of them as folders in the cloud. To store tensor in the cloud we first should put it in dataset and then store the dataset. 


## Store
To create and store dataset you would need to define tensors and specify the dataset dictionary. 

```python
from hub import dataset, tensor

tensor1 = tensor.from_zeros((20,512,512), dtype="uint8", dtag="image")
tensor2 = tensor.from_zeros((20), dtype="bool", dtag="label")

dataset.from_tensors({"name1": tensor1, "name2": tensor2})

dataset.store("username/namespace")
```

## Load

To load a dataset from a central repository

```python
from hub import dataset

ds = dataset.load("mnist/mnist")
```

## Combine

You could combine datasets or concat them.

```python
from hub import dataset

... 

#vertical
dataset.concat(ds1, ds2)

#horizontal
dataset.combine(ds1, ds2)
```

## Get text labels
To get text labels from a dataset  

###### Pytorch

```python
from hub import dataset
import torch

ds = dataset.load("mnist/fashion-mnist")

ds = ds.to_pytorch()

data_loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=ds.collate_fn)

for batch in data_loader:
    tl = dataset.get_text(batch['named_label'])
```
    
###### Tensorflow  

```python
from hub import dataset
import tensorflow as tf

ds = dataset.load("mnist/fashion-mnist")

ds = ds.to_tensorflow()

dataset = ds.batch(BATCH_SIZE)

for batch in dataset:
    tl = dataset.get_text(batch['named_label'])
```

## How to Upload a Dataset

For small datasets that would fit into your RAM you can directly upload by converting a numpy array into hub tensor. For complete example please check [Uploading MNIST](https://github.com/activeloopai/Hub/blob/master/examples/mnist/upload.py) and [Uploading CIFAR](https://github.com/activeloopai/Hub/blob/master/examples/cifar/upload_cifar10.py)

For larger datasets you would need to define a dataset generator and apply the transformation iteratively. Please see an example below [Uploading COCO](https://github.com/activeloopai/Hub/blob/master/examples/coco/upload_coco2017.py).
Please pay careful attention to `meta(...)` function where you describe each tensor properties. Please pay careful attention providing full meta description including shape, dtype, dtag, chunk_shape etc.

### Dtag
For each tensor you would need to specify a dtag so that visualizer knows how draw it or transformations have context how to transform it.

| Dtag          |      Shape      |  Types  |
|---------------|:---------------:|--------:|
| default       |    any array    |   any   |
| image         |    (width, height), (channel, width, height) or (width, height, channel)                  | int, float |
| text          |   used for label   | str or object  |
| box           |  [(4)]          |   int32   |
| mask          | (width, height) |    bool  |
| segmentation  | (width, height), (channel, width, height) or (width, height, channel)|   int  |
| video          |     (sequence, width, height, channel) or (sequence, channel, width, height)          |    int, float      |
| embedding      |               |          |
| tabular        |               |          |
| time    |               |          |
| event     |               |          |
| audio          |               |          |
| pointcloud    |               |          |
| landmark      |               |          |
| polygon        |               |          |
| mesh           |               |          |
| document       |               |          |



### Guidelines
1. Fork the github repo and create a folder under `examples/dataset`

2. Train a model using Pytorch

```python
import hub
import pytorch

ds = hub.load("username/dataset")
ds = ds.to_pytorch()

# Implement a training loop for the dataset in pytorch
...
```

3. Train a model using Tensorflow 

```python
import hub
import tensorflow

ds = hub.load("username/dataset")
ds = ds.to_tensorflow()

# Implement a training loop for the dataset in tensorflow
...
```

4. Make sure visualization works perfectly at [app.activeloop.ai](https://app.activeloop.ai)

### Final Checklist
So here is the checklist, the pull request.
- Accessible using the sdk
- Trainable on Tensorflow
- Trainable on PyTorch 
- Visualizable at [app.activeloop.ai](https://app.activeloop.ai)
- Pull Request merged into master

### Issues

If you spot any trouble or have any question, please open a github issue.


## API

```eval_rst
.. autoclass:: hub.dataset.Dataset
   :members:
```
