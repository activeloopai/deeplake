# How to Upload a Dataset

1. First install the package, create an account and then authenticate.
```
> pip3 install hub
> hub register
> hub login
```

2. Then create your dataset
```python
from hub import tensor, dataset

images = tensor.from_array(np.zeros((4, 512, 512)))
labels = tensor.from_array(np.zeros((4, 512, 512)))

ds = dataset.from_tensors({"images": images, "labels": labels})
ds.store("username/dataset") # Upload
```


## Notes 

For small datasets that would fit into your RAM you can directly upload by converting a numpy array into hub tensor. For complete example please check [Uploading MNIST](https://github.com/activeloopai/Hub/blob/master/examples/old/mnist/upload.py) and [Uploading CIFAR](https://github.com/activeloopai/Hub/blob/master/examples/old/cifar/upload_cifar10.py)

For larger datasets you would need to define a dataset generator and apply the transformation iteratively. Please see an example below [Uploading COCO](https://github.com/activeloopai/Hub/blob/master/examples/old/coco/upload_coco2017.py).
Please pay careful attention to `meta(...)` function where you describe each tensor properties. Please pay careful attention providing full meta description including shape, dtype, dtag, chunk_shape etc.

## Dtag
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



## Guidelines
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

## Final Checklist
So here is the checklist, the pull request.
-  [ ] Accessible using the sdk
-  [ ] Trainable on Tensorflow
-  [ ] Trainable on PyTorch 
-  [ ] Visualizable at [app.activeloop.ai](https://app.activeloop.ai)
-  [ ] Pull Request merged into master

## Issues

If you spot any trouble or have any question, please open a github issue.
