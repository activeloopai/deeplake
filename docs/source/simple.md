# Getting Started with Hub



## Access public data. Fast

We’ve talked the talk, now let’s walk through how it works: 

```
pip3 install hub
```

You can access public datasets with a few lines of code.

```python
import hub

mnist = hub.load("mnist/mnist")
mnist["data"][0:1000].compute()
```


## Train a model

Load the data and directly train your model using Pytorch

```python
import hub
import pytorch

cifar = hub.load("cifar/cifar10")
cifar = cifar.to_pytorch()

train_loader = torch.utils.data.DataLoader(
      cifar, batch_size=1, num_workers=0, collate_fn=cifar.collate_fn
)

for images, labels in train_loader:
   # your training loop here
```

## Upload your dataset and access it from anywhere

Register a free account at [Activeloop](https://app.activeloop.ai)

```
hub login
```

Then create a dataset and upload

```python

from hub import tensor, dataset

images = tensor.from_array(np.zeros((4, 512, 512)))
labels = tensor.from_array(np.zeros((4, 512, 512)))

ds = dataset.from_tensors({"images": images, "labels": labels})
ds.store("username/basic")

# Access it from anywhere else in the world
import hub
ds = hub.load("username/basic")
```