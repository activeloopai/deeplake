<p 
Hub provides fast access to the state-of-the-art datasets for Deep Learning, enabling data scientists to manage them, build scalable data pipelines and connect to Pytorch and Tensorflow 


### Contributors

[![](https://sourcerer.io/fame/davidbuniat/activeloopai/Hub/images/0)](https://sourcerer.io/fame/davidbuniat/activeloopai/Hub/links/0)[![](https://sourcerer.io/fame/davidbuniat/activeloopai/Hub/images/1)](https://sourcerer.io/fame/davidbuniat/activeloopai/Hub/links/1)[![](https://sourcerer.io/fame/davidbuniat/activeloopai/Hub/images/2)](https://sourcerer.io/fame/davidbuniat/activeloopai/Hub/links/2)[![](https://sourcerer.io/fame/davidbuniat/activeloopai/Hub/images/3)](https://sourcerer.io/fame/davidbuniat/activeloopai/Hub/links/3)[![](https://sourcerer.io/fame/davidbuniat/activeloopai/Hub/images/4)](https://sourcerer.io/fame/davidbuniat/activeloopai/Hub/links/4)[![](https://sourcerer.io/fame/davidbuniat/activeloopai/Hub/images/5)](https://sourcerer.io/fame/davidbuniat/activeloopai/Hub/links/5)[![](https://sourcerer.io/fame/davidbuniat/activeloopai/Hub/images/6)](https://sourcerer.io/fame/davidbuniat/activeloopai/Hub/links/6)[![](https://sourcerer.io/fame/davidbuniat/activeloopai/Hub/images/7)](https://sourcerer.io/fame/davidbuniat/activeloopai/Hub/links/7)


### Problems with Current Workflows

We realized that there are a few problems related with current workflow in deep learning data management through our experience of working with deep learning companies and researchers. Most of the time Data Scientists/ML researchers work on data management and preprocessing instead of doing modeling. Deep Learning often requires to work with large datasets. Those datasets can grow up to terabyte or even petabyte size.  It is hard to manage data, store, access, and version-control. It is time-consuming to download the data and link with the training or inference code. There is no easy way to access a chunk of it and possibly visualize. Wouldn’t it be more convenient to have large datasets stored & version-controlled as single numpy-like array on the cloud and have access to it from any machine at scale?

## Getting Started

### Access public data. Fas

We’ve talked the talk, now let’s walk through how it works:
```sh
pip3 install hub
```

You can access public datasets with a few lines of code.
```python
import hub

mnist = hub.load("mnist/mnist")
mnist["data"][0:1000].compute()
```



Load the data and directly train your model using pytorch

```python
import hub
import torch

mnist = hub.load("mnist/mnist")
mnist = mnist.to_pytorch(lambda x: (x["data"], x["labels"]))

train_loader = torch.utils.data.DataLoader(mnist, batch_size=1, num_workers=0)

for image, label in train_loader:
    # Training loop here
```

### Upload your dataset and access it from <ins>anywhere</ins> in 3 simple steps

1. Register a free account at [Activeloop](http://app.activeloop.ai) and authenticate locally
```sh
hub register
hub login
```

2. Then create a dataset and upload
```python
from hub import tensor, dataset

images = tensor.from_array(np.zeros((4, 512, 512)))
labels = tensor.from_array(np.zeros((4, 512, 512)))

<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/snarkai/Hub/master/docs/logo/hub_logo.png" width="50%"/>
    </br>
</p>
<p align="center">
    <a href="http://docs.activeloop.ai/">
        <img alt="Docs" src="https://readthedocs.org/projects/hubdb/badge/?version=latest">
    </a>
    <a href="https://pypi.org/project/hub/"><img src="https://badge.fury.io/py/hub.svg" alt="PyPI version" height="18"></a>
    <a href="https://pypi.org/project/hub/"><img src="https://img.shields.io/pypi/dm/hub.svg" alt="PyPI version" height="18"></a>
    <a href="https://codecov.io/gh/activeloopai/Hub/branch/master"><img src="https://codecov.io/gh/activeloopai/Hub/branch/master/graph/badge.svg" alt="codecov" height="18"></a>
    <a href="https://twitter.com/intent/tweet?text=The%20fastest%20way%20to%20access%20and%20manage%20PyTorch%20and%20Tensorflow%20datasets%20is%20open-source&url=https://activeloop.ai/&via=activeloopai&hashtags=opensource,pytorch,tensorflow,data,datascience,datapipelines,sqlforimages,activeloop"> 
        <img alt="tweet" src="https://img.shields.io/twitter/url/http/shields.io.svg?style=social">
    </a>
  </a>
</p>
<h3 align="center">
The fastest way to access and manage datasets for PyTorch and TensorFlow
</h3>




