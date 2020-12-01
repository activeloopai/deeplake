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
    <a href="https://app.circleci.com/pipelines/github/activeloopai/Hub">
    <img alt="CircleCI" src="https://img.shields.io/circleci/build/github/activeloopai/Hub?logo=circleci"> </a>
     <a href="https://github.com/activeloopai/Hub/issues">
    <img alt="GitHub issues" src="https://img.shields.io/github/issues/activeloopai/Hub"> </a>
    <a href="https://codecov.io/gh/activeloopai/Hub/branch/master"><img src="https://codecov.io/gh/activeloopai/Hub/branch/master/graph/badge.svg" alt="codecov" height="18"></a>
    <a href="https://twitter.com/intent/tweet?text=The%20fastest%20way%20to%20access%20and%20manage%20PyTorch%20and%20Tensorflow%20datasets%20is%20open-source&url=https://activeloop.ai/&via=activeloopai&hashtags=opensource,pytorch,tensorflow,data,datascience,datapipelines,activeloop,dockerhubfordatasets"> 
        <img alt="tweet" src="https://img.shields.io/twitter/url/http/shields.io.svg?style=social"> </a>  
   </br> 
    <a href="https://join.slack.com/t/hubdb/shared_invite/zt-ivhsj8sz-GWv9c5FLBDVw8vn~sxRKqQ">
  <img src="https://user-images.githubusercontent.com/13848158/97266254-9532b000-1841-11eb-8b06-ed73e99c2e5f.png" height="35" />
    </a>

---

</a>
</p>


 <img src="https://img.shields.io/badge/-news-red"> Access and visualize 200 of world's most popular datasets in under a few minutes instead of hours with Hub. Read below.

<h3 align="center"> The Docker Hub for datasets. </h3>
<h4 align="center"> Hub is the fastest way to access & manage datasets for PyTorch and TensorFlow, and build scalable data pipelines.</h4>

---

### What is Hub for?

Most of the time Data Scientists/ML researchers work on data management and preprocessing instead of training models. With Hub, we are fixing this. We store your (even petabyte-scale) datasets as single numpy-like array on the cloud, so you can seamlessly access and work with it from any machine. Hub makes any data type (images, text files, audio, or video) stored in cloud usable as fast as if it were stored on premise. With same dataset view, your team can always be in sync. 

### Features 

* Store and retrieve large datasets with version-control
* Collaborate as in Google Docs: Multiple data scientists working on the same data in sync with no interruptions
* Access from multiple machines simultaneously
* Integrate with your ML tools like Numpy, Dask, [PyTorch](https://docs.activeloop.ai/en/latest/integrations/pytorch.html), or [TensorFlow](https://docs.activeloop.ai/en/latest/integrations/tensorflow.html).
* Create arrays as big as you want
* Visualize any slice of the data in a matter of seconds without redundant manipulations.

## Getting Started

### Access public data. Fast

To load a public dataset, one needs to write dozens of lines of code and spend hours accessing and understanding the API, as well as downloading the data. With Hub, you only need 2 lines of code, and you **can get started working on your dataset in under 3 minutes**. 

```sh
pip3 install hub
```

You can access public datasets with a few lines of code.
```python
import hub

mnist = hub.load("mnist/mnist")
mnist["data"][0:1000].compute()
```

### Train a model

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
from hub import Dataset, features
import numpy as np

ds = Dataset(
    "username/basic",
    schema={
        "image": features.Tensor((512, 512), dtype="float"),
        "label": features.Tensor((512, 512), dtype="float"),
    },
)

ds["image"][:] = np.zeros((4, 512, 512))
ds["label"][:] = np.zeros((4, 512, 512))
ds.commit()
```

3. Access it from anywhere else in the world, on any device having a command line.
```python
import hub

ds = hub.load("username/basic")
```
### Look at Hub in action on Google Colab
- MNIST Classification with Hub and PyTorch  
&nbsp;
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1LUeZG20A4X4WZX2AYHdI4F6InG6Jb51i?usp=sharing)

## Documentation

For more advanced data pipelines like uploading large datasets or applying many transformations, please read the [docs](http://docs.activeloop.ai).

## Community

Join our [Slack community](https://join.slack.com/t/hubdb/shared_invite/zt-ivhsj8sz-GWv9c5FLBDVw8vn~sxRKqQ) for help from Activeloop team and other users as well as dataset management/preprocessing tips and tricks.

<img alt="tweet" src="https://img.shields.io/twitter/follow/activeloopai?label=stay%20in%20the%20Loop&style=social"> on Twitter.

As always, thanks to our amazing contributors!
[![](https://sourcerer.io/fame/davidbuniat/activeloopai/Hub/images/0)](https://sourcerer.io/fame/davidbuniat/activeloopai/Hub/links/0)[![](https://sourcerer.io/fame/davidbuniat/activeloopai/Hub/images/1)](https://sourcerer.io/fame/davidbuniat/activeloopai/Hub/links/1)[![](https://sourcerer.io/fame/davidbuniat/activeloopai/Hub/images/2)](https://sourcerer.io/fame/davidbuniat/activeloopai/Hub/links/2)[![](https://sourcerer.io/fame/davidbuniat/activeloopai/Hub/images/3)](https://sourcerer.io/fame/davidbuniat/activeloopai/Hub/links/3)[![](https://sourcerer.io/fame/davidbuniat/activeloopai/Hub/images/4)](https://sourcerer.io/fame/davidbuniat/activeloopai/Hub/links/4)[![](https://sourcerer.io/fame/davidbuniat/activeloopai/Hub/images/5)](https://sourcerer.io/fame/davidbuniat/activeloopai/Hub/links/5)[![](https://sourcerer.io/fame/davidbuniat/activeloopai/Hub/images/6)](https://sourcerer.io/fame/davidbuniat/activeloopai/Hub/links/6)[![](https://sourcerer.io/fame/davidbuniat/activeloopai/Hub/images/7)](https://sourcerer.io/fame/davidbuniat/activeloopai/Hub/links/7)

## Use Cases
* **Aerial images**: [Satellite and drone imagery](https://activeloop.ai/usecase/intelinair)
* **Medical Images**: Volumetric images such as MRI or Xray
* **Self-Driving Cars**: [Radar, 3D LIDAR, Point Cloud, Semantic Segmentation, Video Objects](https://medium.com/snarkhub/extending-snark-hub-capabilities-to-handle-waymo-open-dataset-4dc7b7d8ab35)
* **Retail**: Self-checkout datasets
* **Media**: Images, Video, Audio storage

## Examples
Activeloop’s Hub format lets you achieve faster inference at a lower cost. Test out the datasets we’ve converted into Hub format - see for yourself!
- [Waymo Open Dataset](https://medium.com/snarkhub/extending-snark-hub-capabilities-to-handle-waymo-open-dataset-4dc7b7d8ab35)
- [Aptiv nuScenes](https://medium.com/snarkhub/snark-hub-is-hosting-nuscenes-dataset-for-autonomous-driving-1470ae3e1923)


## Disclaimers

Similarly to other dataset management packages, `Hub` is a utility library that downloads and prepares public datasets. We do not host or distribute these datasets, vouch for their quality or fairness, or claim that you have license to use the dataset. It is your responsibility to determine whether you have permission to use the dataset under the dataset's license.

If you're a dataset owner and wish to update any part of it (description, citation, etc.), or do not want your dataset to be included in this library, please get in touch through a [GitHub issue](https://github.com/activeloopai/Hub/issues/new). Thanks for your contribution to the ML community!
