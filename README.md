 <p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/activeloopai/Hub/master/docs/logo/logo-explainer-bg.png" width="50%"/>
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
  <img src="https://user-images.githubusercontent.com/13848158/97266254-9532b000-1841-11eb-8b06-ed73e99c2e5f.png" height="35" /> </a>

---

</a>
</p>

<h3 align="center"> Introducing Data 2.0, powered by Hub. </br>The fastest way to access & manage datasets for PyTorch/TensorFlow, and build scalable data pipelines.</h3>

---

[ English | [简体中文](./README_CN.md) ]

### What is Hub for?

Software 2.0 needs Data 2.0, and Hub delivers it. Most of the time Data Scientists/ML researchers work on data management and preprocessing instead of training models. With Hub, we are fixing this. We store your (even petabyte-scale) datasets as single numpy-like array on the cloud, so you can seamlessly access and work with it from any machine. Hub makes any data type (images, text files, audio, or video) stored in cloud usable as fast as if it were stored on premise. With same dataset view, your team can always be in sync. 

Hub is being used by Waymo, Red Cross, World Resources Institute, Omdena, and others.

### Features 

* Store and retrieve large datasets with version-control
* Collaborate as in Google Docs: Multiple data scientists working on the same data in sync with no interruptions
* Access from multiple machines simultaneously
* Deploy anywhere - locally, on Google Cloud, S3, Azure as well as Activeloop (by default - and for free!) 
* Integrate with your ML tools like Numpy, Dask, Ray, [PyTorch](https://docs.activeloop.ai/en/latest/integrations/pytorch.html), or [TensorFlow](https://docs.activeloop.ai/en/latest/integrations/tensorflow.html)
* Create arrays as big as you want. You can store images as big as 100k by 100k!
* Keep shape of each sample dynamic. This way you can store small and big arrays as 1 array. 
* [Visualize](http://app.activeloop.ai/?utm_source=github&utm_medium=repo&utm_campaign=readme) any slice of the data in a matter of seconds without redundant manipulations

## Getting Started

### Access public data. FAST

To load a public dataset, one needs to write dozens of lines of code and spend hours accessing and understanding the API as well as downloading the data. With Hub, you only need 2 lines of code, and you **can get started working on your dataset in under 3 minutes**.

```sh
pip3 install hub
```

Access public datasets in Hub by following a straight-forward convention which merely requires a few lines of simple code. Run this excerpt to get the first thousand images in the [MNIST database](https://app.activeloop.ai/dataset/activeloop/mnist/?utm_source=github&utm_medium=repo&utm_campaign=readme) in the numpy array format:
```python
from hub import Dataset

mnist = Dataset("activeloop/mnist")  # loading the MNIST data lazily
# saving time with *compute* to retrieve just the necessary data
mnist["image"][0:1000].compute()
```
You can find all the other popular datasets on [app.activeloop.ai](https://app.activeloop.ai/datasets/popular/?utm_source=github&utm_medium=repo&utm_campaign=readme).

### Train a model

Load the data and train your model **directly**. Hub is integrated with PyTorch and TensorFlow and performs conversions between formats in an understandable fashion. Take a look at the example with PyTorch below:

```python
from hub import Dataset
import torch

mnist = Dataset("activeloop/mnist")
# converting MNIST to PyTorch format
mnist = mnist.to_pytorch(lambda x: (x["image"], x["label"]))

train_loader = torch.utils.data.DataLoader(mnist, batch_size=1, num_workers=0)

for image, label in train_loader:
    # Training loop here
```

### Create a local dataset 
If you want to work on your own data locally, you can start by creating a dataset:
```python
from hub import Dataset, schema
import numpy as np

ds = Dataset(
    "./data/dataset_name",  # file path to the dataset
    shape = (4,),  # follows numpy shape convention
    mode = "w+",  # reading & writing mode
    schema = {  # named blobs of data that may specify types
    # Tensor is a generic structure that can contain any type of data
        "image": schema.Tensor((512, 512), dtype="float"),
        "label": schema.Tensor((512, 512), dtype="float"),
    }
)

# filling the data containers with data (here - zeroes to initialize)
ds["image"][:] = np.zeros((4, 512, 512))
ds["label"][:] = np.zeros((4, 512, 512))
ds.commit()  # executing the creation of the dataset
```

You can also specify `s3://bucket/path`, `gcs://bucket/path` or azure path. [Here](https://docs.activeloop.ai/en/latest/simple.html#data-storage) you can find more information on cloud storage.
Also, if you need a publicly available dataset that you cannot find in the Hub, you may [file a request](https://github.com/activeloopai/Hub/issues/new?assignees=&labels=i%3A+enhancement%2C+i%3A+needs+triage&template=feature_request.md&title=[FEATURE]+New+Dataset+Required%3A+%2Adataset_name%2A). We will enable it for everyone as soon as we can!

### Upload your dataset and access it from <ins>anywhere</ins> in 3 simple steps

1. Register a free account at [Activeloop](https://app.activeloop.ai/register/?utm_source=github&utm_medium=repo&utm_campaign=readme) and authenticate locally:
```sh
hub register
hub login

# alternatively, add username and password as arguments (use on platforms like Kaggle)
hub login -u username -p password
```

2. Then create a dataset, specifying its name and upload it to your account. For instance:
```python
from hub import Dataset, schema
import numpy as np

ds = Dataset(
    "username/dataset_name",
    shape = (4,),
    mode = "w+",
    schema = {
        "image": schema.Tensor((512, 512), dtype="float"),
        "label": schema.Tensor((512, 512), dtype="float"),
    }
)

ds["image"][:] = np.zeros((4, 512, 512))
ds["label"][:] = np.zeros((4, 512, 512))
ds.commit()
```

3. Access it from anywhere else in the world, on any device having a command line:
```python
from hub import Dataset

ds = Dataset("username/dataset_name")
```


## Documentation

For more advanced data pipelines like uploading large datasets or applying many transformations, please refer to our [documentation](http://docs.activeloop.ai/?utm_source=github&utm_medium=repo&utm_campaign=readme).

## Use Cases
* **Satellite and drone imagery**: [Smarter farming with scalable aerial pipelines](https://activeloop.ai/usecase/intelinair?utm_source=github&utm_medium=repo&utm_campaign=readme), [Mapping Economic Well-being in India](https://towardsdatascience.com/faster-machine-learning-using-hub-by-activeloop-4ffb3420c005), [Fighting desert Locust in Kenya with Red Cross](https://omdena.com/projects/ai-desert-locust/)
* **Medical Images**: Volumetric images such as MRI or Xray
* **Self-Driving Cars**: [Radar, 3D LIDAR, Point Cloud, Semantic Segmentation, Video Objects](https://medium.com/snarkhub/extending-snark-hub-capabilities-to-handle-waymo-open-dataset-4dc7b7d8ab35)
* **Retail**: Self-checkout datasets
* **Media**: Images, Video, Audio storage

## Community

Join our [**Slack community**](https://join.slack.com/t/hubdb/shared_invite/zt-ivhsj8sz-GWv9c5FLBDVw8vn~sxRKqQ) to get help from Activeloop team and other users, as well as stay up-to-date on dataset management/preprocessing best practices.

<img alt="tweet" src="https://img.shields.io/twitter/follow/activeloopai?label=stay%20in%20the%20Loop&style=social"> on Twitter.

As always, thanks to our amazing contributors!    

<a href="https://github.com/activeloopai/hub/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=activeloopai/hub" />
</a>

Made with [contributors-img](https://contrib.rocks).

Please read [CONTRIBUTING.md](CONTRIBUTING.md) to know how to get started with making contributions to Hub.

## Examples
Activeloop's Hub format lets you achieve faster inference at a lower cost. We have 30+ popular datasets already on our platform. These include:
- COCO
- CIFAR-10
- PASCAL VOC
- Cars196
- KITTI
- EuroSAT 
- Caltech-UCSD Birds 200
- Food101

Check these and many more popular datasets on our [visualizer web app](https://app.activeloop.ai/datasets/popular/?utm_source=github&utm_medium=repo&utm_campaign=readme) and load them directly for model training!

## README Badge

Using Hub? Add a README badge to let everyone know: 


[![hub](https://img.shields.io/badge/powered%20by-hub%20-ff5a1f.svg)](https://github.com/activeloopai/Hub)

```
[![hub](https://img.shields.io/badge/powered%20by-hub%20-ff5a1f.svg)](https://github.com/activeloopai/Hub)
```

## Disclaimers

Similarly to other dataset management packages, `Hub` is a utility library that downloads and prepares public datasets. We do not host or distribute these datasets, vouch for their quality or fairness, or claim that you have license to use the dataset. It is your responsibility to determine whether you have permission to use the dataset under the dataset's license.

If you're a dataset owner and wish to update any part of it (description, citation, etc.), or do not want your dataset to be included in this library, please get in touch through a [GitHub issue](https://github.com/activeloopai/Hub/issues/new). Thanks for your contribution to the ML community!


## Acknowledgement
 This technology was inspired from our experience at Princeton University and would like to thank William Silversmith @SeungLab with his awesome [cloud-volume](https://github.com/seung-lab/cloud-volume) tool. We are heavy users of [Zarr](https://zarr.readthedocs.io/en/stable/) and would like to specially thank their community for building such a great fundamental block. 
