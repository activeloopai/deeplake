| :zap:        Hacktoberfest 2021 is here! Contribute and win Activeloop swag. Grab an issue in our  <a href="https://github.com/activeloopai/Hub/projects/11"><b>Hacktoberfest dashboard</b></a>! :zap: |
|-----------------------------------------|


<img src="https://static.scarf.sh/a.png?x-pxid=bc3c57b0-9a65-49fe-b8ea-f711c4d35b82" /><p align="center">
    <img src="https://www.linkpicture.com/q/hub_logo-1.png" width="35%"/>
    </br>
    <h1 align="center">Dataset Format for AI
 </h1>
<p align="center">
    <a href="http://docs.activeloop.ai/">
        <img alt="Docs" src="https://readthedocs.org/projects/hubdb/badge/?version=latest">
    </a>
    <a href="https://pypi.org/project/hub/"><img src="https://badge.fury.io/py/hub.svg" alt="PyPI version" height="18"></a>
    <a href="https://pepy.tech/project/hub"><img src="https://static.pepy.tech/personalized-badge/hub?period=month&units=international_system&left_color=grey&right_color=orange&left_text=Downloads" alt="PyPI version" height="18"></a>
    <a href="https://app.circleci.com/pipelines/github/activeloopai/Hub">
    <img alt="CircleCI" src="https://img.shields.io/circleci/build/github/activeloopai/Hub?logo=circleci"> </a>
     <a href="https://github.com/activeloopai/Hub/issues">
    <img alt="GitHub issues" src="https://img.shields.io/github/issues/activeloopai/Hub"> </a>
    <a href="https://codecov.io/gh/activeloopai/Hub/branch/main"><img src="https://codecov.io/gh/activeloopai/Hub/branch/main/graph/badge.svg" alt="codecov" height="18"></a>
  <h3 align="center">
   <a href="https://activeloop.gitbook.io/hub-2-0/"><b>Documentation</b></a> &bull;
   <a href="https://activeloop.gitbook.io/hub-2-0/getting-started/"><b>Getting Started</b></a> &bull;
   <a href="https://api-docs.activeloop.ai/"><b>API Reference</b></a> &bull;  
  <a href="http://slack.activeloop.ai"><b>Slack Community</b></a> &bull;
  <a href="https://twitter.com/intent/tweet?text=The%20fastest%20way%20to%20access%20and%20manage%20PyTorch%20and%20Tensorflow%20datasets%20is%20open-source&url=https://activeloop.ai/&via=activeloopai&hashtags=opensource,pytorch,tensorflow,data,datascience,datapipelines,activeloop,dockerhubfordatasets"><b>Twitter</b></a>
 </h3>

## About Hub

Hub is a dataset format with a simple API for creating, storing, and collaborating on AI datasets of any size. The hub data layout enables rapidly transform and stream data while training models at scale. Hub is used Google, Waymo, Red Cross, Omdena, and Rarebase.


Hub includes the following features:

* **Storage agnostic API**: Use the same API to upload, download, and stream datasets to/from AWS S3/S3-compatible storage, GCP, Activeloop cloud, local storage as well as in-memory.
* **Compressed storage**: Store images and audios in their native compression (full list [here](https://docs.activeloop.ai/getting-started/understanding-compression)), decompressing them only when needed, for e.g, when training a model.
* **Lazy NumPy-like slicing**: Treat your S3 or GCP datasets as if they are a collection of NumPy arrays in your system's memory. Slice them, index them, or iterate through them. Only the bytes you ask for will be downloaded!
* **Dataset version control**: Commits, branches, checkout - Concepts you are already familiar with in your code repositories can now be applied to your datasets as well.
* **Third-party integrations**: Hub comes with built-in integrations for Pytorch and Tensorflow. Train your model with a few lines of code - we even take care of dataset shuffling. :)
* **Distributed transforms**: Rapidly apply transformations on your datasets using multi-threading, multi-processing, or our built-in [Ray](https://www.ray.io/) integration.




## Getting Started with Hub


### How to install Hub
Hub is written in 100% python and can be quickly installed using pip.

```sh
pip3 install hub
```


### How to create a Hub Dataset

A hub dataset can be created in various locations (Storage providers). This is how the paths for each of them would look like:

| Storage provider        | Example path                   |
| ----------------------- | ------------------------------ |
| Activeloop cloud        | hub://user_name/dataset_name   |
| AWS S3 / S3 compatible  | s3://bucket_name/dataset_name  |
| GCP                     | gcp://bucket_name/dataset_name |
| Local storage           | path to local directory        |
| In-memory               | mem://dataset_name             |



Let's create a dataset in the Activeloop cloud. Activeloop cloud provides free storage upto 300 GB per user (more info [here](#-for-students-and-educators)). Create a new account with Hub from the terminal using `activeloop register` if you haven't already. You will be asked for a user name, email id and passowrd. The user name you enter here will be used in the dataset path.

```sh
$ activeloop register
Enter your details. Your password must be atleast 6 characters long.
Username:
Email:
Password:
```

Initialize an empty dataset in the hub cloud:

```python
import hub

ds = hub.empty("hub://<USERNAME>/test-dataset")
```


Next, create a tensor to hold images in the dataset we just initialized:

```python
images = ds.create_tensor("images", htype="image", sample_compression="jpg")
```

Assuming you have a list of image file paths, lets upload them to the dataset:

```python
image_paths = ...
with ds:
    for image_path in image_paths:
        image = hub.read(image_path)
        ds.images.append(image)
```

Alternatively, you can also upload numpy arrays. Since the `images` tensor was created with `sample_compression="jpg"`, the arrays will be compressed with jpeg compression.


```python
import numpy as np

with ds:
    for _ in range(1000):  # 1000 random images
        radnom_image = np.random.randint(0, 256, (100, 100, 3))  # 100x100 image with 3 channels
        ds.images.append(image)
```



### How to load a Hub Dataset


You can load the dataset you just created with a single line of code:

```python
import hub

ds = hub.load("hub://<USERNAME>/test-dataset")
```

You can also access other publicly available hub datasets, not just the ones you created. Here is how you would load the [Objectron Bikes Dataset](https://github.com/google-research-datasets/Objectron):

```python
import hub

ds = hub.load('hub://activeloop/objectron_bike_train')
```

To get the first image in the Objectron Bikes dataset in numpy format:


```python
image_arr = ds.image[0].numpy()
```



## Documentation
Getting started guides, examples, tutorials, API reference, and other usage information can be found on our [documentation page](http://docs.activeloop.ai/?utm_source=github&utm_medium=repo&utm_campaign=readme). 

## 🎓 For Students and Educators
Hub users can access and visualize a variety of popular datasets through a free integration with Activeloop's Platform. Users can also create and store their own datasets and make them available to the public. Free storage of up to 200 GB is available.


## Comparisons to Familiar Tools
### Hub and DVC
Hub and DVC offer dataset version control similar to git for data, but their methods for storing data differ significantly. Hub converts and stores data as chunked compressed arrays, which enables rapid streaming to ML models, whereas DVC operates on top of data stored in less efficient traditional file structures. The Hub format makes dataset versioning significantly easier compared to a traditional file structures by DVC when datasets are composed of many files (i.e. many images). An additional distinction is that DVC primarily uses a command line interface, where as Hub is a python package. Lastly, Hub offers an API to easily connect datasets to ML frameworks and other common ML tools.

### Hub and TensorFlow Datasets (TFDS)
Hub and TFDS seamlessly connect popular datasets to ML frameworks. Hub datasets are compatible with both PyTorch and TensorFlow, whereas TFDS are only compatible with TensorFlow. A key difference between Hub and TFDS is that Hub datasets are designed for streaming from the cloud, whereas TFDS must be downloaded locally prior to use. In addition to providing access to popular publicly-available datasets, Hub also offers powerful tools for creating custom datasets, storing them on a variety of cloud storage providers, and collaborating with others. TFDS is primarily focused on giving the public easy access to commonly available datasets, and management of custom datasets is not the primary focus.

### Hub and HuggingFace 
Hub and HuggingFace offer access to popular datasets, but Hub primarily focuses on computer vision, whereas HuggingFace primarily focuses on natural language processing. HuggingFace Transforms and other computational tools for NLP are not analogous to features offered by Hub.

## Community

Join our [**Slack community**](https://join.slack.com/t/hubdb/shared_invite/zt-ivhsj8sz-GWv9c5FLBDVw8vn~sxRKqQ) to learn more about unstructured dataset management using Hub and to get help from the Activeloop team and other users.

We'd love your feedback by completing our 3-minute [**survey**](https://forms.gle/rLi4w33dow6CSMcm9).

As always, thanks to our amazing contributors!    

<a href="https://github.com/activeloopai/hub/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=activeloopai/hub" />
</a>

Made with [contributors-img](https://contrib.rocks).

Please read [CONTRIBUTING.md](CONTRIBUTING.md) to get started with making contributions to Hub.


## README Badge

Using Hub? Add a README badge to let everyone know: 


[![hub](https://img.shields.io/badge/powered%20by-hub%20-ff5a1f.svg)](https://github.com/activeloopai/Hub)

```
[![hub](https://img.shields.io/badge/powered%20by-hub%20-ff5a1f.svg)](https://github.com/activeloopai/Hub)
```



## Disclaimers

### Dataset Licenses
Hub users may have access to a variety of publicly available datasets. We do not host or distribute these datasets, vouch for their quality or fairness, or claim that you have license to use the datasets. It is your responsibility to determine whether you have permission to use the datasets under their license.

If you're a dataset owner and do not want your dataset to be included in this library, please get in touch through a [GitHub issue](https://github.com/activeloopai/Hub/issues/new). Thank you for your contribution to the ML community!

### Usage Tracking
By default, we collect anonymous usage data using Bugout (here's the [code](https://github.com/activeloopai/Hub/blob/853456a314b4fb5623c936c825601097b0685119/hub/__init__.py#L24) that does it). It does not collect user data and it only logs the Hub library's own actions. This helps our team understand how the tool is used and how to build features that matter to you! After you register with Activeloop, data is no longer anonymous, but you can  opt-out of reporing using the CLI command below:

```
activeloop reporting --off
```

## Acknowledgment
This technology was inspired by our research work at Princeton University. We would like to thank William Silversmith @SeungLab for his awesome [cloud-volume](https://github.com/seung-lab/cloud-volume) tool.
