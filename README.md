<p align="center">
 <a href="https://github.com/activeloopai/Hub/wiki/Hub-Kaggle's-Herbarium-2021--Half-Earth-Challenge">
  <img src="https://raw.githubusercontent.com/activeloopai/Hub/master/docs/Use_Hub_in_a_CVPR_challenge_and_win_up_to_$1000_(read_more).png" height="35" /></a>
</p>
<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/activeloopai/Hub/master/docs/logo/logo-explainer-bg.png" width="50%"/>
    </br>
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
    <a href="https://codecov.io/gh/activeloopai/Hub/branch/master"><img src="https://codecov.io/gh/activeloopai/Hub/branch/master/graph/badge.svg" alt="codecov" height="18"></a>
    <a href="https://twitter.com/intent/tweet?text=The%20fastest%20way%20to%20access%20and%20manage%20PyTorch%20and%20Tensorflow%20datasets%20is%20open-source&url=https://activeloop.ai/&via=activeloopai&hashtags=opensource,pytorch,tensorflow,data,datascience,datapipelines,activeloop,dockerhubfordatasets"> 
        <img alt="tweet" src="https://img.shields.io/twitter/url/http/shields.io.svg?style=social"> </a>  
   </br> 
    <a href="https://join.slack.com/t/hubdb/shared_invite/zt-ivhsj8sz-GWv9c5FLBDVw8vn~sxRKqQ">
  <img src="https://user-images.githubusercontent.com/13848158/97266254-9532b000-1841-11eb-8b06-ed73e99c2e5f.png" height="35" /> </a>
    <a href="https://docs.activeloop.ai/en/latest/?utm_source=github&utm_medium=readme&utm_campaign=button">
  <img src="https://i.ibb.co/YBTCcJc/output-onlinepngtools.png" height="35" /></a>

---

</a>
</p>

<h3 align="center"> Introducing Data 2.0, powered by Hub. </br>The fastest way to store, access & manage datasets with version-control for PyTorch/TensorFlow.  Works locally or on any cloud. Scalable data pipelines.</h3>

---

[ English | [Français](./README_translations/README_FR.md) | [简体中文](./README_translations/README_CN.md) | [Türkçe](./README_translations/README_TR.md) | [한글](./README_translations/README_KR.md) | [Bahasa Indonesia](./README_translations/README_ID.md)] | [Русский](./README_translations/README_RU.md)]

<i> Note: the translations of this document may not be up-to-date. For the latest version, please check the README in English. </i>

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

 <p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/activeloopai/Hub/master/docs/visualizer%20gif.gif" width="75%"/>
    </br>
Visualization of a dataset uploaded to Hub via app.activeloop.ai (free tool).

</p>


## Getting Started
Work with public or your own data, locally or on any cloud.

### Access public data. Fast.

To load a public dataset, one needs to write dozens of lines of code and spend hours accessing and understanding the API as well as downloading the data. With Hub, you only need 2 lines of code, and you **can get started working on your dataset in under 3 minutes**.

```sh
pip3 install hub
```

Access public datasets in Hub by following a straight-forward convention which merely requires a few lines of simple code. Run this excerpt to get the first thousand images in the [MNIST database](https://app.activeloop.ai/dataset/activeloop/mnist/?utm_source=github&utm_medium=repo&utm_campaign=readme) in the numpy array format:
```python
from hub_v1 import Dataset

mnist = Dataset("activeloop/mnist")  # loading the MNIST data lazily
# saving time with *compute* to retrieve just the necessary data
mnist["image"][0:1000].compute()
```
You can find all the other popular datasets on [app.activeloop.ai](https://app.activeloop.ai/datasets/popular/?utm_source=github&utm_medium=repo&utm_campaign=readme).

### Train a model

Load the data and train your model **directly**. Hub is integrated with PyTorch and TensorFlow and performs conversions between formats in an understandable fashion. Take a look at the example with PyTorch below:

```python
from hub_v1 import Dataset
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
from hub_v1 import Dataset, schema
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
ds.flush()  # executing the creation of the dataset
```

You can also specify `s3://bucket/path`, `gcs://bucket/path` or azure path. [Here](https://docs.activeloop.ai/en/latest/simple.html#data-storage) you can find more information on cloud storage.
Also, if you need a publicly available dataset that you cannot find in the Hub, you may [file a request](https://github.com/activeloopai/Hub/issues/new?assignees=&labels=i%3A+enhancement%2C+i%3A+needs+triage&template=feature_request.md&title=[FEATURE]+New+Dataset+Required%3A+%2Adataset_name%2A). We will enable it for everyone as soon as we can!

### Upload your dataset and access it from <ins>anywhere</ins> in 3 simple steps

1. Register a free account at [Activeloop](https://app.activeloop.ai/register/?utm_source=github&utm_medium=repo&utm_campaign=readme) and authenticate locally:

    ```sh
    activeloop register
    activeloop login

    # alternatively, add username and password as arguments (use on platforms like Kaggle)
    activeloop login -u username -p password
    ```

2. Then create a dataset, specifying its name and upload it to your account. For instance:
    ```python
    from hub_v1 import Dataset, schema
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
    ds.flush()
    ```

3. Access it from anywhere else in the world, on any device having a command line:
    ```python
    from hub_v1 import Dataset

    ds = Dataset("username/dataset_name")
    ```


## Documentation

For more advanced data pipelines like uploading large datasets or applying many transformations, please refer to our [documentation](http://docs.activeloop.ai/?utm_source=github&utm_medium=repo&utm_campaign=readme).

## Tutorial Notebooks
The [examples](https://github.com/activeloopai/Hub/tree/master/examples) directory has a series of examples and the [notebooks](https://github.com/activeloopai/Hub/tree/master/examples/notebooks) has some notebooks with use cases. Some of the notebooks are listed of below.

| Notebook  	|   Description	|   	|
|:---	|:---	|---:	|
| [Uploading Images](https://github.com/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%201a%20-%20Uploading%20Images.ipynb) | Overview on how to upload and store images on Hub |  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%201a%20-%20Uploading%20Images.ipynb) |
| [Uploading Dataframes](https://github.com/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%201b%20-%20Uploading%20Dataframes.ipynb)  	| Overview on how to upload Dataframes on Hub  	| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%201b%20-%20Uploading%20Dataframes.ipynb)  	|
| [Uploading Audio](https://github.com/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%201c%20-%20Uploading%20Audio.ipynb) | Explains how to handle audio data in Hub|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%201c%20-%20Uploading%20Audio.ipynb) |
| [Retrieving Remote Data](https://github.com/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%202%20-%20Retrieving%20Remote%20Data.ipynb) | Explains how to retrieve Data| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/activeloopai/Hub/blob/master/examples/tutorial/tutorial/Tutorial%202%20-%20Retrieving%20Remote%20Data.ipynb) |
| [Transforming Data](https://github.com/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%203%20-%20Transforming%20Data.ipynb) | Briefs on how data transformation with Hub|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%203%20-%20Transforming%20Data.ipynb) |
| [Dynamic Tensors](https://github.com/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%204%20-%20What%20are%20Dynamic%20Tensors.ipynb) | Handling data with variable shape and sizes|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/activeloopai/Hub/blob/master/examples/tutorial/Tutorial%204%20-%20What%20are%20Dynamic%20Tensors.ipynb) |
| [NLP using Hub](https://github.com/activeloopai/Hub/blob/master/examples/notebooks/nlp_using_hub.ipynb) | Fine Tuning Bert for CoLA|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/activeloopai/Hub/blob/master/examples/notebooks/nlp_using_hub.ipynb) |
| [Getting Started with Text on Hub](https://github.com/activeloopai/Hub/blob/master/examples/notebooks/Getting_Started_with_Text_on_Hub.ipynb) | Overview on using Text datasets in Hub | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/activeloopai/Hub/blob/master/examples/notebooks/Getting_Started_with_Text_on_Hub.ipynb) |


## Use Cases
* **Satellite and drone imagery**: [Smarter farming with scalable aerial pipelines](https://activeloop.ai/usecase/intelinair?utm_source=github&utm_medium=repo&utm_campaign=readme), [Mapping Economic Well-being in India](https://towardsdatascience.com/faster-machine-learning-using-hub-by-activeloop-4ffb3420c005), [Fighting desert Locust in Kenya with Red Cross](https://omdena.com/projects/ai-desert-locust/)
* **Medical Images**: Volumetric images such as MRI or Xray
* **Self-Driving Cars**: [Radar, 3D LIDAR, Point Cloud, Semantic Segmentation, Video Objects](https://medium.com/snarkhub/extending-snark-hub-capabilities-to-handle-waymo-open-dataset-4dc7b7d8ab35)
* **Retail**: Self-checkout datasets
* **Media**: Images, Video, Audio storage

## Why Hub specifically?

There are quite a few dataset management libraries which offer functionality that might seem similar to Hub. In fact, quite a few users migrate data from PyTorch or Tensorflow Datasets to Hub. Here are a few startling differences you will encounter after switching to Hub:
* the data is provided in chunks, which you may stream from a remote location, instead of downloading all of it at once
* as only the necessary portion of the dataset is evaluated, you are able to work on the data immediately
* you are able to store the data that would not fit in your memory in its entirety
* you may version control and collaborate with multiple users on your datasets across different machines
* you are equipped with tools that enhance your understanding of the data in a manner of seconds, such as our visualization tool
* you can easily prepare your data for multiple training libraries at ones (e.g. you can use the same dataset for training with PyTorch and Tensorflow)


## Community

Join our [**Slack community**](https://join.slack.com/t/hubdb/shared_invite/zt-ivhsj8sz-GWv9c5FLBDVw8vn~sxRKqQ) to get help from Activeloop team and other users, as well as stay up-to-date on dataset management/preprocessing best practices.

We'd love your feedback by completing our 3-minute [**survey**](https://forms.gle/rLi4w33dow6CSMcm9).

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
## Usage Tracking
By default, we collect anonymous usage data using Bugout (here's the [code](https://github.com/activeloopai/Hub/blob/853456a314b4fb5623c936c825601097b0685119/hub/__init__.py#L24) that does it). It only logs Hub library's own actions and parameters, and no user/ model data is collected.

This helps the Activeloop team to understand how the tool is used and how to deliver maximum value to the community by building features that matter to you. You can easily opt-out of usage tracking during login.


## Disclaimers

Similarly to other dataset management packages, `Hub` is a utility library that downloads and prepares public datasets. We do not host or distribute these datasets, vouch for their quality or fairness, or claim that you have license to use the dataset. It is your responsibility to determine whether you have permission to use the dataset under the dataset's license.

If you're a dataset owner and wish to update any part of it (description, citation, etc.), or do not want your dataset to be included in this library, please get in touch through a [GitHub issue](https://github.com/activeloopai/Hub/issues/new). Thanks for your contribution to the ML community!


## Acknowledgement
 This technology was inspired from our experience at Princeton University and would like to thank William Silversmith @SeungLab with his awesome [cloud-volume](https://github.com/seung-lab/cloud-volume) tool. We are heavy users of [Zarr](https://zarr.readthedocs.io/en/stable/) and would like to thank their community for building such a great fundamental block. 
