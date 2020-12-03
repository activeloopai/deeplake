 <p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/activeloopai/Hub/master/docs/logo/hub_logo_explainer.png" width="50%"/>
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

### What is Hub for?

Software 2.0 needs Data 2.0, and Hub delivers it. Most of the time Data Scientists/ML researchers work on data management and preprocessing instead of training models. With Hub, we are fixing this. We store your (even petabyte-scale) datasets as single numpy-like array on the cloud, so you can seamlessly access and work with it from any machine. Hub makes any data type (images, text files, audio, or video) stored in cloud usable as fast as if it were stored on premise. With same dataset view, your team can always be in sync. 

Hub is being used by Waymo, Red Cross, World Resources Institute, Omdena, and others.

### Features 

* Store and retrieve large datasets with version-control
* Collaborate as in Google Docs: Multiple data scientists working on the same data in sync with no interruptions
* Access from multiple machines simultaneously
* Integrate with your ML tools like Numpy, Dask, Ray, [PyTorch](https://docs.activeloop.ai/en/latest/integrations/pytorch.html), or [TensorFlow](https://docs.activeloop.ai/en/latest/integrations/tensorflow.html)
* Deploy on Google Cloud, S3, Azure as well as Activeloop (by default - and for free!) 
* Create arrays as big as you want. You can store images as big as 100k by 100k!
* Keep shape of each sample dynamic. This way you can store small and big arrays as 1 array. 
* [Visualize](http://app.activeloop.ai/?utm_source=github&utm_medium=repo&utm_campaign=readme) any slice of the data in a matter of seconds without redundant manipulations

## Getting Started

### Access public data. Fast

To load a public dataset, one needs to write dozens of lines of code and spend hours accessing and understanding the API, as well as downloading the data. With Hub, you only need 2 lines of code, and you **can get started working on your dataset in under 3 minutes**. 

```sh
pip3 install hub
```

You can access public datasets with a few lines of code.
```python
import hub

mnist = hub.load("activeloop/mnist")
mnist["image"][0:1000].compute()
```

### Train a model

Load the data and directly train your model using pytorch

```python
import hub
import torch

mnist = hub.load("activeloop/mnist")
mnist = mnist.to_pytorch(lambda x: (x["image"], x["label"]))

train_loader = torch.utils.data.DataLoader(mnist, batch_size=1, num_workers=0)

for image, label in train_loader:
    # Training loop here
```

### Upload your dataset and access it from <ins>anywhere</ins> in 3 simple steps

1. Register a free account at [Activeloop](http://app.activeloop.ai/?utm_source=github&utm_medium=repo&utm_campaign=readme) and authenticate locally
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
Instead of `username/basic` you could also use `./local/path/`, `s3://path` or `gcs://`

## Documentation

For more advanced data pipelines like uploading large datasets or applying many transformations, please read the [docs](http://docs.activeloop.ai/?utm_source=github&utm_medium=repo&utm_campaign=readme).

## Use Cases
* **Satellite and drone imagery**: [Smarter farming with scalable aerial pipelines](https://activeloop.ai/usecase/intelinair?utm_source=github&utm_medium=repo&utm_campaign=readme), [Mapping Economic Well-being in India](https://towardsdatascience.com/faster-machine-learning-using-hub-by-activeloop-4ffb3420c005), [Fighting desert Locust in Kenya with Red Cross](https://omdena.com/projects/ai-desert-locust/)
* **Medical Images**: Volumetric images such as MRI or Xray
* **Self-Driving Cars**: [Radar, 3D LIDAR, Point Cloud, Semantic Segmentation, Video Objects](https://medium.com/snarkhub/extending-snark-hub-capabilities-to-handle-waymo-open-dataset-4dc7b7d8ab35)
* **Retail**: Self-checkout datasets
* **Media**: Images, Video, Audio storage

## Community

Join our [Slack community](https://join.slack.com/t/hubdb/shared_invite/zt-ivhsj8sz-GWv9c5FLBDVw8vn~sxRKqQ) to get help from Activeloop team and other users, as well as stay up-to-date on dataset management/preprocessing best practices.

<img alt="tweet" src="https://img.shields.io/twitter/follow/activeloopai?label=stay%20in%20the%20Loop&style=social"> on Twitter.

As always, thanks to our amazing contributors!     </br>

[![](https://sourcerer.io/fame/davidbuniat/activeloopai/Hub/images/0)](https://sourcerer.io/fame/davidbuniat/activeloopai/Hub/links/0)[![](https://sourcerer.io/fame/davidbuniat/activeloopai/Hub/images/1)](https://sourcerer.io/fame/davidbuniat/activeloopai/Hub/links/1)[![](https://sourcerer.io/fame/davidbuniat/activeloopai/Hub/images/2)](https://sourcerer.io/fame/davidbuniat/activeloopai/Hub/links/2)[![](https://sourcerer.io/fame/davidbuniat/activeloopai/Hub/images/3)](https://sourcerer.io/fame/davidbuniat/activeloopai/Hub/links/3)[![](https://sourcerer.io/fame/davidbuniat/activeloopai/Hub/images/4)](https://sourcerer.io/fame/davidbuniat/activeloopai/Hub/links/4)[![](https://sourcerer.io/fame/davidbuniat/activeloopai/Hub/images/5)](https://sourcerer.io/fame/davidbuniat/activeloopai/Hub/links/5)[![](https://sourcerer.io/fame/davidbuniat/activeloopai/Hub/images/6)](https://sourcerer.io/fame/davidbuniat/activeloopai/Hub/links/6)[![](https://sourcerer.io/fame/davidbuniat/activeloopai/Hub/images/7)](https://sourcerer.io/fame/davidbuniat/activeloopai/Hub/links/7)


## Examples
Activeloop’s Hub format lets you achieve faster inference at a lower cost. Test out the datasets we’ve converted into Hub format - see for yourself!
- [Waymo Open Dataset](https://medium.com/snarkhub/extending-snark-hub-capabilities-to-handle-waymo-open-dataset-4dc7b7d8ab35)
- [Aptiv nuScenes](https://medium.com/snarkhub/snark-hub-is-hosting-nuscenes-dataset-for-autonomous-driving-1470ae3e1923)


## Disclaimers

Similarly to other dataset management packages, `Hub` is a utility library that downloads and prepares public datasets. We do not host or distribute these datasets, vouch for their quality or fairness, or claim that you have license to use the dataset. It is your responsibility to determine whether you have permission to use the dataset under the dataset's license.

If you're a dataset owner and wish to update any part of it (description, citation, etc.), or do not want your dataset to be included in this library, please get in touch through a [GitHub issue](https://github.com/activeloopai/Hub/issues/new). Thanks for your contribution to the ML community!


## Acknowledgement
 This technology was inspired from our experience at Princeton University and would like to thank William Silversmith @SeungLab with his awesome [cloud-volume](https://github.com/seung-lab/cloud-volume) tool. We are heavy users of [Zarr](https://zarr.readthedocs.io/en/stable/) and would like to specially thank their community for building such a great fundamental block. 
