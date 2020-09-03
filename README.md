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
    <a href="https://twitter.com/intent/tweet?text=The%20fastest%20way%20to%20access%20and%20manage%20PyTorch%20and%20Tensorflow%20datasets%20is%20open-source&url=https://activeloop.ai/&via=activeloopai&hashtags=opensource,pytorch,tensorflow,data,datascience,datapipelines,sqlforimages,activeloop"> 
        <img alt="tweet" src="https://img.shields.io/twitter/url/http/shields.io.svg?style=social">
    </a>
  </a>
</p>
<h3 align="center">
The fastest way to access and manage datasets for PyTorch and TensorFlow
</h3>

Hub provides fast access to the state-of-the-art datasets for Deep Learning, enabling data scientists to manage them, build scalable data pipelines and connect to Pytorch and Tensorflow 

### Problems with Current Workflows

We realized that there are a few problems related with current workflow in deep learning data management through our experience of working with deep learning companies and researchers. Most of the time Data Scientists/ML researchers work on data management and preprocessing instead of doing modeling. Deep Learning often requires to work with large datasets. Those datasets can grow up to terabyte or even petabyte size.  It is hard to manage data, store, access, and version-control. It is time-consuming to download the data and link with the training or inference code. There is no easy way to access a chunk of it and possibly visualize. Wouldn’t it be more convenient to have large datasets stored & version-controlled as single numpy-like array on the cloud and have access to it from any machine at scale?

## Getting Started

### Access public data. Fast

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
from hub import tensor, dataset

images = tensor.from_array(np.zeros((4, 512, 512)))
labels = tensor.from_array(np.zeros((4, 512, 512)))

ds = dataset.from_tensors({"images": images, "labels": labels})
ds.store("username/basic")
```

3. Access it from anywhere else in the world, on any device having a command line.
```python
import hub

ds = hub.load("username/basic")
```
For more advanced data pipelines like uploading large datasets or applying many transformations, please see [docs](http://docs.activeloop.ai).

## Things you can do with Hub
* Store large datasets with version-control
* Collaborate as in Google Docs: Multiple data scientists working on the same data in sync with no interruptions
* Access from multiple machines simultaneously
* Integration with your ML tools like Numpy, Dask, PyTorch, or TensorFlow.
* Create arrays as big as you want
* Take a quick look on your data without redundant manipulations/in a matter of seconds/etc.

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



# Disclaimers

Similarly to other dataset management packages, `Hub` is a utility library that downloads and prepares public datasets. We do not host or distribute these datasets, vouch for their quality or fairness, or claim that you have license to use the dataset. It is your responsibility to determine whether you have permission to use the dataset under the dataset's license.

If you're a dataset owner and wish to update any part of it (description, citation, etc.), or do not want your dataset to be included in this library, please get in touch through a [GitHub issue](https://github.com/activeloopai/Hub/issues/new). Thanks for your contribution to the ML community!

