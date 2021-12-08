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
     <a href="https://github.com/activeloopai/Hub/issues">
    <img alt="GitHub issues" src="https://img.shields.io/github/issues/activeloopai/Hub"> </a>
    <a href="https://codecov.io/gh/activeloopai/Hub/branch/main"><img src="https://codecov.io/gh/activeloopai/Hub/branch/main/graph/badge.svg" alt="codecov" height="18"></a>
  <h3 align="center">
   <a href="https://docs.activeloop.ai/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme"><b>Documentation</b></a> &bull;
   <a href="https://docs.activeloop.ai/getting-started/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme"><b>Getting Started</b></a> &bull;
   <a href="https://api-docs.activeloop.ai/"><b>API Reference</b></a> &bull;  
   <a href="https://github.com/activeloopai/examples/"><b>Examples</b></a> &bull; 
   <a href="https://www.activeloop.ai/resources/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme"><b>Blog</b></a> &bull;  
  <a href="http://slack.activeloop.ai"><b>Slack Community</b></a> &bull;
  <a href="https://twitter.com/intent/tweet?text=The%20fastest%20way%20to%20access%20and%20manage%20PyTorch%20and%20Tensorflow%20datasets%20is%20open-source&url=https://activeloop.ai/&via=activeloopai&hashtags=opensource,pytorch,tensorflow,data,datascience,datapipelines,activeloop,dockerhubfordatasets"><b>Twitter</b></a>
 </h3>

## About Hub

Hub is a dataset format with a simple API for creating, storing, and collaborating on AI datasets of any size. The hub data layout enables rapid tranformations and streaming of data while training models at scale. Hub is used by Google, Waymo, Red Cross, Oxford University, and Omdena.


Hub includes the following features:

* **Storage agnostic API**: Use the same API to upload, download, and stream datasets to/from AWS S3/S3-compatible storage, GCP, Activeloop cloud, local storage, as well as in-memory.
* **Compressed storage**: Store images and audios in their native compression, decompressing them only when needed, for e.g., when training a model.
* **Lazy NumPy-like slicing**: Treat your S3 or GCP datasets as if they are a collection of NumPy arrays in your system's memory. Slice them, index them, or iterate through them. Only the bytes you ask for will be downloaded!
* **Dataset version control**: Commits, branches, checkout - Concepts you are already familiar with in your code repositories can now be applied to your datasets as well.
* **Third-party integrations**: Hub comes with built-in integrations for Pytorch and Tensorflow. Train your model with a few lines of code - we even take care of dataset shuffling. :)
* **Distributed transforms**: Rapidly apply transformations on your datasets using multi-threading, multi-processing, or our built-in [Ray](https://www.ray.io/) integration.



## Getting Started with Hub


### 🚀 How to install Hub
Hub is written in 100% Python and can be quickly installed using pip.

```sh
pip3 install hub
```

### 🧠 Training a pytorch model on a hub datasset

#### Load CIFAR-10, one of the readily available datasets in Hub:

```python
import hub
ds = hub.load('hub://activeloop/cifar10-train')
```

#### Inspect tensors in the dataset:

```
print(list(ds.tensors.keys()))

# ['images', 'labels']
```

#### Inspect the shapes of the images and labels tensors:

```python
print(ds.images.shape)

# (50000, 32, 32, 3)
```

```python
print(ds.labels.shape)

# (50000, 1)
```

#### Train a PyTorch model on the Cifar-10 dataset without the need to download it

Define model, loss and optimizer:

```python

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.transpose(x, 1, 3)  # NHWC -> NCHW
        x = x / 255.  # Normalize
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```
Finally, the training loop:

```python
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(ds.pytorch(num_workers=0, batch_size=4, shuffle=True)):
        images, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(images)
        loss = criterion(outputs, labels.reshape(-1))
        loss.backward()
        optimizer.step()
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
```


### 🏗️ How to create a Hub Dataset

A hub dataset can be created in various locations (Storage providers). This is how the paths for each of them would look like:

| Storage provider        | Example path                   |
| ----------------------- | ------------------------------ |
| Activeloop cloud        | hub://user_name/dataset_name   |
| AWS S3 / S3 compatible  | s3://bucket_name/dataset_name  |
| GCP                     | gcp://bucket_name/dataset_name |
| Local storage           | path to local directory        |
| In-memory               | mem://dataset_name             |



Let's create a dataset in the Activeloop cloud. Activeloop cloud provides free storage up to 300 GB per user (more info [here](#-for-students-and-educators)). Create a new account with Hub from the terminal using `activeloop register` if you haven't already. You will be asked for a user name, email ID, and password. The user name you enter here will be used in the dataset path.

```sh
$ activeloop register
Enter your details. Your password must be at least 6 characters long.
Username:
Email:
Password:
```

Initialize an empty dataset in the Activeloop Cloud:

```python
import hub

ds = hub.empty("hub://<USERNAME>/test-dataset")
```


Next, create a tensor to hold images in the dataset we just initialized:

```python
images = ds.create_tensor("images", htype="image", sample_compression="jpg")
```

Assuming you have a list of image file paths, let's upload them to the dataset:

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
        random_image = np.random.randint(0, 256, (100, 100, 3))  # 100x100 image with 3 channels
        ds.images.append(random_image)
```



### 🚀 How to load a Hub Dataset


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



## 📚 Documentation
Getting started guides, examples, tutorials, API reference, and other useful information can be found on our [documentation page](http://docs.activeloop.ai/?utm_source=github&utm_medium=repo&utm_campaign=readme). 

## 🎓 For Students and Educators
Hub users can access and visualize a variety of popular datasets through a free integration with Activeloop's Platform. Users can also create and store their own datasets and make them available to the public. Free storage of up to 300 GB is available for students and educators:

| <!-- -->    | <!-- -->    |
| ---------------------------------------------------- | ------------- |
| Storage for public datasets hosted by Activeloop     | 200GB Free    |
| Storage for private datasets hosted by Activeloop    | 100GB Free    |



## 👩‍💻 Comparisons to Familiar Tools


<details>
  <summary><b>Hub vs DVC</b></summary>
  
Hub and DVC offer dataset version control similar to git for data, but their methods for storing data differ significantly. Hub converts and stores data as chunked compressed arrays, which enables rapid streaming to ML models, whereas DVC operates on top of data stored in less efficient traditional file structures. The Hub format makes dataset versioning significantly easier compared to traditional file structures by DVC when datasets are composed of many files (i.e., many images). An additional distinction is that DVC primarily uses a command-line interface, whereas Hub is a Python package. Lastly, Hub offers an API to easily connect datasets to ML frameworks and other common ML tools and enables instant dataset visualization through [Activeloop's visualization tool](http://app.activeloop.ai/?utm_source=github&utm_medium=repo&utm_campaign=readme).

</details>


<details>
  <summary><b> Activeloop Hub vs TensorFlow Datasets (TFDS)</b></summary>
  
Hub and TFDS seamlessly connect popular datasets to ML frameworks. Hub datasets are compatible with both PyTorch and TensorFlow, whereas TFDS are only compatible with TensorFlow. A key difference between Hub and TFDS is that Hub datasets are designed for streaming from the cloud, whereas TFDS must be downloaded locally prior to use. As a result, with Hub, one can import datasets directly from TensorFlow Datasets and stream them either to PyTorch or TensorFlow. In addition to providing access to popular publicly available datasets, Hub also offers powerful tools for creating custom datasets, storing them on a variety of cloud storage providers, and collaborating with others via simple API. TFDS is primarily focused on giving the public easy access to commonly available datasets, and management of custom datasets is not the primary focus. A full comparison article can be found [here](https://www.activeloop.ai/resources/7jWZXOEJwDoNJS25uiforF/tensorflow-tf.data-&-hub:-how-to-implement-your-tensorflow-data-pipelines-with-hub-/?utm_source=github&utm_medium=repo&utm_campaign=readme).

</details>



<details>
  <summary><b>Activeloop Hub vs HuggingFace</b></summary>

Hub and HuggingFace offer access to popular datasets, but Hub primarily focuses on computer vision, whereas HuggingFace focuses on natural language processing. HuggingFace Transforms and other computational tools for NLP are not analogous to features offered by Hub.


</details>


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

<details>
  <summary><b> Dataset Licenses</b></summary>

Hub users may have access to a variety of publicly available datasets. We do not host or distribute these datasets, vouch for their quality or fairness, or claim that you have a license to use the datasets. It is your responsibility to determine whether you have permission to use the datasets under their license.

If you're a dataset owner and do not want your dataset to be included in this library, please get in touch through a [GitHub issue](https://github.com/activeloopai/Hub/issues/new). Thank you for your contribution to the ML community!

</details>

<details>
  <summary><b> Usage Tracking</b></summary>

By default, we collect usage data using Bugout (here's the [code](https://github.com/activeloopai/Hub/blob/853456a314b4fb5623c936c825601097b0685119/hub/__init__.py#L24) that does it). It does not collect user data other than anonymized IP address data, and it only logs the Hub library's own actions. This helps our team understand how the tool is used and how to build features that matter to you! After you register with Activeloop, data is no longer anonymous. You can always opt-out of reporting using the CLI command below:

```
activeloop reporting --off
```
</details>

## Acknowledgment
This technology was inspired by our research work at Princeton University. We would like to thank William Silversmith @SeungLab for his awesome [cloud-volume](https://github.com/seung-lab/cloud-volume) tool.
