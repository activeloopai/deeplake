<img src="https://static.scarf.sh/a.png?x-pxid=bc3c57b0-9a65-49fe-b8ea-f711c4d35b82" /><p align="center">
     <img src="https://user-images.githubusercontent.com/83741606/156426873-c0a77da0-9e0f-41a0-a4fb-cf77eb2fe35e.png" width="300"/>
</h1>
    </br>
    <h1 align="center">Dataset Format for AI
 </h1>
<p align="center">
    <a href="https://github.com/activeloopai/Hub/actions/workflows/test-pr-on-label.yml"><img src="https://github.com/activeloopai/Hub/actions/workflows/test-push.yml/badge.svg" alt="PyPI version" height="18"></a>
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
  <a href="https://twitter.com/intent/tweet?text=The%20dataset%20format%20for%20AI.%20Stream%20data%20to%20PyTorch%20and%20Tensorflow%20datasets&url=https://activeloop.ai/&via=activeloopai&hashtags=opensource,pytorch,tensorflow,data,datascience,datapipelines,activeloop,databaseforAI"><b>Twitter</b></a>
 </h3>
 

*Read this in other languages: [简体中文](README.zh-cn.md)*

## About Hub

Hub is a dataset format with a simple API for creating, storing, and collaborating on AI datasets of any size. It enables you to store all of your data in one place, ranging from simple annotations to large videos, and it unlocks rapid streaming of data while training models at scale. Hub is used by Google, Waymo, Red Cross, Oxford University, and Omdena. Hub includes the following features:

<details>
  <summary><b>Storage Agnostic API</b></summary>
Use the same API to upload, download, and stream datasets to/from AWS S3/S3-compatible storage, GCP, Activeloop cloud, local storage, as well as in-memory.
</details>
<details>
  <summary><b>Compressed Storage</b></summary>
Store images, audios and videos in their native compression, decompressing them only when needed, for e.g., when training a model.
</details>
<details>
  <summary><b>Lazy NumPy-like Indexing</b></summary>
Treat your S3 or GCP datasets as if they are a collection of NumPy arrays in your system's memory. Slice them, index them, or iterate through them. Only the bytes you ask for will be downloaded!
</details>
<details>
  <summary><b>Dataset Version Control</b></summary>
Commits, branches, checkout - Concepts you are already familiar with in your code repositories can now be applied to your datasets as well!
</details>
<details>
  <summary><b>Integrations with Deep Learning Frameworks</b></summary>
Hub comes with built-in integrations for Pytorch and Tensorflow. Train your model with a few lines of code - we even take care of dataset shuffling. :)
</details>
<details>
  <summary><b>Distributed Transformations</b></summary>
Rapidly apply transformations on your datasets using multi-threading, multi-processing, or our built-in <a href="https://www.ray.io/">Ray</a> integration.</details>
<details>
  <summary><b>100+ most-popular image, video, and audio datasets available in seconds</b></summary>
Hub community has uploaded <a href="https://docs.activeloop.ai/datasets/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme">100+ image, video and audio datasets</a> like <a href="https://docs.activeloop.ai/datasets/mnist/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme">MNIST</a>, <a href="https://docs.activeloop.ai/datasets/coco-dataset/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme">COCO</a>,  <a href="https://docs.activeloop.ai/datasets/imagenet-dataset/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme">ImageNet</a>,  <a href="https://docs.activeloop.ai/datasets/cifar-10-dataset/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme">CIFAR</a>,  <a href="https://docs.activeloop.ai/datasets/gtzan-genre-dataset/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme">GTZAN</a> and others.
</details>
</details>
<details>
  <summary><b>Instant Visualization Support in <a href="https://app.activeloop.ai/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme">Activeloop Platform</a></b></summary>
Hub datasets are instantly visualized with bounding boxes, masks, annotations, etc. in <a href="https://app.activeloop.ai/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme">Activeloop Platform</a> (see below).
</details>


<div align="center">
<a href="https://www.linkpicture.com/view.php?img=LPic61b13e5c1c539681810493"><img src="https://www.linkpicture.com/q/ReadMe.gif" type="image"></a>
</div>

    
## Getting Started with Hub


### 🚀 How to install Hub
Hub is written in 100% Python and can be quickly installed using pip.

```sh
pip3 install hub
```

**By default, Hub does not install dependencies for audio, video, and google-cloud (GCS) support. They can be installed using**:
```sh
pip3 install hub[av]          -> Audio and video support via PyAV
pip3 install hub[gcp]         -> GCS support via google-* dependencies
pip3 install hub[visualizer]  -> Visualizer support in Jupyter Notebooks
pip3 install hub[all]         -> Installs everything - audio, video and GCS support
```

### 🧠 Training a PyTorch model on a Hub dataset

#### Load CIFAR 10, one of the readily available datasets in Hub:

```python
import hub
import torch
from torchvision import transforms, models

ds = hub.load('hub://activeloop/cifar10-train')
```

#### Inspect tensors in the dataset:

```python
ds.tensors.keys()    # dict_keys(['images', 'labels'])
ds.labels[0].numpy() # array([6], dtype=uint32)
```

#### Train a PyTorch model on the CIFAR 10 dataset without the need to download it

First, define a transform for the images and use Hub's built-in PyTorch one-line dataloader to connect the data to the compute:

```python
tform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

hub_loader = ds.pytorch(num_workers=0, batch_size=4, transform={
                        'images': tform, 'labels': None}, shuffle=True)
```

Next, define the model, loss and optimizer:

```python
net = models.resnet18(pretrained=False)
net.fc = torch.nn.Linear(net.fc.in_features, len(ds.labels.info.class_names))
    
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
```

Finally, the training loop for 2 epochs:

```python
for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(hub_loader):
        images, labels = data['images'], data['labels']
        
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(images)
        loss = criterion(outputs, labels.reshape(-1))
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
```


### 🏗️ How to create a Hub Dataset

A hub dataset can be created in various locations (Storage providers). This is how the paths for each of them would look like:

| Storage provider        | Example path                   |
| ----------------------- | ------------------------------ |
| Activeloop cloud        | hub://user_name/dataset_name   |
| AWS S3 / S3 compatible  | s3://bucket_name/dataset_name  |
| GCP                     | gcp://bucket_name/dataset_name |
| Google Drive            | gdrive://path_to_dataset
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

You can also access one of the <a href="https://docs.activeloop.ai/datasets/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme">100+ image, video and audio datasets in Hub format</a>, not just the ones you created. Here is how you would load the [Objectron Bikes Dataset](https://github.com/google-research-datasets/Objectron):

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
  <summary><b>Activeloop Hub vs DVC</b></summary>
  
Hub and DVC offer dataset version control similar to git for data, but their methods for storing data differ significantly. Hub converts and stores data as chunked compressed arrays, which enables rapid streaming to ML models, whereas DVC operates on top of data stored in less efficient traditional file structures. The Hub format makes dataset versioning significantly easier compared to traditional file structures by DVC when datasets are composed of many files (i.e., many images). An additional distinction is that DVC primarily uses a command-line interface, whereas Hub is a Python package. Lastly, Hub offers an API to easily connect datasets to ML frameworks and other common ML tools and enables instant dataset visualization through [Activeloop's visualization tool](http://app.activeloop.ai/?utm_source=github&utm_medium=repo&utm_campaign=readme).

</details>


<details>
  <summary><b>Activeloop Hub vs TensorFlow Datasets (TFDS)</b></summary>
  
Hub and TFDS seamlessly connect popular datasets to ML frameworks. Hub datasets are compatible with both PyTorch and TensorFlow, whereas TFDS are only compatible with TensorFlow. A key difference between Hub and TFDS is that Hub datasets are designed for streaming from the cloud, whereas TFDS must be downloaded locally prior to use. As a result, with Hub, one can import datasets directly from TensorFlow Datasets and stream them either to PyTorch or TensorFlow. In addition to providing access to popular publicly available datasets, Hub also offers powerful tools for creating custom datasets, storing them on a variety of cloud storage providers, and collaborating with others via simple API. TFDS is primarily focused on giving the public easy access to commonly available datasets, and management of custom datasets is not the primary focus. A full comparison article can be found [here](https://www.activeloop.ai/resources/tensor-flow-tf-data-activeloop-hub-how-to-implement-your-tensor-flow-data-pipelines-with-hub/).

</details>



<details>
  <summary><b>Activeloop Hub vs HuggingFace</b></summary>
Hub and HuggingFace offer access to popular datasets, but Hub primarily focuses on computer vision, whereas HuggingFace focuses on natural language processing. HuggingFace Transforms and other computational tools for NLP are not analogous to features offered by Hub.


</details>

<details>
  <summary><b>Activeloop Hub vs WebDatasets</b></summary>
Hub and WebDatasets both offer rapid data streaming across networks. They have nearly identical steaming speeds because the underlying network requests and data structures are very similar. However, Hub offers superior random access and shuffling, its simple API is in python instead of command-line, and Hub enables simple indexing and modification of the dataset without having to recreate it.


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

## Citation
If you use Hub in your research, please cite Activeloop using:
```
@article{2022ActiveloopHub,
  title={Hub: A Dataset Format for AI. A simple API for creating, storing, collaborating on AI datasets of any size & streaming them to ML frameworks at scale.},
  author={Activeloop Developer Team},
  journal={GitHub. Note: https://github.com/activeloopai/Hub},
  year={2022}
}
```
## Acknowledgment
This technology was inspired by our research work at Princeton University. We would like to thank William Silversmith @SeungLab for his awesome [cloud-volume](https://github.com/seung-lab/cloud-volume) tool.
