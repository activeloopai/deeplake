<img src="https://static.scarf.sh/a.png?x-pxid=bc3c57b0-9a65-49fe-b8ea-f711c4d35b82" /><p align="center">
     <img src="https://i.postimg.cc/rsjcWc3S/deeplake-logo.png" width="400"/>
</h1>
    </br>
    <h1 align="center">Deep Lake: Data Lake for Deep Learning
 </h1>
<p align="center">
    <a href="https://github.com/activeloopai/Hub/actions/workflows/test-pr-on-label.yml"><img src="https://github.com/activeloopai/Hub/actions/workflows/test-push.yml/badge.svg" alt="PyPI version" height="18"></a>
    <a href="https://pypi.org/project/deeplake/"><img src="https://badge.fury.io/py/deeplake.svg" alt="PyPI version" height="18"></a>
    <a href='https://docs.deeplake.ai/en/latest/?badge=latest'>
     <img src='https://readthedocs.org/projects/deep-lake/badge/?version=latest' alt='Documentation Status' />
     </a>
    <a href="https://pepy.tech/project/deeplake"><img src="https://static.pepy.tech/personalized-badge/deeplake?period=month&units=international_system&left_color=grey&right_color=orange&left_text=Downloads" alt="PyPI version" height="18"></a>
     <a href="https://github.com/activeloopai/deeplake/issues">
    <img alt="GitHub issues" src="https://img.shields.io/github/issues/activeloopai/deeplake"> </a>
    <a href="https://codecov.io/gh/activeloopai/deeplake/branch/main"><img src="https://codecov.io/gh/activeloopai/deeplake/branch/main/graph/badge.svg" alt="codecov" height="18"></a>
  <h3 align="center">
   <a href="https://docs.activeloop.ai/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme"><b>Documentation</b></a> &bull;
   <a href="https://docs.activeloop.ai/getting-started/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme"><b>Getting Started</b></a> &bull;
   <a href="https://docs.deeplake.ai/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme"><b>API Reference</b></a> &bull;  
   <a href="https://github.com/activeloopai/examples/"><b>Examples</b></a> &bull; 
   <a href="https://www.activeloop.ai/resources/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme"><b>Blog</b></a> &bull; 
   <a href="https://www.deeplake.ai/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme"><b>Whitepaper</b></a> &bull;  
  <a href="http://slack.activeloop.ai"><b>Slack Community</b></a> &bull;
  <a href="https://twitter.com/intent/tweet?text=The%20dataset%20format%20for%20AI.%20Stream%20data%20to%20PyTorch%20and%20Tensorflow%20datasets&url=https://activeloop.ai/&via=activeloopai&hashtags=opensource,pytorch,tensorflow,data,datascience,datapipelines,activeloop,databaseforAI"><b>Twitter</b></a>
 </h3>
 

*Read this in other languages: [简体中文](README.zh-cn.md)*

## About Deep Lake

<b> 🚀 Deep Lake's efficient new C++ implementation speeds up data streaming by >2x compared to Hub 2.x (Ofeidis et al. 2022) 🚀</b>


Deep Lake (formerly known as Activeloop Hub) is a data lake for deep learning applications. Open-source dataset format provides a simple API for creating, storing, and collaborating on AI datasets of any size. It enables you to store all of your data in one place, ranging from simple annotations to large videos, and it unlocks rapid streaming of data while training models at scale. Deep Lake is used by Google, Waymo, Red Cross, Omdena, Yale, & Oxford. Deep Lake includes the following features:

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
Deep Lake comes with built-in integrations for Pytorch and Tensorflow. Train your model with a few lines of code - we even take care of dataset shuffling. :)
</details>
<details>
  <summary><b>Distributed Transformations</b></summary>
Rapidly apply transformations on your datasets using multi-threading, multi-processing, or our built-in <a href="https://www.ray.io/">Ray</a> integration.</details>
<details>
  <summary><b>100+ most-popular image, video, and audio datasets available in seconds</b></summary>
Deep Lake community has uploaded <a href="https://docs.activeloop.ai/datasets/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme">100+ image, video and audio datasets</a> like <a href="https://docs.activeloop.ai/datasets/mnist/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme">MNIST</a>, <a href="https://docs.activeloop.ai/datasets/coco-dataset/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme">COCO</a>,  <a href="https://docs.activeloop.ai/datasets/imagenet-dataset/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme">ImageNet</a>,  <a href="https://docs.activeloop.ai/datasets/cifar-10-dataset/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme">CIFAR</a>,  <a href="https://docs.activeloop.ai/datasets/gtzan-genre-dataset/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme">GTZAN</a> and others.
</details>
</details>
<details>
  <summary><b>Instant Visualization Support in <a href="https://app.activeloop.ai/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme">Activeloop Platform</a></b></summary>
Deep Lake datasets are instantly visualized with bounding boxes, masks, annotations, etc. in <a href="https://app.activeloop.ai/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme">Deep Lake Visualizer</a> (see below).
</details>


<div align="center">
<a href="https://www.linkpicture.com/view.php?img=LPic61b13e5c1c539681810493"><img src="https://www.linkpicture.com/q/ReadMe.gif" type="image"></a>
</div>

    
## Getting Started with Deep Lake


### 🚀 How to install Deep Lake
Deep Lake's core is efficiently built in C++ and can be quickly installed using pip.

```sh
pip3 install deeplake
```

**By default, Deep Lake does not install dependencies for audio, video, google-cloud, and other features. Details on all installation options are [available here](https://docs.deeplake.ai/en/latest/Installation.html).**

### 🧠 Training a PyTorch model on a Deep Lake dataset

#### Load CIFAR 10, one of the readily available datasets in Deep Lake:

```python
import deeplake
import torch
from torchvision import transforms, models

ds = deeplake.load('hub://activeloop/cifar10-train')
```

#### Inspect tensors in the dataset:

```python
ds.tensors.keys()    # dict_keys(['images', 'labels'])
ds.labels[0].numpy() # array([6], dtype=uint32)
```

#### Train a PyTorch model on the CIFAR 10 dataset without the need to download it

First, define a transform for the images and use Deep Lake's built-in PyTorch one-line dataloader to connect the data to the compute:

```python
tform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

deeplake_loader = ds.pytorch(num_workers=0, batch_size=4, transform={
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
    for i, data in enumerate(deeplake_loader):
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


### 🏗️ How to create a Deep Lake Dataset

A Deep Lake dataset can be created in various locations (Storage providers). This is how the paths for each of them would look like:

| Storage provider        | Example path                   |
| ----------------------- | ------------------------------ |
| Activeloop cloud        | hub://user_name/dataset_name   |
| AWS S3 / S3 compatible  | s3://bucket_name/dataset_name  |
| GCP                     | gcp://bucket_name/dataset_name |
| Google Drive            | gdrive://path_to_dataset
| Local storage           | path to local directory        |
| In-memory               | mem://dataset_name             |



Let's create a dataset in the Activeloop cloud. Activeloop cloud provides free storage up to 300 GB per user (more info [here](#-for-students-and-educators)). Create a new account with Deep Lake from the terminal using `activeloop register` or in the [Deep Lake UI](https://app.activeloop.ai/register/). You will be asked for a user name, email ID, and password.

```sh
$ activeloop register
Enter your details. Your password must be at least 6 characters long.
Username:
Email:
Password:
```

After registration, an ORGANIZATION is automatically created that shares your username. You can use it for creating and managing your datasets, or you can create a new one for your company or team.

Initialize an empty dataset in the Activeloop Cloud:

```python
import deeplake

ds = deeplake.empty('hub://<ORGANIZATION_NAME>/test-dataset')
```


Next, create a tensor to hold images in the dataset we just initialized:

```python
images = ds.create_tensor('images', htype='image', sample_compression='jpg')
```

Assuming you have a list of image file paths, let's upload them to the dataset:

```python
image_paths = ...
with ds:
    for image_path in image_paths:
        image = deeplake.read(image_path)
        ds.images.append(image)
```

Alternatively, you can also upload numpy arrays. Since the `images` tensor was created with `sample_compression='jpg'`, the arrays will be compressed with jpeg compression.


```python
import numpy as np

with ds:
    for _ in range(1000):  # 1000 random images
        random_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)  # 100x100 image with 3 channels
        ds.images.append(random_image)
```



### 🚀 How to load a Deep Lake Dataset


You can load the dataset you just created with a single line of code:

```python
import deeplake

ds = deeplake.load('hub://<ORGANIZATION_NAME>/test-dataset')
```

You can also access one of the <a href="https://docs.activeloop.ai/datasets/?utm_source=github&utm_medium=github&utm_campaign=github_readme&utm_id=readme">100+ image, video and audio datasets in Deep Lake format</a>, not just the ones you created. Here is how you would load the [Objectron Bikes Dataset](https://github.com/google-research-datasets/Objectron):

```python
import deeplake

ds = deeplake.load('hub://activeloop/objectron_bike_train')
```

To get the first image in the Objectron Bikes dataset in numpy format:


```python
image_arr = ds.image[0].numpy()
```



## 📚 Documentation
Getting started guides, examples, tutorials, API reference, and other useful information can be found on our [documentation page](http://docs.activeloop.ai/?utm_source=github&utm_medium=repo&utm_campaign=readme). 

## 🎓 For Students and Educators
Deep Lake users can access and visualize a variety of popular datasets through a free integration with Activeloop's Platform. Users can also create and store their own datasets and make them available to the public. Free storage of up to 300 GB is available for students and educators:

| <!-- -->    | <!-- -->    |
| ---------------------------------------------------- | ------------- |
| Storage for public datasets hosted by Activeloop     | 200GB Free    |
| Storage for private datasets hosted by Activeloop    | 100GB Free    |



## 👩‍💻 Comparisons to Familiar Tools


<details>
  <summary><b>Deep Lake vs DVC</b></summary>
  
Deep Lake and DVC offer dataset version control similar to git for data, but their methods for storing data differ significantly. Deep Lake converts and stores data as chunked compressed arrays, which enables rapid streaming to ML models, whereas DVC operates on top of data stored in less efficient traditional file structures. The Deep Lake format makes dataset versioning significantly easier compared to traditional file structures by DVC when datasets are composed of many files (i.e., many images). An additional distinction is that DVC primarily uses a command-line interface, whereas Deep Lake is a Python package. Lastly, Deep Lake offers an API to easily connect datasets to ML frameworks and other common ML tools and enables instant dataset visualization through [Activeloop's visualization tool](http://app.activeloop.ai/?utm_source=github&utm_medium=repo&utm_campaign=readme).

</details>


<details>
  <summary><b>Deep Lake vs TensorFlow Datasets (TFDS)</b></summary>
  
Deep Lake and TFDS seamlessly connect popular datasets to ML frameworks. Deep Lake datasets are compatible with both PyTorch and TensorFlow, whereas TFDS are only compatible with TensorFlow. A key difference between Deep Lake and TFDS is that Deep Lake datasets are designed for streaming from the cloud, whereas TFDS must be downloaded locally prior to use. As a result, with Deep Lake, one can import datasets directly from TensorFlow Datasets and stream them either to PyTorch or TensorFlow. In addition to providing access to popular publicly available datasets, Deep Lake also offers powerful tools for creating custom datasets, storing them on a variety of cloud storage providers, and collaborating with others via simple API. TFDS is primarily focused on giving the public easy access to commonly available datasets, and management of custom datasets is not the primary focus. A full comparison article can be found [here](https://www.activeloop.ai/resources/tensor-flow-tf-data-activeloop-hub-how-to-implement-your-tensor-flow-data-pipelines-with-hub/).

</details>



<details>
  <summary><b>Deep Lake vs HuggingFace</b></summary>
Deep Lake and HuggingFace offer access to popular datasets, but Deep Lake primarily focuses on computer vision, whereas HuggingFace focuses on natural language processing. HuggingFace Transforms and other computational tools for NLP are not analogous to features offered by Deep Lake.


</details>

<details>
  <summary><b>Deep Lake vs WebDatasets</b></summary>
Deep Lake and WebDatasets both offer rapid data streaming across networks. They have nearly identical steaming speeds because the underlying network requests and data structures are very similar. However, Deep Lake offers superior random access and shuffling, its simple API is in python instead of command-line, and Deep Lake enables simple indexing and modification of the dataset without having to recreate it.


</details>

<details>
  <summary><b>Deep Lake vs Zarr</b></summary>
Deep Lake and Zarr both offer storage of data as chunked arrays. However, Deep Lake is primarily designed for returning data as arrays using a simple API, rather than actually storing raw arrays (even though that's also possible). Deep Lake stores data in use-case-optimized formats, such as jpeg or png for images, or mp4 for video, and the user treats the data as if it's an array, because Deep Lake handles all the data processing in between. Deep Lake offers more flexibility for storing arrays with dynamic shape (ragged tensors), and it provides several features that are not naively available in Zarr such as version control, data streaming, and connecting data to ML Frameworks.


</details>

## Community

Join our [**Slack community**](https://join.slack.com/t/hubdb/shared_invite/zt-ivhsj8sz-GWv9c5FLBDVw8vn~sxRKqQ) to learn more about unstructured dataset management using Deep Lake and to get help from the Activeloop team and other users.

We'd love your feedback by completing our 3-minute [**survey**](https://forms.gle/rLi4w33dow6CSMcm9).

As always, thanks to our amazing contributors!    

<a href="https://github.com/activeloopai/deeplake/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=activeloopai/hub" />
</a>

Made with [contributors-img](https://contrib.rocks).

Please read [CONTRIBUTING.md](CONTRIBUTING.md) to get started with making contributions to Deep Lake.


## README Badge

Using Deep Lake? Add a README badge to let everyone know: 


[![deeplake](https://img.shields.io/badge/powered%20by-Deep%20Lake%20-ff5a1f.svg)](https://github.com/activeloopai/deeplake)

```
[![deeplake](https://img.shields.io/badge/powered%20by-Deep%20Lake%20-ff5a1f.svg)](https://github.com/activeloopai/deeplake)
```



## Disclaimers

<details>
  <summary><b> Dataset Licenses</b></summary>
  
Deep Lake users may have access to a variety of publicly available datasets. We do not host or distribute these datasets, vouch for their quality or fairness, or claim that you have a license to use the datasets. It is your responsibility to determine whether you have permission to use the datasets under their license.

If you're a dataset owner and do not want your dataset to be included in this library, please get in touch through a [GitHub issue](https://github.com/activeloopai/deeplake/issues/new). Thank you for your contribution to the ML community!

</details>

<details>
  <summary><b> Usage Tracking</b></summary>

By default, we collect usage data using Bugout (here's the [code](https://github.com/activeloopai/deeplake/blob/853456a314b4fb5623c936c825601097b0685119/deeplake/__init__.py#L24) that does it). It does not collect user data other than anonymized IP address data, and it only logs the Deep Lake library's own actions. This helps our team understand how the tool is used and how to build features that matter to you! After you register with Activeloop, data is no longer anonymous. You can always opt-out of reporting using the CLI command below, or by setting an environmental variable ```BUGGER_OFF``` to ```True```:

```
activeloop reporting --off
```
</details>

## Citation
If you use Deep Lake in your research, please cite Activeloop using:

```
@article{deeplake,
  doi = {10.48550/ARXIV.2209.10785},
  url = {https://arxiv.org/abs/2209.10785},
  author = {Hambardzumyan, Sasun and Tuli, Abhinav and Ghukasyan, Levon and Rahman, Fariz and Topchyan, Hrant and Isayan, David and Harutyunyan, Mikayel and Hakobyan, Tatevik and Stranic, Ivo and Buniatyan, Davit},
  title = {Deep Lake: a Lakehouse for Deep Learning},
  publisher = {arXiv},
  year = {2022},
}
```

## Acknowledgment
This technology was inspired by our research work at Princeton University. We would like to thank William Silversmith @SeungLab for his awesome [cloud-volume](https://github.com/seung-lab/cloud-volume) tool.
