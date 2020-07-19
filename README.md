<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/snarkai/Hub/master/docs/logo/hub_logo.png" width="50%"/>
    </br>
</p>
<p align="center">
    <a href="https://docs.snark.ai/">
        <img alt="Docs" src="https://readthedocs.org/projects/hubdb/badge/?version=latest">
    </a>
</p>




<h3 align="center">
The fastest way to access and manage datasets for PyTorch and TensorFlow
</h3>

Hub provides fast access to the state-of-the-art datasets for Deep Learning, manage them, build data pipelines and connect to Pytorch and Tensorflow with ridiculous ease.

## 1. Quick Start

Let's see how it works in action:
```sh
pip3 install hub
```

Access public datasets in few lines.
```python
import hub

mnist = hub.load("mnist/mnist")
mnist["images"][0:1000].compute()
```

## 2. Train a model

Load the data and directly train a model using pytorch

```python
import hub
import pytorch

cifar = hub.load("cifar/cifar10")
cifar = cifar.to_pytorch()

train_loader = torch.utils.data.DataLoader(
        cifar, batch_size=1, num_workers=0, collate_fn=cifar.collate_fn
)

for images, labels in train_loader:
    # your training loop here
```

## 3. Upload your dataset and access it from anywhere in 3 steps

1. Register a free account at [Snark](https://app.snark.ai) and authenticate locally
```sh
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

3. Access it from anywhere else in the world
```python
import hub

ds = hub.load("username/basic")
```
For more advanced data pipelines please see [docs](https://docs.snark.ai).

## Features
* **Data Management**: Storing large datasets with version control
* **Collaboration**: Multiple data scientists working on the same data in sync
* **Distribute**: Accessing from multiple machines at the same time
* **Machine Learning**: Native integration with Numpy, Dask, PyTorch or TensorFlow.
* **Scale**: Create as big arrays as you want
* **Visualization**: Visualize the data without trouble

## Use Cases
* **Aerial images**: Satellite and drone imagery
* **Medical Images**: Volumetric images such as MRI or Xray
* **Self-Driving Cars**: Radar, 3D LIDAR, Point Cloud, Semantic Segmentation, Video Objects
* **Retail**: Self-checkout datasets
* **Media**: Images, Video, Audio storage

# Examples
- [Waymo Open Dataset](https://medium.com/snarkhub/extending-snark-hub-capabilities-to-handle-waymo-open-dataset-4dc7b7d8ab35)
- [Aptiv nuScenes](https://medium.com/snarkhub/snark-hub-is-hosting-nuscenes-dataset-for-autonomous-driving-1470ae3e1923)


# Problems with Current Workflows

Most of the time Data Scientists/ML researchers work on data management and preprocessing instead of doing modeling. Deep Learning often requires to work with large datasets. Those datasets can grow up to terabyte or even petabyte size. It is hard to manage data, version control and track. It is time consuming to download the data and link with the training or inference code. There is no easy way to access a chunk of it and possibly visualize. **Wouldn’t it be more convenient to have large datasets stored & version-controlled as single numpy-like array on the cloud and have access to it from any machine at scale?**

We realized that there are a few problems related with current workflow in deep learning data management through our experience of working with deep learning companies and researchers.
1. **Data locality**. When you have local GPU servers but store the data in a secure remote data center or on the cloud, you need to plan ahead to download specific datasets to your GPU box because it takes time. Sharing preprocessed dataset from one GPU box across your team is also slow and error-prone if there're multiple preprocessing pipelines.

2. **Code dependency on local folder structure**. People use a folder structure to store images or videos. As a result, the data input pipeline has to take into consideration the raw folder structure which creates unnecessary & error-prone code dependency of the dataset folder structure.

3. **Managing preprocessing pipelines**. If you want to run some preprocessing, it would be ideal to save the preprocessed images as a local cache for training.But it’s usually hard to manage & version control the preprocessed images locally when there are multiple preprocessing pipelies and the dataset is very big.

4. **Visualization**. It's difficult to visualize the raw data or preprocessed dataset on servers.

5. **Reading a small slice of data**. Another popular way is to store in HDF5/TFRecords format and upload to a cloud bucket, but still you have to manage many chunks of HDF5/TFRecords files. If you want to read a small slice of data, it's not clear which TFRecord/HDF5 chunk you need to load. It's also inefficient to load the whole file for a small slice of data.

6. **Synchronization across team**. If multiple users modify the data, there needs to be a data versioning and synchronization protocol implemented.

7. **RAM management**. Whenever you want to create a numpy array you are worried if the numpy array is going to fit in the local RAM/disk limit.


## Benchmarking

For full reproducibility please refer to the [code](/test/benchmark)

### Download Parallelism

The following chart shows that hub on a single machine (aws p3.2xlarge) can achieve up to 875 MB/s download speed with multithreading and multiprocessing enabled. Choosing the chunk size plays a role in reaching maximum speed up. The bellow chart shows the tradeoff using different number of threads and processes.

<img src="https://raw.githubusercontent.com/snarkai/Hub/master/test/benchmark/results/Parallel12MB.png" width="650"/>


### Training Deep Learning Model 

The following benchmark shows that streaming data through Hub package while training deep learning model is equivalent to reading data from local file system. The benchmarks have been produced on AWS using p3.2xlarge machine with V100 GPU. The data is stored on S3 within the same region. In the asynchronous data loading figure, first three models (VGG, Resnet101 and DenseNet) have no data bottleneck. Basically the processing time is greater than loading the data in the background. However for more lightweight models such as Resnet18 or SqueezeNet, training is bottlenecked on reading speed. Number of parallel workers for reading the data has been chosen to be the same. The batch size was chosen smaller for large models to fit in the GPU RAM.  

Training Deep Learning          |  Data Streaming
:-------------------------:|:-------------------------:
<img src="https://raw.githubusercontent.com/snarkai/Hub/master/test/benchmark/results/Training.png" alt="Training" width="440"/>  |   <img src="https://raw.githubusercontent.com/snarkai/Hub/master/test/benchmark/results/Data%20Bottleneck.png" width="440"/>

