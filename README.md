# Introduction
Most of the time Data Scientists/ML researchers work on data management and preprocessing instead of doing modeling. Deep Learning often requires to work with large datasets. Those datasets can grow up to terabyte or even petabyte size. It is hard to manage data, version control and track. It is time consuming to download the data and link with the training or inference code. There is no easy way to access a chunk of it and possibly visualize. **Wouldn’t it be more convenient to have large datasets stored & version-controlled as single numpy-like array on the cloud and have access to it from any machine at scale?**

> **Introducing Hub Arrays**: scalable numpy-like arrays stored on the cloud accessible over internet as if they're local numpy arrays.

# Problems with Current Workflows
We realized that there are a few problems related with current workflow in deep learning data management through our experience of working with deep learning companies and researchers.
1. People use a folder structure to store images or videos. As a result, the data input pipeline has to take into consideration the raw folder structure which creates un-necessary & error-prone code dependency of the dataset folder structure.
2. If you want to run some preprocessing, it would be ideal to cache the preprocessed images for training. But it’s usually hard to cache the preprocessed images locally if the dataset is very big. Version control of caches is also a problem.
3. Another popular way is to store in HDF5/TFRecords format and upload to a cloud bucket, but still you have to manage many chunks of HDF5/TFRecords files.
4. If multiple users modify the data, there needs to be a data versioning protocol implemented and synchronization methods not to overwrite the data.
5. There is no easy way to access only a single image from big HDF5/TFRecords file without downloading the whole chunk.
6. Whenever you want to create a numpy array you are worried if it is going to fit in the RAM limit.

# Workflow with Hub Arrays
Simply declare an array with the namespace inside the code and thats it. “Where and How the data is stored?” is totally abstracted away from the data scientist or machine learning engineer. **You can create a numpy array up to Petabytes scale without worrying if the array will fit into RAM or local disk.** The inner workings are like this:
1. The actual array is created on a cloud bucket (object storage) and partially cached on your local environment. The array size can easily scale to 1PB.
2. When you read/write to the array, the package automatically synchronize the change from local to cloud bucket via internet.

We’re working on simple authentication system, data management, advanced data caching & fetching, and version controls.

```python
> import hub
> import numpy as np

# Create a large array that you can read/write from anywhere.
> bigarray = hub.array((100000, 512, 512, 3), name="test/bigarray:v0")

# Writing to one slice of the array. Automatically syncs to cloud.
> image = np.random((512,512,3))
> bigarray[0, :,:, :] = image

# Load an existing array from cloud
> imagenet = hub.load(name='imagenet')
> imagenet.shape
(1034908, 469, 387, 3)
```

## Usage
1. Install 
```sh
$pip3 install hub
```

Then provide AWS credentials and name of your project.
```sh
$hub configure
```
It will create a bucket to store the data 

2. Load a public dataset on-demand with up to 50MB/s speed and plot
```python
> import hub
> imagenet = hub.load(name='imagenet')
> imagenet.shape
(1034908, 469, 387, 3)

>import matplotlib.pyplot as plt
> plt.imshow(imagenet[0])
```

3. Compute the mean and standard deviation of any chunk of the full dataset
```python
> imagenet[0:10,100:200,100:200].mean()
0.132
> imagenet[0:10,100:200,100:200].std()
0.005
```

4. Create your own array and access it from another machine
```python
# Create on one machine
> import numpy as np
> mnist = hub.array((50000,28,28,1), name="name/random_name:v1")
> mnist[0,:,:,:] = np.random.random((1,28,28,1))

# Access it from another machine
> mnist = hub.load(name='name/random_name:v1')
> print(mnist[0])
```


## Features
* **Data Management**: Storing large datasets with version control
* **Collaboration**: Multiple data scientists working on the same data in sync
* **Distribute**: Accessing from multiple machines at the same time
* **Machine Learning**: Native integration with Numpy, Dask, PyTorch or TensorFlow.
* **Scale**: Create as big arrays as you want
* **Visualization**: Visualize the data without trouble

## Use Cases
* **Areal images**: Satellite and drone imagery
* **Medical Images**: Volumetric images such as MRI or Xray
* **Self-Driving Cars**: Radar, 3D LIDAR, Point Cloud, Semantic Segmentation, Video Objects
* **Retail**: Self-checkout datasets
* **Media**: Images, Video, Audio storage

### Acknolowdgement
Acknowledgment: This technology was inspired from our experience at Princeton University at SeungLab and would like to thank William Silversmith @SeungLab and his awesome project [cloud-volume](https://github.com/seung-lab/cloud-volume).
