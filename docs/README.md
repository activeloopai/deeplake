---
home: false
title: Getting started
---

## What is Hub Array?
Hub Array is a scalable numpy-like array stored on the cloud accessible over network as if they're local numpy arrays. You can create a Hub Array on your local environment as large as 5TB, use it as a local numpy array without worrying if the local disk/RAM will hold it or not. The Array is created on an object storage on the cloud and cached partially on your local environment. All read/write to the Array is automatically synchronized to the bucket.

Hub Array aims to provide a cloud-based data management solution for deep learning practitioners. Key features will include version control, feature store, data sharing and visualization for computer vision & NLP tasks, 

## Why Use Hub Array?
We have observed the following problems with existing deep learning data pipelines. If you identify with either of the problems, you should give Hub Array a try.
- People use a folder structure to store images or videos. As a result, the data input pipeline has to take into consideration the raw folder structure which creates **un-necessary & error-prone code dependency of the dataset folder structure**.
- If you want to run some preprocessing, it would be ideal to cache the preprocessed images for training. But it’s usually **hard to cache & version control the preprocessed images locally** if the dataset is very big.
- If you have various preprocessing pipelines, it's often **adhoc to link the raw data, the preprocessing pipeline and the preprocessed dataset together**. The link usually requires writing some mapping rules that depend on local folder structure or environment variables, which hurts reproducibility in another environment.
- It's difficult to **visualize the raw data or preprocessed dataset** on servers.
- Another popular way is to store in HDF5/TFRecords format and upload to a cloud bucket, but still you have to **manage many chunks of HDF5/TFRecords files**.
- If multiple users modify the data, there needs to be a **data versioning and synchronization protocol** implemented.
- There is no easy way to **access only a single image from big HDF5/TFRecords file without downloading the entire file**.
- Whenever you want to create a numpy array **you are worried if the numpy array is going to fit in the local RAM/disk limit**.

## Getting Started

**Step 1.** Install
```sh
pip3 install hub
```

**Step 2.** Lazy-load a public dataset, and fetch a single image with up to 50MB/s speed and plot
```python
> import hub
> imagenet = hub.load(name='imagenet')
> imagenet.shape
(1034908, 469, 387, 3)

> import matplotlib.pyplot as plt
> plt.imshow(imagenet[0])
```

**Step 3.** Compute the mean and standard deviation of any chunk of the full dataset. The package will download the chunk to the local environment and compute locally as a numpy array.
```python
> imagenet[0:10,100:200,100:200].mean()
0.132
> imagenet[0:10,100:200,100:200].std()
0.005
```

**Step 4.** Create your own array and access it from another machine
```python
# Create on one machine
> import numpy as np
> mnist = hub.array((50000,28,28,1), name="name/random_name:v1")
> mnist[0,:,:,:] = np.random.random((1,28,28,1))

# Access it from another machine
> mnist = hub.load(name='name/random_name:v1')
> print(mnist[0])
```

