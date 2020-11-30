# Tutorials

This examples show how to use Hub data structures.

## Creating an array
Hub Arrays are scalable numpy-like arrays stored on the cloud accessible over the internet as if they're local numpy arrays. Their special chunkified structure makes it super fast to interact with them.

### Establish connection to cloud 
In order to store in the cloud, first, you should connect to it.  
For AWS S3 storage. 
```python
import hub
datahub = hub.s3("your_bucket_name", aws_creds_filepath="filepath_to_your_credentials").connect()
```
Google Cloud.
```python
datahub_gs = hub.gs("your_bucket_name", creds_path="filepath_to_your_credentials.json").connect()
```

Local filesystem.
```python
datahub_fs = hub.fs("./data/cache").connect()
```

Now you can simply create Hub Array in the storage and start working with it.
```python
bigarray = datahub.array('your_array_name', shape=(100000, 512, 512, 3), chunk=(100, 512, 512, 3), dtype='int32')
# After creating an array all elements are set to 0s
import numpy as np
image = np.random.random((512,512,3))
# Writing to one slice of the array. Automatically syncs to cloud.
bigarray[0, :,:, :] = image
```

You can open the array from any place/computer that has access to internet and write/read to it 
```python
bigarray = datahub.open('your_array_name')
assert bigarray.shape == (100000, 512, 512, 3)
# note only the requested part of the array is downloaded into RAM
# here arr becomes a regular numpy array.
arr = bigarray[0, :5, :, :]
```

## Creating a dataset
Hub Datasets are dictionaries of Hub Arrays stored in the cloud.  
Usage example.

```python
import hub
datahub = hub.s3("your_bucket_name", aws_creds_filepath="filepath_to_your_credentials").connect()
x = datahub.array(
    name="test/example:input", shape=(100, 25, 25), chunk=(20, 5, 5), dtype="uint8"
)
y = datahub.array(
    name="test/example:label", shape=(100, 4), chunk=(20, 2), dtype="uint8"
)

ds = datahub.dataset(
    name="test/dataset:train", components={"input": x, "label": y} 
)
```

You can open the dataset just like with arrays and write/read to it.

```python
ds = datahub.open("test/dataset:train")
import numpy as np
ds["input"][1:3] =  np.ones((2, 25, 25))
# alternative syntax
ds["input", 1:3] =  np.ones((2, 25, 25))
```

## Idea of chunking 
Chunks are the most important part of Hub arrays. Imagine that you have a really large array stored in the cloud and want to access only some significantly smaller part of it. Let us say you have an array of 100000 images with shape ```(100000, 1024, 1024, 3)```. If we stored this array wholly without dividing into multiple chunks then in order to request only few images from it we would need to load the entire array into RAM which would be impossible and even if some computer would have that big RAM, downloading the whole array would take a lot of time. Instead we store the array in chunks and we only download the chunks that contain the requested part of the array.  

## How to choose a proper chunk size
Choosing a proper chunk size is crucial for performance. The chunks must be much bigger and take longer time to download than the overhead of request to cloud ~1ms. Chunks also should be small enough to fit multiple chunks into RAM. Usually, we can have up to 1 chunk per thread. 

Chunks should have usually from 1MB to 1GB size depending on a specific problem and on RAM size.

Choosing the right chunk shape is also important. If for example we have 2-dimensional array ```shape = (100000, 100000)``` and we know that in general, we will slice along the first dimension than it is better to have a smaller chunk shape for the first dimension and bigger shape for the second dimension ```chunk = (100, 100000)```. 

## Compression 
Today datasets can have massive sizes and it can take a lot of resources to store them in the cloud that is why it is important to compress them before storing. Hub provides many compressions that can be applied to arrays. Hub arrays support the following compressors: 'gzip', 'zlib', 'lz4', 'jpeg', 'png'. If no compressor is specified Hub uses 'default' compressor of pickle library when serializing the data. 

```python
bigarray = datahub.array('your_array_name', shape=(100000, 512, 512, 3), chunk=(100, 512, 512, 3), dtype='int32', compress="gzip", compresslevel=0.3)
```

Compresslevel is a float number from 0 to 1. Where 1 is the fastest and 0 is the most compressed. 
You can easily find about all of our supported compressors, their effectiveness, and performance in the internet.  

## Integration with Pytorch and TensorFlow
Hub datasets can easily be transformed into Pytorch and Tensorflow formats.
Pytorch:
```python
    datahub = hub.fs("./data/cache").connect()
    images = datahub.array(name="test/dataloaders/images3", shape=(100, 100, 100), chunk=(1, 100, 100), dtype="uint8")
    labels = datahub.array(name="test/dataloaders/labels3", shape=(100, 1), chunk=(100, 1), dtype="uint8")
    ds = datahub.dataset(name="test/loaders/dataset2", components={"images": images, "labels": labels})
    # Transform to Pytorch
    train_dataset = ds.to_pytorch()
    # Create data loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, num_workers=2, pin_memory=False, shuffle=False,
        drop_last=False,
    )
    # access batch
    batch = next(iter(train_loader))
    assert batch["images"].shape == (32, 100, 100)
    assert batch["labels"].shape == (32, 1)
```

TensorFlow:
```python
    datahub = hub.fs("./data/cache").connect()
    images = datahub.array(name="test/dataloaders/images3", shape=(100, 100, 100), chunk=(1, 100, 100), dtype="uint8")
    labels = datahub.array(name="test/dataloaders/labels3", shape=(100, 1), chunk=(100, 1), dtype="uint8")
    # Create dataset
    ds = datahub.dataset(name="test/loaders/dataset2", components={"images": images, "labels": labels})
    # Transform to Tensorflow
    train_dataset = ds.to_tensorflow()
    # access batch
    batch = next(iter(train_dataset.batch(batch_size=16)))
    assert batch["images"].shape == (16, 100, 100)
```

## Tips on using Hub arrays
When you want to get numpy array from Hub array just add the slicing. 

```python
bigarray = datahub.open('your_array_name')
arr = bigarray[0:10, :, :, :2]
```
Now arr is a regular numpy array. 

When assigning something to Hub array please note that it is faster to do this together than part by part. For example:
```python
bigarray = datahub.array('your_array_name', shape=(100000, 512, 512, 3), chunk=(100, 512, 512, 3), dtype='int32', compress="gzip", compresslevel=0.3)
import numpy as np
bigarray[0:1] = np.random.random((2, 512, 512, 3))
```

Is faster than doing 

```python
bigarray = datahub.array('your_array_name', shape=(100000, 512, 512, 3), chunk=(100, 512, 512, 3), dtype='int32', compress="gzip", compresslevel=0.3)
import numpy as np
bigarray[0] = np.random.random((512, 512, 3))
bigarray[1] = np.random.random((512, 512, 3))
```

This happens because in the first example we do one request to cloud and in the second example we do two requests.  

