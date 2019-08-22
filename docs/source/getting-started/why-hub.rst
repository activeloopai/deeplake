Why Use Hub Array?
#########################################

We have observed the following problems with existing deep learning data pipelines. 
If you identify with either of the problems, you should give Hub Array a try.

* **Data locality**. When you have local GPU servers but store the data in a secure remote data center or on the cloud,
  you need to plan ahead to download specific datasets to your GPU box because it takes time.
  Sharing preprocessed dataset from one GPU box across your team is also slow and error-prone if there're multiple preprocessing pipelines.

* **Code dependency on local folder structure**. People use a folder structure to store images or videos. 
  As a result, the data input pipeline has to take into consideration the raw folder structure which creates 
  un-necessary & error-prone code dependency of the dataset folder structure.

* **Managing preprocessing pipelines**. If you want to run some preprocessing, it would be ideal to save the preprocessed images as a local cache for training.
  But it’s usually hard to manage & version control the preprocessed images locally 
  when there are multiple preprocessing pipelies and the dataset is very big.

* **Visualization**. It's difficult to visualize the raw data or preprocessed dataset on servers.

* **Reading a small slice of data**. Another popular way is to store in HDF5/TFRecords format and upload to a cloud bucket,
  but still you have to manage many chunks of HDF5/TFRecords files. If you want to read a small slice of data, it's not clear which
  tfrecord/HDF5 chunk you need to load. It's also inefficient to load the whole file for a small slice of data.

* **Synchronization across team**. If multiple users modify the data, there needs to be a data versioning and synchronization protocol implemented.

* **RAM management**. Whenever you want to create a numpy array you are worried if the numpy array is going to fit in the local RAM/disk limit.