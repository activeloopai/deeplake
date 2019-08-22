FAQ
#####################

Q: Will my code stuck on internet download when I run ``hub.load`` to load the dataset?

A: No. ``hub.load`` is a lazy-load. It only downloads the meta data such as shape and data types.
The actual data IO happens when you try to access any slice/chunk of the array.


Q: Where is the data actually stored?

A: It is stored on an object storage. It is by default an object storage on Snark AI's AWS S3 bucket. 
The dev team is working on features so that you can also configure the storage location to be your own
Google Cloud Storage, Azure Blob Storage or AWS S3.


Q: If I read data from Hub Array for deep learning training, will the training speed get hurt? 

A: Performance benchmarks show that training speed is not slowed down if the backend storage and the
compute node is on the same region on AWS/GCP/Azure. The raw data read speed can be as fast as 800MB/s 
on AWS S3 which is faster than reading from local SSD. This is made possible because S3 provides horizontal 
scalability in data I/O.