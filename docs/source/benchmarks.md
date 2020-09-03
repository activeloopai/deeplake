## Benchmarking

For full reproducibility please refer to the [code](/test/benchmark)

### Download Parallelism

The following chart shows that hub on a single machine (aws p3.2xlarge) can achieve up to 875 MB/s download speed with multithreading and multiprocessing enabled. Choosing the chunk size plays a role in reaching maximum speed up. The bellow chart shows the tradeoff using different number of threads and processes.

<img src="https://raw.githubusercontent.com/snarkai/Hub/master/test/benchmark/results/Parallel12MB.png" width="650"/>


### Training Deep Learning Model 

The following benchmark shows that streaming data through Hub package while training deep learning model is equivalent to reading data from local file system. The benchmarks have been produced on AWS using p3.2xlarge machine with V100 GPU. The data is stored on S3 within the same region. In the asynchronous data loading figure, first three models (VGG, Resnet101 and DenseNet) have no data bottleneck. Basically the processing time is greater than loading the data in the background. However for more lightweight models such as Resnet18 or SqueezeNet, training is bottlenecked on reading speed. Number of parallel workers for reading the data has been chosen to be the same. The batch size was chosen smaller for large models to fit in the GPU RAM.  

**Training Deep Learning**

<img src="https://raw.githubusercontent.com/snarkai/Hub/master/test/benchmark/results/Training.png" alt="Training" width="440"/>  


**Data Streaming**

 <img src="https://raw.githubusercontent.com/snarkai/Hub/master/test/benchmark/results/Data%20Bottleneck.png" width="440"/>