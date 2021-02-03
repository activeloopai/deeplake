
# Benchmarking

## Motivation
As the number of Hub users grew, it seemed wise to verify one of the key advantages of Hub: its performance. A standard way to measure the performance of a framework is to provide a process for comparisons to discover the industry winner under the same conditions and metrics. Hub claims to be:

> Fastest unstructured dataset management for TensorFlow/PyTorch.

The goal of the benchmarks is to show what areas of performance this claim applies to and to guide Hub's team towards in which Hub has still some room for improvement. The benchmarks are split into internal and external ones. The former suggest the relative conditions which are optimal for Hub to maximize its performance. The latter are to determine Hub's place on the ML scene among other actors like *PyTorch*, *Tensorflow*, *zarr* or *TileDB*.

## Method

All of the benchmarks were conducted on the same machine unless stated otherwise in a section related to a particular benchmark. The specification of the resources used for the benchmarks can be found below:

### Computation
<table>
  <tr>
    <th>Machine</th>
    <td>AWS EC2 m4.10xlarge instance</td>
  </tr>
  <tr>
    <th>Region</th>
    <td>US-East-2c</td>
  </tr>
    <tr>
    <th>Memory</th>
    <td>160 GB</td>
  </tr>
    <tr>
    <th>CPU</th>
    <td>Intel(R) Xeon(R) CPU E5-2676 v3 @ 2.40GHz</td>
  </tr>
    <tr>
    <th>#vCPU</th>
    <td>40</td>
  </tr>
     <tr>
    <th>Network performance</th>
    <td>10 Gb</td>
  </tr>
</table>

### Storage

| Type of storage | Volume | Maximum storage bandwidth |
| --- | --- | --- |
| Instance storage (EBS) | 1000 GB | 4000 Mbps |
| S3 Bucket | *unlimited* | 25 Gbps |
| Wasabi | | |

### Operating System
<table>
  <tr>
    <th>Kernel</th>
    <td>4.14.214-160.339.amzn2.x86_64 GNU/Linux</td>
  </tr>
  <tr>
    <th>OS Name</th>
    <td>Amazon Linux 2 (Karoo)</td>
  </tr>
    <tr>
    <th>Filesystem</th>
    <td>xfs</td>
  </tr>
</table>

### Datasets

#### Internal Use

| Name | Data Description | Split | Size (MB) | Number of items |
| --- | --- | --- | ---: | ---: |
| [MNIST](https://app.activeloop.ai/dataset/activeloop/mnist) | 28x28 grayscale images with 10 class labels | train + test | 23 | 70000 |
| [Omniglot](https://app.activeloop.ai/dataset/activeloop/omniglot_test) | 105x105 color images with 1623 class labels | test |  | 13180 |
| [CIFAR10](https://app.activeloop.ai/dataset/activeloop/cifar10_train) | 32x32 color images with 10 class labels | train | 116 | 50000 |
| [CIFAR100](https://app.activeloop.ai/dataset/activeloop/cifar100_train) | 32x32 color images with 100 class labels | train | 116 | 50000 |

#### External Use
| Name | Data Description | Pytorch Resource | Tensorflow Resource | Split | Size (MB) | Number of items |
| --- |  --- | --- | --- | --- | ---: | ---: |
| [MNIST](https://app.activeloop.ai/dataset/activeloop/mnist) | 28x28 grayscale images with 10 class labels | [`torchvision.datasets.MNIST()`](https://pytorch.org/docs/stable/torchvision/datasets.html#mnist) | [`tfds.load("mnist")`](https://www.tensorflow.org/datasets/catalog/mnist) | train + test | 23 | 70000 |
| [Places365_small](https://app.activeloop.ai/dataset/hydp/places365_small_train) | 256x256 color images with 365 class labels | [`torchvision.datasets.Places365(small=True)`](https://pytorch.org/docs/stable/torchvision/datasets.html#places365) | [`tfds.load("places365_small")`](https://www.tensorflow.org/datasets/catalog/places365_small) | train | 23671 | 1803460 |

### Configuration

In all of the benchmarks caching (including storage caching) is disabled.

Some benchmarks are parametrized by a variety of arguments, such as:
* dataset
* batch size
* prefetch factor
* number of workers

Relevant configuration details for the parametrized benchmarks are noted in respective sections.

## Reproducibility

Presented benchmarks are intended to be reproducible and easy to replicate manually or through automation.
### Step by step guide
1. Launch the AWS EC2 instance according to the specification in the Method section.
2. Install Hub in the edit mode along with the necessary packages found in all of the requirements files or run `sh benchmark_setup.sh` and source into the virtual environment with `source ./hub-env/bin/activate`.
3. Sequentially run all of the Python files in the `benchmarks` folder. The results for most benchmarks are released to the standard output. For the external dataset iteration benchmark only, you may collect the results with `grep 'BENCHMARK'`.

Note that access to the datasets stored in the S3 bucket is limited. However, you might replicate this set-up by creating a bucket which contains the data in Hub format.

## Internal Benchmarks

### Image Compression

We measure the time to compress (PNG) a [sample image](images/compression_benchmark_image.png) using PIL and Hub.

#### Results

The results below measure compression time of the sample image at a batch size of 100.

```
PIL compression: 25.102458238601685s
Hub compression: 25.102383852005005s
```

#### Observations
There are no drops of performance of Hub in relation to the Python Imaging Library while compressing images. In fact, Hub performs slightly better than `PIL` library.

### Random Access

We measure the time to fetch an uncached random sample from a dataset,
varying over several standard datasets and further at several batch sizes.
Random offsets are also used to ensure that no caching is being taken advantage of externally.

#### Results

```
activeloop/mnist read at offset 613 of length 001: 0.506598711013794s
activeloop/mnist read at offset 505 of length 002: 0.40556788444519043s
activeloop/mnist read at offset 694 of length 004: 0.413848876953125s
activeloop/mnist read at offset 120 of length 008: 0.4096333980560303s
activeloop/mnist read at offset 678 of length 016: 0.41056180000305176s
activeloop/mnist read at offset 191 of length 032: 0.40457701683044434s
activeloop/mnist read at offset 819 of length 064: 0.4001600742340088s
activeloop/mnist read at offset 006 of length 128: 0.4083399772644043s
activeloop/mnist read at offset 672 of length 256: 0.4075479507446289s
activeloop/mnist read at offset 319 of length 512: 0.4023756980895996s

activeloop/omniglot_test read at offset 361 of length 001: 0.18370699882507324s
activeloop/omniglot_test read at offset 193 of length 002: 0.14581060409545898s
activeloop/omniglot_test read at offset 245 of length 004: 0.15089797973632812s
activeloop/omniglot_test read at offset 276 of length 008: 0.13914799690246582s
activeloop/omniglot_test read at offset 937 of length 016: 0.16128158569335938s
activeloop/omniglot_test read at offset 668 of length 032: 0.14351749420166016s
activeloop/omniglot_test read at offset 081 of length 064: 0.16651391983032227s
activeloop/omniglot_test read at offset 453 of length 128: 0.23402667045593262s
activeloop/omniglot_test read at offset 916 of length 256: 0.2857668399810791s
activeloop/omniglot_test read at offset 232 of length 512: 0.24763941764831543s

activeloop/cifar10_train read at offset 636 of length 001: 0.8322362899780273s
activeloop/cifar10_train read at offset 106 of length 002: 0.9116883277893066s
activeloop/cifar10_train read at offset 175 of length 004: 0.7623577117919922s
activeloop/cifar10_train read at offset 082 of length 008: 0.7663559913635254s
activeloop/cifar10_train read at offset 073 of length 016: 0.7576076984405518s
activeloop/cifar10_train read at offset 833 of length 032: 0.7388653755187988s
activeloop/cifar10_train read at offset 192 of length 064: 0.7494456768035889s
activeloop/cifar10_train read at offset 425 of length 128: 0.7731473445892334s
activeloop/cifar10_train read at offset 844 of length 256: 0.755255937576294s
activeloop/cifar10_train read at offset 332 of length 512: 0.7510733604431152s

activeloop/cifar100_train read at offset 707 of length 001: 0.8900413513183594s
activeloop/cifar100_train read at offset 493 of length 002: 0.7479977607727051s
activeloop/cifar100_train read at offset 944 of length 004: 0.7581956386566162s
activeloop/cifar100_train read at offset 468 of length 008: 0.7560327053070068s
activeloop/cifar100_train read at offset 994 of length 016: 0.7357666492462158s
activeloop/cifar100_train read at offset 467 of length 032: 0.7644472122192383s
activeloop/cifar100_train read at offset 482 of length 064: 0.7389593124389648s
activeloop/cifar100_train read at offset 742 of length 128: 0.7508723735809326s
activeloop/cifar100_train read at offset 691 of length 256: 0.7472829818725586s
activeloop/cifar100_train read at offset 636 of length 512: 0.7655932903289795s
```

#### Observations
Hub performs relatively uniformly over the various batch sizes with the notable exception of Omniglot test dataset. It can be speculated that a few times lower number of images in the dataset compared to others allow Hub to perform much better than in the case of other datasets. Reading single element batches is slower than of batches containing multiple elements. 

### Dataset Iteration

We measure the time to iterate over a full dataset in both pytorch and tensorflow (separately).
Benchmarks also vary over multiple preset batch sizes and prefetch factors.

#### Results

```
activeloop/mnist PyTorch prefetch 001 in batches of 001: 114.81040406227112s
activeloop/mnist TF prefetch 001 in batches of 001: 26.655298948287964s
activeloop/mnist PyTorch prefetch 004 in batches of 001: 93.09556984901428s
activeloop/mnist TF prefetch 004 in batches of 001: 20.980616569519043s
activeloop/mnist PyTorch prefetch 016 in batches of 001: 96.3225359916687s
activeloop/mnist TF prefetch 016 in batches of 001: 20.642109632492065s
activeloop/mnist PyTorch prefetch 128 in batches of 001: 100.28293037414551s
activeloop/mnist TF prefetch 128 in batches of 001: 23.141404628753662s
activeloop/mnist PyTorch prefetch 001 in batches of 016: 14.027148246765137s
activeloop/mnist TF prefetch 001 in batches of 016: 11.463196039199829s
activeloop/mnist PyTorch prefetch 004 in batches of 016: 12.892241477966309s
activeloop/mnist TF prefetch 004 in batches of 016: 11.235910654067993s
activeloop/mnist PyTorch prefetch 016 in batches of 016: 12.55230164527893s
activeloop/mnist TF prefetch 016 in batches of 016: 10.931262016296387s
activeloop/mnist PyTorch prefetch 128 in batches of 016: 12.502347946166992s
activeloop/mnist TF prefetch 128 in batches of 016: 11.023484230041504s
activeloop/mnist PyTorch prefetch 001 in batches of 128: 8.963737726211548s
activeloop/mnist TF prefetch 001 in batches of 128: 9.750889301300049s
activeloop/mnist PyTorch prefetch 004 in batches of 128: 8.980962991714478s
activeloop/mnist TF prefetch 004 in batches of 128: 9.708252429962158s
activeloop/mnist PyTorch prefetch 016 in batches of 128: 9.048558235168457s
activeloop/mnist TF prefetch 016 in batches of 128: 10.368900299072266s
activeloop/mnist PyTorch prefetch 128 in batches of 128: 8.343258380889893s
activeloop/mnist TF prefetch 128 in batches of 128: 10.840083122253418s
```

#### Observations
Increasing the batch size leads to a better performance. The transition from the size of 1 to 16 leads to a decrease in iteration time by over 85%. Tensorflow's performance seems not to be drastically improved by prefetching. For PyTorch, however, in smaller batches, an appropriate prefetch factor can elicit a 5-20% improvement. For both Tensorflow and PyTorch a relatively optimal balance is achieved at the prefetch factor equal to 4 and the batch size of length 16. These parameters are used in the external dataset iteration section described below.


## External Benchmarks

### Read and Write Sequential Access

*How does Hub compare to zarr and tiledb in terms of read / write sequential access to the dataset?*

Remote Hub already performs \~1.14x better than TileDB (which offers local storage only) whereas Hub used locally **is over 26x better** than TileDB on the access to the entire dataset. The results are even more explicit in batched access.

Note: Writing tests are awaiting.

#### Results
The datasets for TileDB and zarr are stored locally.

MNIST (entire dataset)
| Framework| Read | Write |
| --- | --- | --- |
| TileDB | 1.3106651306152344s | |
| zarr | 0.3550150394439697s | |
| Hub (remote - Wasabi) | 1.1537418365478516s | |
| Hub (local) | 0.0483090877532959s | |

MNIST (in batches of 7000)
| Framework| Read | Write |
| --- | --- | --- |
| TileDB | 12.647251844406128s | |
| zarr | 0.34612417221069336s | |
| Hub (remote - Wasabi) | 1.0862374305725098s | |
| Hub (local) | 0.12435555458068848s | |


#### Observations
Hub performs better than zarr despite being based on the framework. TileDB is an outlier among all frameworks.

Remote access to Hub is 8-24x times slower than local.

### Dataset Iteration

*Is Hub faster in iterating over a dataset than PyTorch DataLoader and Tensorflow Dataset?*

**Yes, Hub fetching data remotely outperforms both Pytorch and Tensorflow on MNIST dataset.**
It is 1.12x better than PyTorch and 1.004x better than Tensorflow.

#### Parameters
1. Datasets: MNIST & Places365
2. Batch size: 16
3. Prefetch factor: 4
4. Number of workers: 1

#### Results
| Loader | MNIST | Places365 |
| --- | --- | --- |
| Hub (remote - Wasabi) `.to_pytorch()` | 12.460094690322876s | 6033.249919652939s |
| Hub (remote - S3) `.to_pytorch()` | 8.437131643295288s | 4590.98117518425s |
| Hub (local) `.to_pytorch()` | 353.3983402252197s | *timed out* |
| PyTorch (local, native) | 13.931219339370728s | 4305.066396951675s |
| Hub (remote - Wasabi) `.to_tensorflow()` | 10.866756200790405s | 5725.523037910461s |
| Hub (remote - S3) `.to_tensorflow()` | 11.888715505599976s | 4524.5225484371185s |
| Hub (local) `.to_tensorflow()` | 11.07367753982544s | 2141.250002384186s |
| Tensorflow (local, native - TFDS) | 10.913283348083496s | 1051.0043654441833s |

#### Observations

Except for the relatively slow performance of Hub's `to_pytorch` in the local environment, the results of all loaders on MNIST are comparable.

Places365, a significantly larger dataset, sheds light on the real differences among the frameworks. Not surprisingly, local storage surpasses the remote ones - S3 followed by Wasabi, heavily affected by the network latency. The best performing framework turns out to be Tensorflow, closely followed by Hub's `to_tensorflow` implementation. The biggest outlier is Hub's local `to_pytorch` which could not be measured on time as it is over 10x slower than other loaders.

PyTorch's native `DataLoader` as well as Hub's `to_pytorch` function are generally slower than Tensorflow.

## Limitations
*This section is incomplete.*

## Conclusions
Hub team needs to continue improving `to_pytorch` and `to_tensorflow` functions. Benchmarks should be re-calculated every time new features are added to Hub. Further plans with regards to the benchmarks are outlined [here](https://github.com/activeloopai/Hub/issues/529).
