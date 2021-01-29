# Benchmarking

## Method

All of the benchmarks were conducted on the same machine unless stated otherwise in a section related to a particular benchmark. The specification of the machine we used for the benchmarks can be found below:

Machine: AWS EC2 m4.10xlarge instance
Region: US-East-2c
Memory: 160 GB
CPU: Intel(R) Xeon(R) CPU E5-2676 v3 @ 2.40GHz
#vCPU: 40
Network: 10 Gb


## Image Compression

We measure the time to compress (PNG) a [sample image](images/compression_benchmark_image.png) using PIL and Hub.

### Results

The results below measure compression time of the sample image at a batch size of 100.

```
PIL compression: 25.102458238601685s
Hub compression: 25.102383852005005s
```

## Random Access

We measure the time to fetch an uncached random sample from a dataset,
varying over several standard datasets and further at several batch sizes.
Random offsets are also used to ensure that no caching is being taken advantage of externally.

### Results

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

## Dataset Iteration

We measure the time to iterate over a full dataset in both pytorch and tensorflow (separately).
Benchmarks also vary over multiple preset batch sizes and prefetch factors.

### Results

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
