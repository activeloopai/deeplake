# Benchmarking

## Image Compression

We measure the time to compress (PNG) a [sample image](images/compression_benchmark_image.png) using PIL and Hub.

### Results

The results below measure compression time of the sample image at a batch size of 100.

```
PIL compression: 16.31s
Hub compression: 16.19s
```

## Random Access

We measure the time to fetch an uncached random sample from a dataset,
varying over several standard datasets and further at several batch sizes.
Random offsets are also used to ensure that no caching is being taken advantage of externally.

### Results

```
activeloop/mnist read at offset 101 of length 001: 1.625s
activeloop/mnist read at offset 854 of length 002: 0.913s
activeloop/mnist read at offset 633 of length 004: 1.765s
activeloop/mnist read at offset 252 of length 008: 1.572s
activeloop/mnist read at offset 186 of length 016: 1.600s
activeloop/mnist read at offset 737 of length 032: 1.288s
activeloop/mnist read at offset 466 of length 064: 1.063s
activeloop/mnist read at offset 989 of length 128: 0.848s
activeloop/mnist read at offset 907 of length 256: 2.404s
activeloop/mnist read at offset 157 of length 512: 0.757s

activeloop/omniglot_test read at offset 622 of length 001: 0.837s
activeloop/omniglot_test read at offset 821 of length 002: 0.351s
activeloop/omniglot_test read at offset 304 of length 004: 0.480s
activeloop/omniglot_test read at offset 137 of length 008: 0.317s
activeloop/omniglot_test read at offset 768 of length 016: 0.323s
activeloop/omniglot_test read at offset 772 of length 032: 0.479s
activeloop/omniglot_test read at offset 639 of length 064: 0.322s
activeloop/omniglot_test read at offset 051 of length 128: 0.318s
activeloop/omniglot_test read at offset 993 of length 256: 0.634s
activeloop/omniglot_test read at offset 353 of length 512: 0.734s

activeloop/cifar10_train read at offset 393 of length 001: 3.378s
activeloop/cifar10_train read at offset 131 of length 002: 2.970s
activeloop/cifar10_train read at offset 932 of length 004: 2.816s
activeloop/cifar10_train read at offset 218 of length 008: 3.511s
activeloop/cifar10_train read at offset 414 of length 016: 3.186s
activeloop/cifar10_train read at offset 231 of length 032: 1.892s
activeloop/cifar10_train read at offset 529 of length 064: 3.045s
activeloop/cifar10_train read at offset 304 of length 128: 2.513s
activeloop/cifar10_train read at offset 815 of length 256: 3.073s
activeloop/cifar10_train read at offset 012 of length 512: 1.811s

activeloop/cifar100_train read at offset 830 of length 001: 2.886s
activeloop/cifar100_train read at offset 609 of length 002: 3.076s
activeloop/cifar100_train read at offset 036 of length 004: 2.939s
activeloop/cifar100_train read at offset 316 of length 008: 2.631s
activeloop/cifar100_train read at offset 161 of length 016: 1.709s
activeloop/cifar100_train read at offset 159 of length 032: 3.489s
activeloop/cifar100_train read at offset 132 of length 064: 2.834s
activeloop/cifar100_train read at offset 224 of length 128: 2.045s
activeloop/cifar100_train read at offset 061 of length 256: 2.169s
activeloop/cifar100_train read at offset 940 of length 512: 2.186s
```

## Dataset Iteration

We measure the time to iterate over a full dataset in both pytorch and tensorflow (separately).
Benchmarks also vary over multiple preset batch sizes and prefetch factors.

### Results

Results are blocked by `fixes/to_tensorflow` issue. Coming soon.
