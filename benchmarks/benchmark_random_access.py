from random import randint
from itertools import chain

from hub import Dataset
from hub.utils import Timer

DATASET_NAMES = ['activeloop/mnist', 
                 'activeloop/omniglot_test',
                 'activeloop/cifar10_train', 
                 'activeloop/cifar100_train']

SPAN_POWER_MAX = 10

def time_random_access(dataset_name="activeloop/mnist", offset=1000, span=1000, field="image"):
    dset = Dataset(dataset_name, cache=False, storage_cache=False)
    with Timer(f"{dataset_name} read at offset {offset:03} of length {span:03}"):
        dset[field][offset:offset+span].compute()

if __name__ == "__main__":
    for name in DATASET_NAMES:
        for span in range(SPAN_POWER_MAX):
            offset = randint(0,999)
            time_random_access(name, offset, 2**span)
        print()
