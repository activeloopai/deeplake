"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import time
import platform
from memory_profiler import profile

import hub
from hub import Dataset
from hub.utils import Timer

start_time = time.time()

# Variables
cache_var = [True, False]

# Files
fp = open("cifar100_mem_prof_hub_cache.txt", "w+")

fp.write(f"Platform: {platform.system()} 10\nVersion: {platform.version()}\nArchitecture: {platform.machine()}\n"
         f"Processor: {platform.processor()}\nRAM: 16GB\nPython Version: 3.8.0 ")

@profile(stream=fp)
def cifar100_profile_cache(cache_var=False):
    ds = Dataset("activeloop/cifar100_train", cache=cache_var)

if __name__ == "__main__":
    for cache in cache_var:
        cifar100_profile_cache(cache)

    fp.write(f"\n Time taken for execution: {round(time.time() - start_time, 3)}")
    fp.close()