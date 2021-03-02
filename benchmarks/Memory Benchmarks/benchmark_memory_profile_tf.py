"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import time
import platform
from memory_profiler import profile

from tensorflow.keras.datasets import cifar100

start_time = time.time()

# Files
fp = open("cifar100_mem_prof_tf.txt", "w+")

fp.write(f"Platform: {platform.system()} 10\nVersion: {platform.version()}\nArchitecture: {platform.machine()}\n"
         f"Processor: {platform.processor()}\nRAM: 16GB\nPython Version: 3.8.0")

@profile(stream=fp)
def cifar100_tf_profile():
    ds_train, ds_test = cifar100.load_data()

if __name__ == "__main__":
    cifar100_tf_profile()

    fp.write(f"\n Time taken for execution: {round(time.time() - start_time, 3)}")
    fp.close()