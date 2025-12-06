---
seo_title: "Async Data Loader | Speed Up Multi-Modal Data Ingestion"
description: "Use The Async Data Loader To Parallelize Data Ingestion, Enhancing Speed And Efficiency When Loading Multi-Modal Data For ML Model Training or Fast AI Search."
---

# Async Data Loader

## Overview

This document describes the implementation of a custom DataLoader for handling data retrieval using `deeplake.Dataset` with `PyTorch`. The DataLoader supports both sequential and asynchronous data fetching, with the asynchronous approach being optimized for performance and speed.

## Dataset Structure

The dataset comprises pairs of images and their respective masks. Each image is a high-resolution file, while each mask is a binary image indicating the regions of interest within the corresponding image.

## Sequential data fetching

This ImageDataset class is a custom implementation of a PyTorch dataset that uses `deeplake.Dataset` as a datasource.

<!-- test-context
```python
import numpy as np
import torch
import deeplake
from deeplake import types
from typing import Callable

ds = deeplake.create("tmp://")
```
-->

```python
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, deeplake_ds: deeplake.Dataset, transform: Callable = None):
        self.ds = deeplake_ds
        self.transform = transform
    
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, item):
        image =  self.ds[item]["images"]
        mask = self.ds[item]["masks"]

        if self.transform is not None:
            image, mask = self.transform((image, mask))

        return image, mask
    
```

In the sequential fetching approach, data is loaded one item at a time in a synchronous manner. While this method is straightforward, it can become a bottleneck when working with large datasets with multiple tensors.

## Asynchronous Data Fetching

The asynchronous fetching method utilizes asyncio and threading to load data in parallel. This significantly improves loading times, especially for large datasets with multiple tensors.

```python
import deeplake

import asyncio
from threading import Thread, Lock
from multiprocessing import Queue

lock = Lock()
index = -1
def generate_data(ds: deeplake.Dataset):
    total_count = len(ds)
    global index
    while True:
        idx = 0
        with lock:
            index = (index + 1) % (total_count - 1)
            idx = index
        yield ds[idx]

class AsyncImageDataset(torch.utils.data.IterableDataset):
    def __init__(self, deeplake_ds: deeplake.Dataset, transform: Callable = None, max_queue_size: int = 1024):
        self.ds = deeplake_ds
        self.transform = transform
        self.worked_started = False
        self.data_generator = generate_data(self.ds)
        self.q = Queue(maxsize=max_queue_size)
    
    async def run_async(self):
        for item in self.data_generator:
            data = await asyncio.gather(
                item.get_async("images"),
                item.get_async("masks")
            )
            self.q.put(data)

    def start_worker(self):
        loop = asyncio.new_event_loop()

        for _ in range(128):
            loop.create_task(self.run_async())

        def loop_in_thread(loop):
            asyncio.set_event_loop(loop)
            loop.run_forever()

        self.loop_thread = Thread(target=loop_in_thread, args=(loop,), daemon=True)
        self.loop_thread.start()

        self.worked_started = True

    def __iter__(self):
        while True:
            if not self.worked_started:
                self.start_worker()

            # wait until some data is filled
            while self.q.empty():
                pass

            image, mask = self.q.get()
            if self.transform is not None:
                image, mask = self.transform((image, mask))

            yield image, mask
```

The `AsyncImageDataset` utilizes Python’s `asyncio` library to fetch images and masks concurrently from `deeplake.Dataset`, minimizing data loading times. The class implements a separate thread to run an event loop, allowing multiple data retrieval tasks to operate simultaneously. A multiprocessing `Queue` is used to store retrieved items, enabling thread-safe communication between data loading threads and the main processing loop.

## Benchmark results

| Method            | Average Loading Time (seconds per batch) |
| --------          |                               -------:    |
| Sequential        |                               6.2         |
| Asynchronous      |                               2.15        |

While the sequential method is simpler to implement, the asynchronous approach offers substantial performance benefits, making it the preferred choice for handling larger datasets in machine learning workflows. This flexibility allows users to choose the best method suited to their specific use case, ensuring efficient data handling and model training.
