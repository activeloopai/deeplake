import time
import hub
from hub import tensor, dataset
import numpy as np

hub.local_mode()

images = tensor.from_array(np.zeros((4, 512, 512)))
labels = tensor.from_array(np.zeros((4, 512, 512)))
ds = dataset.from_tensors({"images": images, "labels": labels})
ds.store("basic2")

# ds = hub.load("davit/basic2")
print(ds["images"][0].compute())
