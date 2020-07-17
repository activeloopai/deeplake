import hub
import numpy as np

images = hub.tensor.from_array(np.zeros((4, 512, 512)))
labels = hub.tensor.from_array(np.zeros((4, 512, 512)))

# ds = hub.dataset.from_tensors({"images": images, "labels": labels})
# ds.store("basic")

ds = hub.load("test/test3")
print(ds["images"][0].compute())
