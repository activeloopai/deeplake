import hub
import numpy as np

hub.local_mode()

images = hub.tensor.from_array(np.zeros((4, 512, 512)))
labels = hub.tensor.from_array(np.zeros((4, 512, 512)))

# ds = hub.dataset.from_tensors({"images": images, "labels": labels})
# ds.store("davit2/basic2")

ds = hub.load("davit2/basic")
