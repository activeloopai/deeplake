import hub
from PIL import Image
import numpy as np
import os

ds = hub.empty("./png")  # Creates the dataset

# Find the class_names and list of files that need to be uploaded
dataset_folder = "dataset/png"

class_names = os.listdir(dataset_folder)

files_list = []
for dirpath, dirnames, filenames in os.walk(dataset_folder):
    for filename in filenames:
        files_list.append(os.path.join(dirpath, filename))

with ds:
    # Create the tensors with names of your choice.
    ds.create_tensor("images", htype="image", sample_compression="png")
    ds.create_tensor("labels", htype="class_label", class_names=class_names)

    # # Add arbitrary metadata - Optional
    # ds.info.update(description = 'My first Hub dataset')
    # ds.images.info.update(camera_type = 'SLR')

with ds:
    # Iterate through the files and append to hub dataset
    for file in files_list:
        label_text = os.path.basename(os.path.dirname(file))
        label_num = class_names.index(label_text)

        ds.images.append(hub.read(file))  # Append to images tensor using hub.read
        ds.labels.append(np.uint32(label_num))  # Append to labels tensor
