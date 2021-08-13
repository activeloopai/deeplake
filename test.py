import glob
import os

import numpy as np
from PIL import Image
import sys, getopt

import hub
from hub.api.dataset import dataset
from hub.core.dataset import Dataset


if __name__ == "__main__":

    hub_dataset_path = "~/projects/test/data/hub_dataset"
    # dataset_folders = glob.glob('./data/medium-dataset-1/seg_train/seg_train/*') #Paths to source data
    dataset_folders = glob.glob(
        "/home/rahulbs/projects/test/data/small-dataset-1/*"
    )  # Paths to source data

    # Initialize dataset
    ds = hub.dataset(hub_dataset_path)

    print("Dataset 1:")
    with ds:
        ds.create_tensor(
            "images", htype="image", sample_compression="jpeg", hash_samples=True
        )

        for label, folder_path in enumerate(dataset_folders):
            paths = glob.glob(os.path.join(folder_path, "*"))  # Get subfolders

            # Iterate through images in the subfolders
            for path in paths:
                print("Path: ", path)
                ds.images.append(
                    hub.read(path)
                )  # Append to images tensor using hub.load

    hub_dataset_path_2 = "./data/hub_dataset_2"
    # dataset_folders_2 = glob.glob('./data/medium-dataset-2/seg_train/seg_train/*') #Paths to source data
    dataset_folders_2 = glob.glob("./data/small-dataset-2/*")  # Paths to source data

    # print('Dataset 2:')
    # # Initialize dataset
    # ds_2 = hub.dataset(hub_dataset_path_2)

    # with ds_2:
    #     ds_2.create_tensor('images', htype = 'image', sample_compression='jpeg', hash_samples=True)

    #     for label, folder_path in enumerate(dataset_folders_2):
    #         paths = glob.glob(os.path.join(folder_path, '*')) # Get subfolders

    #         # Iterate through images in the subfolders
    #         for path in paths:
    #             print("Path: ", path)
    #             ds_2.images.append(hub.read(path))  # Append to images tensor using hub.load

    # hub.compare(ds, ds_2)


# if __name__ == '__main__':

#     hub_dataset_path = '~/projects/hash-dataset/data/hub_dataset'
#     hub_dataset_path2 = '~/projects/hash-dataset/data/hub_dataset2'
#     dataset_folders = glob.glob('../hash-dataset/data/small-dataset-1/*') #Paths to source data
#     dataset_folders2 = glob.glob('../hash-dataset/data/small-dataset-2/*')

#     # Initialize dataset
#     ds = hub.dataset(hub_dataset_path)
#     ds2 = hub.dataset(hub_dataset_path2)

#     print("\nDATASET1")
#     with ds:
#         ds.create_tensor('images', htype = 'image', sample_compression='jpeg', hash_samples=True)
#         ds.create_tensor('labels', htype = 'class_label')

#         for label, folder_path in enumerate(dataset_folders):
#             paths = glob.glob(os.path.join(folder_path, '*')) # Get subfolders

#             # Iterate through images in the subfolders
#             for path in paths:
#                 print('Path: ', path)
#                 ds.images.append(hub.read(path))  # Append to images tensor using hub.load
#                 ds.labels.append(np.uint32(label)) # Append to labels tensor

#     print("\nDATASET2")
#     with ds2:
#         ds2.create_tensor('images', htype = 'image', sample_compression = 'jpeg', hash_samples
#         =True)
#         ds2.create_tensor('labels', htype = 'class_label')

#         for label, folder_path in enumerate(dataset_folders2):
#             paths = glob.glob(os.path.join(folder_path, '*')) # Get subfolders

#             # Iterate through images in the subfolders
#             for path in paths:
#                 print('Path: ', path)
#                 ds2.images.append(hub.read(path))  # Append to images tensor using hub.load
#                 ds2.labels.append(np.uint32(label)) # Append to labels tensor


#     hub.compare(ds, ds2)
