import argparse
import os
import struct
from array import array as pyarray

import numpy as np
import cv2
import hub
from hub.collections import dataset, tensor



def transform_img(img):
    img = cv2.imread(img)
    return cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)


def load_pascal_segmentation(path):

    # get train path
    fname_img_train = os.path.join(path, "train/Images")
    fname_lbl_train = os.path.join(path, "train/SegmentationLabels")
    # get test path
    fname_img_test = os.path.join(path, "test/Images")
    fname_lbl_test = os.path.join(path, "test/SegmentationLabels")

    # In data we can see that not every image has label, so we are taking images which has label
    train_imgs = sorted(
        [
            os.path.join(fname_img_train, f.replace('png','jpg')) for f in os.listdir(fname_lbl_train)
        ]
    )
    test_imgs = sorted(
        [
            os.path.join(fname_img_test, f.replace('png','jpg')) for f in os.listdir(fname_lbl_test)
        ]
    )
    
    train_lbls = sorted(
        [
            os.path.join(fname_lbl_train, f) for f in os.listdir(fname_lbl_train)
        ]
    )
    test_lbls = sorted(
        [
            os.path.join(fname_lbl_test, f) for f in os.listdir(fname_lbl_test)
        ]
    )
    
    train_imgs = [transform_img(img) for img in train_imgs]
    test_imgs = [transform_img(img) for img in test_imgs]
    
    train_lbls = [transform_img(img) for img in train_lbls]
    test_lbls = [transform_img(img) for img in test_lbls]
    
    train = {"imgs": train_imgs, "lbls": train_lbls}
    test = {"imgs": test_imgs, "lbls": test_lbls}
    
    return train, test     



def main():

    train, test = load_pascal_segmentation(path)
    
    train_images = np.concatenate([img for img in train["imgs"]])
    train_labels = np.concatenate([lbl for lbl in train["lbls"]])
    
    test_images = np.concatenate([img for img in test["imgs"]])
    test_labels = np.concatenate([lbl for lbl in test["lbls"]])

    
    train_images = tensor.from_array(train_images, dtag="imgs")
    train_labels = tensor.from_array(train_labels, dtag="lbls")
    
    test_images = tensor.from_array(test_images, dtag="imgs")
    test_labels = tensor.from_array(test_labels, dtag="lbls")

    train_ds = dataset.from_tensors({"data": train_images, "labels": train_labels})
    test_ds = dataset.from_tensors({"data": test_images, "labels": test_labels})
    
    train_ds.store("arenbeglaryan/vocsegmentation")


if __name__ == "__main__":
    main()                

