from hub.core.dataset import Dataset
import numpy as np


def is_dataset(dataset):
    return isinstance(dataset, Dataset)


def get_num_classes(labels):
    num_classes = len(np.unique(labels))
    return num_classes


def get_labels(dataset, labels_tensor):
    labels = dataset[labels_tensor].numpy().flatten()
    return labels
