import numpy as np
import os
import pandas as pd
from collections import defaultdict


def get_sample_counts(dataset, class_names):
    """
    Get total and class-wise positive sample count of a dataset

    Arguments:
    output_dir - str, folder of dataset.csv
    dataset - str, train|dev|test
    class_names - list of str, target classes

    Returns:
    total_count - int
    class_positive_counts - dict of int, ex: {"Effusion": 300, "Infiltration": 500 ...}
    """
    labels = dataset["label_chexpert"].compute()
    class_positive_counts = defaultdict(int)
    for label in labels:
        for i, val in enumerate(label):
            if val == 1:
                class_positive_counts[class_names[i]] += 1
    return len(dataset), class_positive_counts
    # df = pd.read_csv(os.path.join(output_dir, f"{dataset}.csv"))
    # total_count = df.shape[0]
    # labels = df[class_names].as_matrix()
    # positive_counts = np.sum(labels, axis=0)
    # class_positive_counts = dict(zip(class_names, positive_counts))
    # return total_count, class_positive_counts
