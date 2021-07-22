from typing import Union

import tensorflow  # type: ignore
import tensorflow_datasets as tfds  # type: ignore
from tensorflow_datasets import Split
from tqdm import tqdm  # type: ignore

from hub.core.dataset import Dataset
import hub


def from_tfds_to_path(
    tfds_dataset_name: str,
    split: Union[str, Split],
    hub_ds_path: str,
    batch_size: int = 100,
):
    """Converts the tfds dataset with name `tfds_dataset_name` into a hub dataset and saves it at `hub_ds_path`
    Args:
        tfds_dataset_name (str): Name of tfds dataset.You can see a list of all tfds datasets here:
            https://www.tensorflow.org/datasets/catalog/overview
        split (str, Split) : Used for dataset splits as defined here: https://www.tensorflow.org/datasets/splits
        hub_ds_path (str): Path where new hub dataset will be created
        batch_size (int): Batch size for tfds dataset. Has no effect on output, but may affect performance.
    Returns:
        A hub dataset
    """
    tfds_ds = tfds.load(tfds_dataset_name, split=split).batch(batch_size)
    ds = hub.dataset(hub_ds_path)

    return from_tfds(tfds_ds=tfds_ds, ds=ds)  # type: ignore


def from_tfds(tfds_ds: tensorflow.data.Dataset, ds: Dataset):
    """Converts a tfds dataset to hub dataset
    Args:
        tfds_ds (tensorflow.data.Dataset): A tfds_dataset object.
        ds (Dataset) : A Hub dataset object where Tensor will be created.
    Returns:
        A hub dataset
    """
    tfds_numpy = tfds.as_numpy(tfds_ds)  # Convert `tf.data.Dataset` to Python generator

    for sample in tqdm(tfds_numpy):
        for col in sample:
            if col not in ds.tensors:
                ds.create_tensor(col)
            ds[col].extend(sample[col])
    return ds
