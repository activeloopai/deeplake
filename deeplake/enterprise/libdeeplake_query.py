from deeplake.enterprise.convert_to_libdeeplake import dataset_to_libdeeplake
from deeplake.core.dataset.deeplake_query_dataset import DeepLakeQueryDataset
from typing import Optional, Union

import numpy as np


def query(dataset, query_string: str):
    """Returns a sliced :class:`~deeplake.core.dataset.Dataset` with given query results.

    It allows to run SQL like queries on dataset and extract results. See supported keywords and the Tensor Query Language documentation
    :ref:`here <tql>`.


    Args:
        dataset: deeplake.Dataset object on which the query needs to be run
        query_string (str): An SQL string adjusted with new functionalities to run on the given :class:`deeplake.Dataset` object


    Returns:
        Dataset: A deeplake.Dataset object.

    Examples:

        Query from dataset all the samples with lables other than ``5``

        >>> import deeplake
        >>> from deeplake.enterprise import query
        >>> ds = deeplake.load('hub://activeloop/fashion-mnist-train')
        >>> query_ds_train = query(ds_train, "select * where labels != 5")

        Query from dataset first appeard ``1000`` samples where the ``categories`` is ``car`` and ``1000`` samples where the ``categories`` is ``motorcycle``

        >>> ds_train = deeplake.load('hub://activeloop/coco-train')
        >>> query_ds_train = query(ds_train, "(select * where contains(categories, 'car') limit 1000) union (select * where contains(categories, 'motorcycle') limit 1000)")
    """
    if isinstance(dataset, DeepLakeQueryDataset):
        ds = dataset.indra_ds
    elif dataset.libdeeplake_dataset is not None:
        ds = dataset.libdeeplake_dataset
        slice_ = dataset.index.values[0].value
        if slice_ != slice(None):
            if isinstance(slice_, tuple):
                slice_ = list(slice_)
            ds = ds[slice_]
    else:
        ds = dataset_to_libdeeplake(dataset)
    dsv = ds.query(query_string)
    from deeplake.enterprise.convert_to_libdeeplake import INDRA_API

    if not isinstance(dataset, DeepLakeQueryDataset) and INDRA_API.tql.parse(query_string).is_filter:  # type: ignore
        indexes = dsv.indexes
        return dataset.no_view_dataset[indexes]
    else:
        view = DeepLakeQueryDataset(deeplake_ds=dataset, indra_ds=dsv)
        view._tql_query = query_string
        if hasattr(dataset, "is_actually_cloud"):
            view.is_actually_cloud = dataset.is_actually_cloud
        return view


def sample_by(
    dataset,
    weights: Union[str, list, tuple, np.ndarray],
    replace: Optional[bool] = True,
    size: Optional[int] = None,
):
    """Returns a sliced :class:`~deeplake.core.dataset.Dataset` with given sampler applied.


    Args:
        dataset: deeplake.Dataset object on which the query needs to be run
        weights: (Union[str, list, tuple, np.ndarray]): If it's string then tql will be run to calculate the weights based on the expression. list, tuple and ndarray will be treated as the list of the weights per sample
        replace: Optional[bool] If true the samples can be repeated in the result view.
            (default: ``True``).
        size: Optional[int] The length of the result view.
            (default: ``len(dataset)``)


    Returns:
        Dataset: A deeplake.Dataset object.

    Raises:
        ValueError: When the given np.ndarray is multidimensional

    Examples:

        Sample the dataset with ``labels == 5`` twice more than ``labels == 6``

        >>> import deeplake
        >>> from deeplake.experimental import query
        >>> ds = deeplake.load('hub://activeloop/fashion-mnist-train')
        >>> sampled_ds = sample_by(ds, "max_weight(labels == 5: 10, labels == 6: 5)")

        Sample the dataset treating `labels` tensor as weights.

        >>> import deeplake
        >>> from deeplake.experimental import query
        >>> ds = deeplake.load('hub://activeloop/fashion-mnist-train')
        >>> sampled_ds = sample_by(ds, "labels")

        Sample the dataset with the given weights;

        >>> ds = deeplake.load('hub://activeloop/coco-train')
        >>> weights = list()
        >>> for i in range(0, len(ds)):
        >>>     weights.append(i % 5)
        >>> sampled_ds = sample_by(ds, weights, replace=False)
    """
    if isinstance(weights, np.ndarray):
        if len(weights.shape) != 1:
            raise ValueError("weights should be 1 dimensional array.")
        weights = tuple(weights)
    ds = dataset_to_libdeeplake(dataset)
    if size is None:
        dsv = ds.sample(weights, replace=replace)
    else:
        dsv = ds.sample(weights, replace=replace, size=size)
    indexes = dsv.indexes
    return dataset.no_view_dataset[indexes]
