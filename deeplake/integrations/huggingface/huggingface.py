import pathlib
from typing import Dict, Set
from deeplake.core.dataset import Dataset
import posixpath
import deeplake
from typing import Optional
from tqdm import tqdm
from deeplake.util.bugout_reporter import feature_report_path, deeplake_reporter


def _is_seq_convertible(seq):
    from datasets import Sequence  # type: ignore

    if isinstance(seq, Sequence):
        feature = seq.feature
    else:
        feature = seq[0]
    if isinstance(feature, dict):
        return False
    if feature.dtype in (
        "string",
        "large_string",
    ):  # no support for sequences of strings.
        return False
    if isinstance(feature, Sequence):
        return _is_seq_convertible(feature)
    return True


def _create_tensor_from_feature(key, feature, src, ds):
    from datasets import Sequence, ClassLabel
    from datasets import Dataset as hfDataset

    curr = posixpath.split(key)[-1]
    if isinstance(feature, (dict, Sequence, list)):
        if isinstance(feature, dict):
            for x in feature:
                _create_tensor_from_feature(f"{key}/{x}", feature[x], src[curr], ds)
            return True
        elif isinstance(feature, (Sequence, list)):
            if isinstance(feature, Sequence):
                inner = feature.feature
            else:
                inner = feature[0]
            if isinstance(inner, dict):
                _create_tensor_from_feature(key, feature.feature, src, ds)
                return True
            elif _is_seq_convertible(feature):
                ds.create_tensor(key)
            else:
                return False
    elif feature.dtype in ("string", "large_string"):
        ds.create_tensor(key, htype="text")
    elif isinstance(feature, ClassLabel):
        ds.create_tensor(key, htype="class_label", class_names=feature.names)
    else:
        ds.create_tensor(key)

    if isinstance(src, (hfDataset, dict)):
        ds[key].extend(src[curr])
    else:
        values = []
        for i in range(len(src)):
            values.extend(src[i][curr])
        ds[key].extend(values)
    return True


def ingest_huggingface(
    src,
    dest,
    use_progressbar=True,
    token: Optional[str] = None,
    connect_kwargs: Optional[Dict] = None,
    **dataset_kwargs,
) -> Dataset:
    """Converts Hugging Face datasets to Deep Lake format.

    Args:
        src (hfDataset, DatasetDict): Hugging Face Dataset or DatasetDict to be converted. Data in different splits of a
            DatasetDict will be stored under respective tensor groups.
        dest (Dataset, str, pathlib.Path): Destination dataset or path to it.
        use_progressbar (bool): Defines if progress bar should be used to show conversion progress.
        token (Optional[str]): The token to use for accessing the dataset and/or connecting it to Deep Lake.
        connect_kwargs (Optional[Dict]): If specified, the dataset will be connected to Deep Lake, and connect_kwargs will be passed to :meth:`Dataset.connect <deeplake.core.dataset.Dataset.connect>`.
        **dataset_kwargs: Any arguments passed here will be forwarded to the dataset creator function. See :func:`deeplake.empty`.

    Returns:
        Dataset: The destination Deep Lake dataset.

    Raises:
        ValueError: If ``dest`` is not a path or a Deep Lake :class:`Dataset <deeplake.core.dataset.Dataset>`.

    Note:
        - if DatasetDict looks like:

            >>> {
            ...    train: Dataset({
            ...        features: ['data']
            ...    }),
            ...    validation: Dataset({
            ...        features: ['data']
            ...    }),
            ...    test: Dataset({
            ...        features: ['data']
            ...    }),
            ... }

        it will be converted to a Deep Lake :class:`Dataset` with tensors ``['train/data', 'validation/data', 'test/data']``.

        Features of the type ``Sequence(feature=Value(dtype='string'))`` are not supported. Columns of such type are skipped.

    """

    feature_report_path(
        dest if isinstance(dest, str) else dest.path,
        "ingest_huggingface",
        parameters={},
        token=token,
    )

    from datasets import DatasetDict

    if isinstance(dest, (str, pathlib.Path)):
        ds = deeplake.empty(dest, token=token, **dataset_kwargs)
    elif isinstance(dest, Dataset):
        if dataset_kwargs.get("overwrite"):
            ds = deeplake.empty(dest.path, token=token, **dataset_kwargs)
        else:
            ds = dest  # type: ignore
    else:
        raise ValueError(
            f"Expected `dest` to be a path or deeplake Dataset object, got {type(dest)}."
        )

    if connect_kwargs is not None:
        connect_kwargs["token"] = token or connect_kwargs.get("token")
        ds.connect(**connect_kwargs)

    if isinstance(src, DatasetDict):
        for split, src_ds in src.items():
            for key, feature in src_ds.features.items():
                _create_tensor_from_feature(f"{split}/{key}", feature, src_ds, ds)
    else:
        skipped_keys: Set[str] = set()
        features = tqdm(
            src.features.items(),
            desc=f"Converting...({len(skipped_keys)} skipped)",
            disable=not use_progressbar,
        )
        for key, feature in features:
            if not _create_tensor_from_feature(key, feature, src, ds):
                skipped_keys.add(key)
                features.set_description(f"Converting...({len(skipped_keys)} skipped)")
    return ds  # type: ignore
