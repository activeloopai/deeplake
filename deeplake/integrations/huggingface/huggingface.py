import pathlib
from typing import Union, Set, Any
from deeplake.core.dataset import Dataset
import posixpath
import deeplake
from tqdm import tqdm  # type: ignore


def _is_seq_convertible(seq):
    from datasets import Sequence

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


def _create_tensor_from_feature(
    key: str, feature: Any, src: Any, ds: Dataset, exist_ok: bool = False
):
    from datasets import Sequence, ClassLabel
    from datasets import Dataset as hfDataset

    curr = posixpath.split(key)[-1]
    if isinstance(feature, (dict, Sequence, list)):
        if isinstance(feature, dict):
            for x in feature:
                _create_tensor_from_feature(
                    f"{key}/{x}", feature[x], src[curr], ds, exist_ok=exist_ok
                )
            return True
        elif isinstance(feature, (Sequence, list)):
            if isinstance(feature, Sequence):
                inner = feature.feature
            else:
                inner = feature[0]
            if isinstance(inner, dict):
                _create_tensor_from_feature(
                    key, feature.feature, src, ds, exist_ok=exist_ok
                )
                return True
            elif _is_seq_convertible(feature):
                ds.create_tensor(key, exist_ok=exist_ok)
            else:
                return False
    elif feature.dtype in ("string", "large_string"):
        ds.create_tensor(key, htype="text", exist_ok=exist_ok)
    elif isinstance(feature, ClassLabel):
        ds.create_tensor(
            key, htype="class_label", class_names=feature.names, exist_ok=exist_ok
        )
    else:
        ds.create_tensor(key, exist_ok=exist_ok)

    if isinstance(src, (hfDataset, dict)):
        ds[key].extend(src[curr])
    else:
        values = []
        for i in range(len(src)):
            values.extend(src[i][curr])
        ds[key].extend(values)
    return True


def ingest_huggingface(
    src: Any,
    dest: Union[Dataset, str, pathlib.Path],
    use_progressbar: bool = True,
    exist_ok: bool = False,
) -> Dataset:
    """Converts Hugging Face datasets to Deep Lake format.

    Args:
        src (hfDataset, DatasetDict): Hugging Face Dataset or DatasetDict to be converted. Data in different splits of a
            DatasetDict will be stored under respective tensor groups.
        dest (Dataset, str, pathlib.Path): Destination dataset or path to it.
        use_progressbar (bool): Defines if progress bar should be used to show conversion progress.

    Returns:
        Dataset: The destination Deep Lake dataset.

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
    from datasets import DatasetDict

    if isinstance(dest, (str, pathlib.Path)):
        ds = deeplake.dataset(dest)
    else:
        ds = dest  # type: ignore

    if isinstance(src, DatasetDict):
        for split, src_ds in src.items():
            for key, feature in src_ds.features.items():
                _create_tensor_from_feature(
                    f"{split}/{key}", feature, src_ds, ds, exist_ok=exist_ok
                )
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
