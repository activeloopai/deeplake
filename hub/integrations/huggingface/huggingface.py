from typing import Union
from hub.core.dataset import Dataset
from datasets import Dataset as hfDataset
from datasets import ClassLabel, Sequence, DatasetDict, Value
import posixpath
import hub


def _is_seq_convertible(feature: Sequence, dtype=None):
    features = feature.feature
    for feature in features:
        if isinstance(feature, Sequence):
            if not _is_seq_convertible(feature, dtype):
                return False
        elif isinstance(feature, Value):
            if not dtype:
                dtype = feature.dtype
            else:
                if feature.dtype != dtype:
                    return False
        else:
            return False
    return True


def _create_tensor_from_feature(key, feature, src, ds):
    curr = posixpath.split(key)[-1]
    if isinstance(feature, (dict, Sequence)):
        if isinstance(feature, dict):
            for x in feature:
                _create_tensor_from_feature(f"{key}/{x}", feature[x], src[curr], ds)
        if isinstance(feature, Sequence):
            if isinstance(feature.feature, dict):
                for x in feature.feature:
                    _create_tensor_from_feature(
                        f"{key}/{x}", feature.feature[x], src[curr], ds
                    )
            elif _is_seq_convertible(feature):
                ds.create_tensor(key)
        return
    elif feature.dtype == "string":
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
    return


def _from_huggingface_ds(src: hfDataset, ds: Dataset) -> Dataset:
    for key, feature in src.features.items():
        _create_tensor_from_feature(key, feature, src, ds)
    return ds


def from_huggingface(
    src: Union[hfDataset, DatasetDict], dest: Union[Dataset, str]
) -> Dataset:
    if isinstance(dest, str):
        ds = hub.dataset(dest)
    else:
        ds = dest

    if isinstance(src, dict):
        # _from_hugging_face_multiple(src, ds)
        pass
    else:
        _from_huggingface_ds(src, ds)

    return ds
