import numpy as np
from pathlib import Path
from typing import List, Optional, Any, Dict

from deeplake.core.dataset.dataset import Dataset

GENERIC_TENSOR_CONFIG = {"htype": "generic", "sample_compression": "lz4"}


def coco_2_deeplake(
    coco_key: str, value: Any, dtype: str, category_lookup: Optional[Dict] = None
):
    """Takes a key-value pair from coco data and converts to Deep Lake format

    Args:
        coco_key (str): The key from the COCO annotation file.
        value (Any): The value corresponding to the key.
        dtype (str): Dtype for the resulting value.
        category_lookup (Dict): The mapping from COCO ``category_id`` to ``category_name``

    """
    if coco_key == "bbox":
        return np.array(value).astype(dtype)
    elif coco_key == "segmentation":
        # Make sure there aren't multiple segementations per single value, because multiple things will break
        if len(value) > 1:
            print("MULTIPLE SEGMENTATIONS PER OBJECT")

        return (
            np.array(value[0]).reshape(((int(len(value[0]) / 2)), 2)).astype(dtype)
        )  # Covert to Deep Lake polygon format

    elif coco_key == "category_id":
        if category_lookup is None:
            return value
        else:
            return category_lookup[str(value)]

    else:
        return value


class ParsedTensorStructure:
    class ParsedGroupStructure:
        def __init__(
            self, file_path: str, raw_keys: List[str], tensor_names: List[str]
        ) -> None:
            self.file_path = file_path
            self.file_name = Path(file_path).name
            self.raw_keys = raw_keys
            self.tensor_names = tensor_names

    def __init__(
        self,
        groups: Dict[str, ParsedGroupStructure],
        tensor_settings: Dict[str, Dict],
        ignore_one_group: bool = False,
    ) -> None:
        self.groups = groups
        self.tensor_settings = tensor_settings
        self.ignore_one_group = ignore_one_group

    def set_group(self, group_name: str, structure: Dict):
        self.groups[group_name] = self.ParsedGroupStructure(
            file_path=structure.get("file_path"),
            raw_keys=structure.get("raw_keys"),
            tensor_names=structure.get("tensor_names"),
        )

    def _create_tensors(self, ds: Dataset, group: ParsedGroupStructure):
        for i, raw_key in enumerate(group.raw_keys):
            tensor_settings = self.tensor_settings.get(raw_key, GENERIC_TENSOR_CONFIG)
            ds.create_tensor(group.tensor_names[i], **tensor_settings)

    def create_tensors(self, ds: Dataset):
        keys = self.groups.keys()
        if len(keys) == 1 and self.ignore_one_group:
            self._create_tensors(ds=ds, group=self.groups[keys[0]])
            return

        for group_name in self.groups.keys():
            group = ds.create_group(group_name)
            self._create_tensors(ds=group, group=self.groups[group_name])
