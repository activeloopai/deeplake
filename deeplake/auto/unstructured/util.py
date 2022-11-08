from typing import Optional, Dict, List, Union

from deeplake.core.dataset import Dataset
from deeplake.core.tensor import Tensor


class TensorStructure:
    def __init__(
        self,
        name: str,
        params: Optional[Dict] = None,
        primary: bool = False,
        meta_data: Optional[Dict] = None,
    ) -> None:
        self.name = name
        self.params = params if params is not None else dict()
        self.primary = primary
        self.meta_data = meta_data

    def create(self, ds: Union[Dataset, Tensor]):
        ds.create_tensor(self.name, **self.params)


class GroupStructure:
    def __init__(
        self,
        name: str,
        items: Optional[List[Union[TensorStructure, "GroupStructure"]]] = None,
        meta_data: Optional[Dict] = None,
    ):
        self.name = name
        self.items = items if items is not None else []
        self.meta_data = meta_data

    @property
    def groups(self):
        return [g for g in self.items if isinstance(g, GroupStructure)]

    @property
    def tensors(self):
        return [t for t in self.items if isinstance(t, TensorStructure)]

    def add_item(self, item: Union[TensorStructure, "GroupStructure"]):
        self.items.append(item)

    def create(self, ds: Union[Dataset, Tensor]):
        ds.create_group(self.name)

        for item in self.items:
            item.create(ds=ds[self.name])


class DatasetStructure:
    def __init__(
        self,
        structure: Optional[List[Union[TensorStructure, GroupStructure]]] = None,
        ignore_one_group: bool = False,
    ) -> None:
        self.structure = structure if structure is not None else []
        self.ignore_one_group = ignore_one_group

    def add_first_level_tensor(self, tensor: TensorStructure):
        self.structure.append(tensor)

    def add_group(self, group: GroupStructure):
        self.structure.append(group)

    @property
    def groups(self):
        return [g for g in self.structure if isinstance(g, GroupStructure)]

    @property
    def tensors(self):
        return [t for t in self.structure if isinstance(t, TensorStructure)]

    def create_structure(self, ds: Dataset):
        first_level_tensors = self.tensors
        groups = self.groups

        for tensor in first_level_tensors:
            tensor.create(ds)

        if self.ignore_one_group and len(groups) == 1:
            for tensor in groups[0].tensors:
                tensor.create(ds)
            return

        for group in groups:
            group.create(ds)
