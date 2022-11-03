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

    def create(self, ds: Dataset):
        ds.create_tensor(self.name, **self.params)


class GroupStructure:
    def __init__(
        self,
        name: str,
        tensors: Optional[List[TensorStructure]] = None,
        meta_data: Optional[Dict] = None,
    ):
        self.name = name
        self.tensors = tensors if tensors is not None else []
        self.meta_data = meta_data

    def add_tensor(self, tensor: TensorStructure):
        self.tensors.append(tensor)

    def create(self, ds: Dataset, create_tensors: bool = False):
        ds.create_group(self.name)
        if create_tensors:
            for tensor in self.tensors:
                tensor.create(ds=ds[self.name])


class DatasetStructure:
    def __init__(
        self,
        structure: Optional[List[Union[TensorStructure, GroupStructure]]] = None,
        ignore_one_group: bool = False,
    ) -> None:
        self.structure = structure if structure is not None else []
        self.ignore_one_group = ignore_one_group

    def add_first_level_tensor(self, tensor: TensorStructure):
        # TODO: Handle validation
        self.structure.append(tensor)

    def add_group(self, group: GroupStructure):
        # TODO: Handle validation
        self.structure.append(group)

    def add_tensor_to_group(self, group: str, tensor: TensorStructure):
        # TODO: Handle validation
        group = [g for g in self.structure if g.name == group][0]
        group.add_tensor(tensor)

    def create_structure(self, ds: Dataset):
        first_level_tensors: List[TensorStructure] = []
        groups = [
            g
            for g in self.structure
            if isinstance(g, GroupStructure) or first_level_tensors.append(g)
        ]

        for tensor in first_level_tensors:
            tensor.create(ds)

        if self.ignore_one_group and len(groups) == 1:
            for tensor in groups[0].tensors:
                tensor.create(ds)
                return

        for group in groups:
            group.create(ds, create_tensors=True)
