from typing import Optional, Dict, List, Union

from deeplake.core.dataset import Dataset


class TensorStructure:
    """Contains the name and necessary parameters of a tensor to be created in a dataset."""

    def __init__(
        self,
        name: str,
        params: Optional[Dict] = None,
    ) -> None:
        self.name = name
        self.params = params if params is not None else dict()

    def create(self, ds: Dataset):
        ds.create_tensor(self.name, **self.params)

    def create_missing(self, ds: Dataset):
        if self.name not in ds.tensors:
            self.create(ds)


class GroupStructure:
    """Represents a group in a dataset, containing a list of TensorStructure and nested GroupStructure objects."""

    def __init__(
        self,
        name: str,
        items: Optional[List[Union[TensorStructure, "GroupStructure"]]] = None,
    ):
        self.name = name
        self.items = items if items is not None else []

    @property
    def groups(self):
        return [g for g in self.items if isinstance(g, GroupStructure)]

    @property
    def tensors(self):
        return [t for t in self.items if isinstance(t, TensorStructure)]

    @property
    def all_keys(self):
        keys = set([f"{self.name}/{t.name}" for t in self.tensors])
        for g in self.groups:
            keys.update([f"{self.name}/{k}" for k in g.all_keys])

        return keys

    def add_item(self, item: Union[TensorStructure, "GroupStructure"]):
        self.items.append(item)

    def create(self, ds: Dataset):
        ds.create_group(self.name)

        for item in self.items:
            item.create(ds=ds[self.name])

    def create_missing(self, ds: Dataset):
        if self.name not in ds.groups:
            ds.create_group(self.name)

        for item in self.items:
            item.create_missing(ds=ds[self.name])


class DatasetStructure:
    """
    Represents a collection of TensorStructure and GroupStructure objects, forming
    the structure of a dataset parsed by an ingestion template.

    Supports adding items, creating the tensors and groups (or only the missing ones) in a given Dataset object.
    """

    def __init__(
        self,
        structure: Optional[List[Union[TensorStructure, GroupStructure]]] = None,
        ignore_one_group: bool = False,
    ) -> None:
        """Creates a new DatasetStructure object.

        Args:
            structure: An initial list of TensorStructure and GroupStructure objects.
            ignore_one_group: If True, the structure will be flattened if it contains only one group.
        """
        self.structure = structure if structure is not None else []
        self.ignore_one_group = ignore_one_group

    def __getitem__(self, key):
        try:
            return [i for i in self.structure if i.name == key][0]
        except IndexError:
            raise KeyError(f"Key {key} not found in structure.")

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

    @property
    def all_keys(self):
        keys = set([t.name for t in self.tensors])
        groups = self.groups

        if self.ignore_one_group and len(groups) == 1:
            keys.update([t.name for t in groups[0].tensors])
            return keys

        for group in groups:
            keys.update(group.all_keys)

        return keys

    def create_full(self, ds: Dataset):
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

    def create_missing(self, ds: Dataset):
        first_level_tensors = self.tensors
        groups = self.groups

        for tensor in first_level_tensors:
            tensor.create_missing(ds)

        if self.ignore_one_group and len(groups) == 1:
            for tensor in groups[0].tensors:
                tensor.create_missing(ds)
            return

        for group in groups:
            group.create_missing(ds)
