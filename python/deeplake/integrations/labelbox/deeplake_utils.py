import deeplake
import numpy as np


def is_v3():
    assert int(deeplake.__version__.split(".")[0]) != 3
    return False


def text_tensor_create_kwargs_():
    if is_v3():
        raise ValueError("unexpected deeplake version 3")
    return {"dtype": deeplake.types.Text()}


def generic_tensor_create_kwargs_(dtype):
    if is_v3():
        raise ValueError("unexpected deeplake version 3")
    return {"dtype": dtype}


def binary_mask_tensor_create_kwargs_(sample_compression="lz4"):
    if is_v3():
        raise ValueError("unexpected deeplake version 3")
    return {"dtype": deeplake.types.BinaryMask(sample_compression=sample_compression)}


def class_label_tensor_create_kwargs_(dtype="int32"):
    if is_v3():
        raise ValueError("unexpected deeplake version 3")
    return {
        "dtype": deeplake.types.ClassLabel(deeplake.types.Array(dtype, dimensions=1))
    }


def image_tensor_create_kwargs_(sample_compression="jpg"):
    if is_v3():
        raise ValueError("unexpected deeplake version 3")
    return {"dtype": deeplake.types.Image(sample_compression=sample_compression)}


def bbox_tensor_create_kwargs_(dtype="int32", type="pixel", mode="LTWH"):
    if is_v3():
        raise ValueError("unexpected deeplake version 3")
    return {"dtype": deeplake.types.BoundingBox(dtype, mode, type)}


def polygon_tensor_create_kwargs_():
    if is_v3():
        raise ValueError("unexpected deeplake version 3")
    return {"dtype": deeplake.types.Polygon()}


def point_tensor_create_kwargs_():
    if is_v3():
        raise ValueError("unexpected deeplake version 3")
    return {"dtype": deeplake.types.Point()}


class tensor_wrapper:
    def __init__(self, ds, name):
        self.ds = ds
        self.name = name
        self.tensor = ds[name]

    def value(self, key, aslist=False):
        v = self.tensor[key]
        if aslist:
            return v.tolist() if v is not None else []

        return v

    def set_value(self, key, value):
        if not isinstance(key, slice):
            self.tensor[key] = value
            return

        for i, v in zip(range(*key.indices(len(self.tensor))), value):
            if v is None:
                continue
            self.tensor[i] = v

    def __len__(self):
        return len(self.tensor)

    def __iter__(self):
        return iter(self.tensor)

    @property
    def info(self):
        return self.metadata

    @property
    def metadata(self):
        if is_v3():
            raise ValueError("unexpected deeplake version 3")
        return self.tensor.metadata

    def update_metadata(self, metadata):
        if is_v3():
            raise ValueError("unexpected deeplake version 3")

        for key, value in metadata.items():
            self.tensor.metadata[key] = value


class dataset_wrapper:
    @staticmethod
    def create(ds_path, token, org_id=None, creds=None, overwrite=False):
        if is_v3():
            raise ValueError("unexpected deeplake version 3")
        credentials = dict()
        if creds:
            credentials["creds"] = creds
        if org_id:
            credentials["org_id"] = org_id

        if overwrite:
            deeplake.delete(ds_path, creds=credentials, token=token)
        return dataset_wrapper(deeplake.create(ds_path, creds=credentials, token=token))

    def __init__(self, ds):
        if isinstance(ds, dataset_wrapper):
            self.ds = ds.ds
        else:
            self.ds = ds

    def create_tensor(self, name, **kwargs):
        return self.add_column(name, **kwargs)

    def add_column(self, name, **kwargs):
        if is_v3():
            raise ValueError("unexpected deeplake version 3")
        return self.ds.add_column(name, **kwargs)

    def extend(self, tensors, values):
        if is_v3():
            raise ValueError("unexpected deeplake version 3")

        self.ds.append([dict(zip(tensors, value)) for value in zip(*values)])

    def fill_data(self, tensor_name, values, offset):
        if is_v3():
            raise ValueError("unexpected deeplake version 3")

        assert len(self.ds) >= (offset + len(values))

        tensor_wrapper(self.ds, tensor_name).set_value(
            slice(offset, offset + len(values)), values
        )

    def pad_all_tensors(self):
        if is_v3():
            raise ValueError("unexpected deeplake version 3")
        # no need to pad in v4
        pass

    @property
    def metadata(self):
        if is_v3():
            raise ValueError("unexpected deeplake version 3")
        return self.ds.metadata

    @property
    def info(self):
        return self.metadata

    def __getitem__(self, key):
        if is_v3():
            raise ValueError("unexpected deeplake version 3")
        return tensor_wrapper(self.ds, key)

    def commit(self, commit_message=None):
        if is_v3():
            raise ValueError("unexpected deeplake version 3")
        self.ds.commit(commit_message)

    def __len__(self):
        if is_v3():
            raise ValueError("unexpected deeplake version 3")
        return len(self.ds)
