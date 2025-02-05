import deeplake


def is_v3():
    assert int(deeplake.__version__.split(".")[0]) == 3
    return int(deeplake.__version__.split(".")[0]) == 3


def text_tensor_create_kwargs_():
    if not is_v3():
        raise ValueError("expected deeplake version 3")
    return {"htype": "text"}


def generic_tensor_create_kwargs_(dtype):
    if not is_v3():
        raise ValueError("expected deeplake version 3")
    return {"htype": "generic", "dtype": dtype}


def binary_mask_tensor_create_kwargs_(sample_compression="lz4"):
    if not is_v3():
        raise ValueError("expected deeplake version 3")
    return {
        "htype": "binary_mask",
        "dtype": "bool",
        "sample_compression": sample_compression,
    }


def class_label_tensor_create_kwargs_(dtype="int32"):
    if not is_v3():
        raise ValueError("expected deeplake version 3")
    return {
        "htype": "class_label",
        "dtype": dtype,
        "class_names": [],
        "chunk_compression": "lz4",
    }


def image_tensor_create_kwargs_(sample_compression="jpg"):
    if not is_v3():
        raise ValueError("expected deeplake version 3")
    return {"htype": "image", "sample_compression": sample_compression}


def bbox_tensor_create_kwargs_(dtype="int32", type="pixel", mode="LTWH"):
    if not is_v3():
        raise ValueError("expected deeplake version 3")
    return {"htype": "bbox", "dtype": dtype, "coords": {"type": type, "mode": mode}}


def polygon_tensor_create_kwargs_():
    if not is_v3():
        raise ValueError("expected deeplake version 3")
    return {"htype": "polygon", "dtype": "int32"}


def point_tensor_create_kwargs_():
    if not is_v3():
        raise ValueError("expected deeplake version 3")
    return {"htype": "point", "dtype": "int32"}


class tensor_wrapper:
    def __init__(self, ds, name):
        self.ds = ds
        self.name = name
        self.tensor = ds[name]

    def value(self, key, aslist=False):
        v = self.tensor[key].numpy(aslist=aslist)
        return v.tolist() if aslist else v

    def set_value(self, key, value):
        self.tensor[key] = value

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
            return self.tensor.info
        return self.tensor.metadata

    def update_metadata(self, metadata):
        if is_v3():
            self.tensor.info.update(metadata)
            return

        for key, value in metadata.items():
            self.tensor.metadata[key] = value


class dataset_wrapper:
    @staticmethod
    def create(ds_path, token=None, org_id=None, creds=None, overwrite=False):
        if not is_v3():
            raise ValueError("expected deeplake version 3")
        return dataset_wrapper(
            deeplake.empty(
                ds_path, token=token, org_id=org_id, creds=creds, overwrite=overwrite
            )
        )

    def __init__(self, ds):
        if isinstance(ds, dataset_wrapper):
            self.ds = ds.ds
        else:
            self.ds = ds

    def create_tensor(self, name, **kwargs):
        return self.add_column(name, **kwargs)

    def add_column(self, name, **kwargs):
        if not is_v3():
            raise ValueError("expected deeplake version 3")
        return self.ds.create_tensor(name=name, **kwargs)

    def extend(self, tensors, values):
        if not is_v3():
            raise ValueError("expected deeplake version 3")
        for tensor, value in zip(tensors, values):
            self.ds[tensor].extend(value)

    def fill_data(self, tensor_name, values, offset):
        if not is_v3():
            raise ValueError("expected deeplake version 3")

        if len(self.ds[tensor_name]) < offset:
            self.ds[tensor_name].extend([None] * (offset - len(self.ds[tensor_name])))

        extend_index = len(self.ds[tensor_name]) - offset

        if extend_index > 0:
            self.ds[tensor_name][offset : offset + extend_index] = values[:extend_index]

        if len(values) > extend_index:
            self.ds[tensor_name].extend(values[extend_index:])

    def pad_all_tensors(self):
        if not is_v3():
            raise ValueError("expected deeplake version 3")

        ml = self.ds.max_len
        for tensor_name in self.ds.tensors:
            if len(self.ds[tensor_name]) < ml:
                self.ds[tensor_name].extend([None] * (ml - len(self.ds[tensor_name])))

    @property
    def metadata(self):
        if not is_v3():
            raise ValueError("expected deeplake version 3")
        return self.ds.info

    @property
    def info(self):
        return self.metadata

    def __getitem__(self, key):
        if is_v3():
            return tensor_wrapper(self.ds, key)
        return tensor_wrapper(self.ds, key)

    def commit(self, commit_message=None):
        if not is_v3():
            raise ValueError("expected deeplake version 3")
        self.ds.commit(commit_message)

    def __len__(self):
        if not is_v3():
            raise ValueError("expected deeplake version 3")
        return len(self.ds)
