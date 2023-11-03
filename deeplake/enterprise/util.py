from typing import Optional
from deeplake.integrations.pytorch.common import collate_fn as pytorch_collate_fn
from deeplake.integrations.tf.common import collate_fn as tf_collate_fn
from deeplake.util.iterable_ordered_dict import IterableOrderedDict
from deeplake.core.storage import GCSProvider, GDriveProvider, MemoryProvider

import os


def raise_indra_installation_error(indra_import_error: Optional[Exception] = None):
    if not indra_import_error:
        if os.name == "nt":
            raise ImportError(
                "High performance features require the libdeeplake package which is not available in Windows OS"
            )
        else:
            raise ImportError(
                "High performance features require the libdeeplake package. This package in typically installed by default, and you may install it separately using pip install libdeeplake."
            )
    raise ImportError(
        "Error while importing C++ backend. One of the dependencies might not be installed."
    ) from indra_import_error


def verify_base_storage(dataset):
    if isinstance(dataset.base_storage, (GDriveProvider, MemoryProvider)):
        raise ValueError(
            "Google Drive and Memory datasets are not supported for high-performance features currently."
        )


def get_collate_fn(collate, mode):
    if collate is None:
        if mode == "pytorch":
            return pytorch_collate_fn
        elif mode == "tensorflow":
            return tf_collate_fn
    return collate


def handle_mode(old_mode, new_mode):
    if old_mode is not None:
        if old_mode != new_mode:
            raise ValueError(f"Can't call .{new_mode}() after .{old_mode}()")
        raise ValueError(f"already called .{new_mode}()")
