from typing import Optional
from deeplake.integrations.pytorch.common import collate_fn as pytorch_collate_fn
from deeplake.integrations.tf.common import collate_fn as tf_collate_fn
from deeplake.util.iterable_ordered_dict import IterableOrderedDict
from deeplake.core.storage import GCSProvider, GDriveProvider, MemoryProvider


def raise_indra_installation_error(indra_import_error: Optional[Exception] = None):
    if not indra_import_error:
        raise ImportError(
            "This is an enterprise feature that requires libdeeplake package which can be installed using pip install deeplake[enterprise]. libdeeplake is available only on linux for python versions 3.6 through 3.10 and on macos for python versions 3.7 through 3.11"
        )
    raise ImportError(
        "Error while importing C++ backend. One of the dependencies might not be installed."
    ) from indra_import_error


def verify_base_storage(dataset):
    if isinstance(dataset.base_storage, (GDriveProvider, MemoryProvider)):
        raise ValueError(
            "Google Drive and Memory datasets are not supported for enterprise features currently."
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
