import os
from typing import Callable, Any, Dict

import numpy

import deeplake
from ._deeplake import *

__version__ = "4.0.3"

__all__ = [
    "__version__",
    "FutureVoid",
    "Future",
    "Tag",
    "TagView",
    "TagNotFoundError",
    "Tags",
    "TagsView",
    "ColumnDefinition",
    "ColumnDefinitionView",
    "ColumnView",
    "Column",
    "Version",
    "Prefetcher",
    "DatasetView",
    "Dataset",
    "ReadOnlyDataset",
    "ExpiredTokenError",
    "FormatNotSupportedError",
    "UnevenColumnsError",
    "UnevenUpdateError",
    "ColumnMissingAppendValueError",
    "ColumnAlreadyExistsError",
    "ColumnDoesNotExistError",
    "InvalidColumnValueError",
    "PushError",
    "GcsStorageProviderFailed",
    "History",
    "InvalidType",
    "LogExistsError",
    "LogNotexistsError",
    "IncorrectDeeplakePathError",
    "AuthenticationError",
    "AuthorizationError",
    "NotFoundError",
    "AgreementError",
    "AgreementNotAcceptedError",
    "NotLoggedInAgreementError",
    "CannotTagUncommittedDatasetError",
    "TagExistsError",
    "JSONKeyNotFound",
    "JSONIndexNotFound",
    "UnknownFormat",
    "UnknownStringType",
    "InvalidChunkStrategyType",
    "InvalidSequenceOfSequence",
    "InvalidTypeAndFormatPair",
    "UnknownType",
    "InvalidTextType",
    "UnsupportedPythonType",
    "UnsupportedSampleCompression",
    "UnsupportedChunkCompression",
    "InvalidImageCompression",
    "InvalidMaskCompression",
    "DtypeMismatch",
    "UnspecifiedDtype",
    "DimensionsMismatch",
    "ShapeIndexOutOfChunk",
    "BytePositionIndexOutOfChunk",
    "TensorAlreadyExists",
    "CanNotCreateTensorWithProvidedCompressions",
    "WrongChunkCompression",
    "WrongSampleCompression",
    "UnknownBoundingBoxCoordinateFormat",
    "UnknownBoundingBoxPixelFormat",
    "InvalidTypeDimensions",
    "Metadata",
    "ReadOnlyMetadata",
    "Row",
    "RowRange",
    "RowRangeView",
    "RowView",
    "Schema",
    "SchemaView",
    "StorageAccessDenied",
    "StorageKeyAlreadyExists",
    "StorageKeyNotFound",
    "StorageNetworkConnectionError",
    "StorageInternalError",
    "WriteFailedError",
    "QuantizationType",
    "core",
    "create",
    "create_async",
    "copy",
    "delete",
    "formats",
    "exists",
    "open",
    "open_async",
    "like",
    "convert",
    "connect",
    "disconnect",
    "open_read_only",
    "open_read_only_async",
    "from_parquet",
    "query",
    "query_async",
    "schemas",
    "storage",
    "tql",
    "types",
    "Client",
    "client",
    "__child_atfork",
    "__prepare_atfork",
    "__parent_atfork",
]


def _tensorflow(self) -> Any:
    from deeplake._tensorflow import _from_dataset
    return _from_dataset(self)


def _pytorch(self, transform: Callable[[Any], Any] = None):
    from deeplake._torch import TorchDataset
    return TorchDataset(self, transform=transform)


DatasetView.pytorch = _pytorch
DatasetView.tensorflow = _tensorflow

def load(*args, **kwargs):
    """
    .. deprecated:: 4.0.0
    """
    raise Exception("""
The API for Deep Lake 4.0 has changed significantly, including the `load` method being replaced by `open`.
To continue using Deep Lake 3.x, use `pip install "deeplake<4"`.
For information on migrating your code, see https://docs.deeplake.ai/latest/details/v3_conversion/
    """.replace("\n", " ").strip())

def empty(*args, **kwargs):
    """
    .. deprecated:: 4.0.0
    """
    raise Exception("""
The API for Deep Lake 4.0 has changed significantly, including the `empty` method being replaced by `create`.
To continue using Deep Lake 3.x, use `pip install "deeplake<4"`.
For information on migrating your code, see https://docs.deeplake.ai/latest/details/v3_conversion/
    """.replace("\n", " ").strip())

def convert(src: str, dst: str, dst_creds: Dict[str, str] = None):
    """
    Copies the v3 dataset at src into a new dataset in the new v4 format.
    """

    source_ds = deeplake.query(f'select * from "{src}"')
    dest_ds = deeplake.like(source_ds, dst, dst_creds)
    dest_ds.commit("Created dataset")

    dl = deeplake.Prefetcher(source_ds, batch_size=10000)
    counter = 0
    print(f"Transferring {len(source_ds)} rows to {dst}...")
    for b in dl:
        dest_ds.append(b)
        counter += 1
        if counter > 0 and counter % 100 == 0:
            dest_ds.commit()
    dest_ds.commit()
    print(f"Transferring data.... to {dst}... DONE")



def __register_at_fork():
    from ._deeplake import __prepare_atfork, __parent_atfork, __child_atfork

    UNSAFE_TYPES = (
    Dataset, DatasetView, ReadOnlyDataset, Column, ColumnView, ColumnDefinition, ColumnDefinitionView, Row, RowView,
    RowRange, RowRangeView, Schema, SchemaView, Version, History, Prefetcher,Tag,Tags)

    def check_main_globals_for_unsafe_types():
        import inspect
        import warnings

        frame = inspect.currentframe()

        try:
            while frame:
                for key, value in frame.f_globals.items():
                    if isinstance(value, UNSAFE_TYPES):
                        warnings.warn(
                            f"Global variable '{key}' of type {type(value)} may cause issues when using fork-based multiprocessing. Consider avoiding global variables of this type, or pass to subprocess as an agrument or by manual pickling."
                        )
                frame = frame.f_back
        finally:
            del frame

    def before_fork():
        check_main_globals_for_unsafe_types()
        __prepare_atfork()

    def after_fork_parent():
        __parent_atfork()

    def after_fork_child():
        __child_atfork()

    os.register_at_fork(
        before=before_fork,
        after_in_parent=after_fork_parent,
        after_in_child=after_fork_child,
    )


__register_at_fork()
