import os
from typing import Callable, Any, Dict, Optional

try:
    from tqdm import tqdm as progress_bar
except ImportError:

    def progress_bar(iterable, *args, **kwargs):
        return iterable


import numpy

import deeplake
from ._deeplake import *
from deeplake.ingestion import from_coco


__version__ = "4.3.0"

__all__ = [
    "__version__",
    "AgreementError",
    "AgreementNotAcceptedError",
    "Array",
    "AuthenticationError",
    "AuthorizationError",
    "BadRequestError",
    "Branch",
    "BranchExistsError",
    "BranchNotFoundError",
    "BranchView",
    "Branches",
    "BranchesView",
    "BytePositionIndexOutOfChunk",
    "CanNotCreateTensorWithProvidedCompressions",
    "CannotDeleteMainBranchError",
    "CannotRenameMainBranchError",
    "Client",
    "Column",
    "ColumnAlreadyExistsError",
    "ColumnDefinition",
    "ColumnDefinitionView",
    "ColumnDoesNotExistError",
    "ColumnMissingAppendValueError",
    "ColumnView",
    "CredsKeyAlreadyAssignedError",
    "Dataset",
    "DatasetUnavailableError",
    "DatasetView",
    "DimensionsMismatch",
    "DimensionsMismatchError",
    "DtypeMismatch",
    "EmbeddingSizeMismatch",
    "EmptyColumnNameError",
    "Executor",
    "ExplainQueryResult",
    "ExpiredTokenError",
    "FormatNotSupportedError",
    "Future",
    "FutureVoid",
    "GcsStorageProviderFailed",
    "HTTPBodyIsMissingError",
    "HTTPBodyIsNotJSONError",
    "HTTPRequestFailedError",
    "History",
    "IncorrectDeeplakePathError",
    "IndexAlreadyExistsError",
    "IndexingMode",
    "InvalidBinaryMaskCompression",
    "InvalidChunkStrategyType",
    "InvalidColumnValueError",
    "InvalidCredsKeyAssignmentError",
    "InvalidImageCompression",
    "InvalidIndexCreationError",
    "InvalidLinkDataError",
    "InvalidLinkType",
    "InvalidMedicalCompression",
    "InvalidPolygonShapeError",
    "InvalidSegmentMaskCompression",
    "InvalidSequenceOfSequence",
    "InvalidTextType",
    "InvalidType",
    "InvalidTypeAndFormatPair",
    "InvalidTypeDimensions",
    "InvalidURIError",
    "JSONIndexNotFound",
    "JSONKeyNotFound",
    "LogExistsError",
    "LogNotexistsError",
    "Metadata",
    "NotFoundError",
    "NotLoggedInAgreementError",
    "PermissionDeniedError",
    "PushError",
    "QuantizationType",
    "Random",
    "random",
    "ReadOnlyDataset",
    "ReadOnlyDatasetModificationError",
    "ReadOnlyMetadata",
    "Row",
    "RowRange",
    "RowRangeView",
    "RowView",
    "Schema",
    "SchemaView",
    "ShapeIndexOutOfChunk",
    "StorageAccessDenied",
    "StorageInternalError",
    "StorageKeyAlreadyExists",
    "StorageKeyNotFound",
    "StorageNetworkConnectionError",
    "Tag",
    "TagExistsError",
    "TagNotFoundError",
    "TagView",
    "Tags",
    "TagsView",
    "TensorAlreadyExists",
    "UnevenColumnsError",
    "UnevenUpdateError",
    "UnexpectedInputDataForDicomColumn",
    "UnexpectedMedicalTypeInputData",
    "UnknownBoundingBoxCoordinateFormat",
    "UnknownBoundingBoxPixelFormat",
    "UnknownFormat",
    "UnknownStringType",
    "UnknownType",
    "UnspecifiedDtype",
    "UnsupportedChunkCompression",
    "UnsupportedPythonType",
    "UnsupportedSampleCompression",
    "Version",
    "VersionNotFoundError",
    "WriteFailedError",
    "WrongChunkCompression",
    "WrongSampleCompression",
    "__prepare_atfork",
    "client",
    "connect",
    "convert",
    "copy",
    "core",
    "create",
    "create_async",
    "_create_global_cache",
    "delete",
    "delete_async",
    "disconnect",
    "exists",
    "explain_query",
    "from_coco",
    "from_csv",
    "from_parquet",
    "like",
    "open",
    "open_async",
    "open_read_only",
    "open_read_only_async",
    "prepare_query",
    "query",
    "query_async",
    "schemas",
    "storage",
    "tql",
    "types",
    "TelemetryClient",
    "telemetry_client"
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
    raise Exception(
        """
The API for Deep Lake 4.0 has changed significantly, including the `load` method being replaced by `open`.
To continue using Deep Lake 3.x, use `pip install "deeplake<4"`.
For information on migrating your code, see https://docs.deeplake.ai/latest/details/v3_conversion/
    """.replace(
            "\n", " "
        ).strip()
    )


def empty(*args, **kwargs):
    """
    .. deprecated:: 4.0.0
    """
    raise Exception(
        """
The API for Deep Lake 4.0 has changed significantly, including the `empty` method being replaced by `create`.
To continue using Deep Lake 3.x, use `pip install "deeplake<4"`.
For information on migrating your code, see https://docs.deeplake.ai/latest/details/v3_conversion/
    """.replace(
            "\n", " "
        ).strip()
    )


def convert(
    src: str,
    dst: str,
    dst_creds: Optional[Dict[str, str]] = None,
    token: Optional[str] = None,
) -> None:
    """
    Copies the v3 dataset at src into a new dataset in the new v4 format.
    """

    def commit_data(dataset, message="Committing data"):
        dataset.commit()

    def get_raw_columns(source):
        return [
            col.name
            for col in source.schema.columns
            if not col.dtype.is_link and col.dtype.kind in {
                deeplake.types.TypeKind.Image,
                deeplake.types.TypeKind.SegmentMask,
                deeplake.types.TypeKind.Medical,
            }
        ]

    def transfer_non_link_data(source, dest):
        dl = deeplake._deeplake._Prefetcher(source, raw_columns=set(get_raw_columns(source)))
        for counter, batch in enumerate(progress_bar(dl), start=1):
            dest.append(batch)
            if counter % 100 == 0:
                commit_data(dest)
        commit_data(dest, "Final commit of non-link data")

    def transfer_with_links(source, dest, links, column_names):
        iterable_cols = [col for col in column_names if col not in links]
        link_sample_info = {link: source[link]._links_info() for link in links}
        dest.set_creds_key(link_sample_info[links[0]]["key"])
        quoted_cols = ['"' + col + '"' for col in iterable_cols]
        joined_cols = ",".join(quoted_cols)
        pref_ds = source.query(f"SELECT {joined_cols}")
        dl = deeplake._deeplake._Prefetcher(pref_ds, raw_columns=set(get_raw_columns(source)))

        for counter, batch in enumerate(progress_bar(dl), start=1):
            batch_size = len(batch[iterable_cols[0]])
            for link in links:
                link_data = link_sample_info[link]["data"]
                start_index = (counter - 1) * batch_size
                end_index = min((counter) * batch_size, len(link_data))
                batch[link] = link_data[start_index:end_index]

            dest.append(batch)
            if counter % 100 == 0:
                commit_data(dest)
        commit_data(dest, "Final commit of linked data")

    source_ds = deeplake.query(f'SELECT * FROM "{src}"', token=token)
    dest_ds = deeplake.like(source_ds, dst, dst_creds, token=token)
    commit_data(dest_ds, "Created dataset")

    column_names = [col.name for col in source_ds.schema.columns]
    links = [
        col.name
        for col in source_ds.schema.columns
        if source_ds.schema[col.name].dtype.is_link
    ]
    print(f"Transferring {len(source_ds)} rows to {dst}...")
    if not links:
        transfer_non_link_data(source_ds, dest_ds)
    else:
        transfer_with_links(source_ds, dest_ds, links, column_names)

    for column in column_names:
        meta = dict(source_ds[column].metadata)
        if meta:
            for key, value in meta.items():
                dest_ds[column].metadata[key] = value

    commit_data(dest_ds, "Final commit of metadata")
    print(f"Data transfer to {dst} complete.")


def __register_at_fork():
    from ._deeplake import __prepare_atfork

    UNSAFE_TYPES = (
        Dataset,
        DatasetView,
        ReadOnlyDataset,
        Column,
        ColumnView,
        ColumnDefinition,
        ColumnDefinitionView,
        Row,
        RowView,
        RowRange,
        RowRangeView,
        Schema,
        SchemaView,
        Version,
        History,
        Tag,
        Tags,
    )

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
        pass

    def after_fork_parent():
        pass

    def after_fork_child():
        pass

    os.register_at_fork(
        before=before_fork,
        after_in_parent=after_fork_parent,
        after_in_child=after_fork_child,
    )

    ff = os.fork
    def fork():
        __prepare_atfork()
        return ff()

    os.fork = fork

__register_at_fork()
