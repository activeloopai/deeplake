import datetime
import typing

import formats
import storage
from . import schemas
from . import types

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


class Future:
    """
    A future that represents a value that will be resolved in the future.

    Once the Future is resolved, it will hold the result, and you can retrieve it 
    using either a blocking call (`result()`) or via asynchronous mechanisms (`await`).

    The future will resolve automatically even if you do not explicitly wait for it.

    Methods:
        result() -> typing.Any:
            Blocks until the Future is resolved and returns the object.
        
        __await__() -> typing.Any:
            Awaits the future asynchronously and returns the object once it's ready.
        
        is_completed() -> bool:
            Returns True if the Future is already resolved, False otherwise.
    """

    def result(self) -> typing.Any:
        """
        Blocks until the Future is resolved, then returns the result.

        Returns:
            typing.Any: The result when the Future is resolved.
        """
        ...

    def __await__(self) -> typing.Any:
        """
        Awaits the resolution of the Future asynchronously.

        Examples:
            >>> result = await future
        
        Returns:
            typing.Any: The result when the Future is resolved.
        """
        ...

    def is_completed(self) -> bool:
        """
        Checks if the Future has been resolved.

        Returns:
            bool: True if the Future is resolved, False otherwise.
        """
        ...

class FutureVoid:
    """
    A future that represents the completion of an operation that returns no result.

    The future will resolve automatically to `None`, even if you do not explicitly wait for it.

    Methods:
        wait() -> None:
            Blocks until the FutureVoid is resolved and then returns `None`.
        
        __await__() -> None:
            Awaits the FutureVoid asynchronously and returns `None` once the operation is complete.
        
        is_completed() -> bool:
            Returns True if the FutureVoid is already resolved, False otherwise.
    """

    def wait(self) -> None:
        """
        Blocks until the FutureVoid is resolved, then returns `None`.

        Examples:
            >>> future_void.wait()  # Blocks until the operation completes.

        Returns:
            None: Indicates the operation has completed.
        """
        ...

    def __await__(self) -> None:
        """
        Awaits the resolution of the FutureVoid asynchronously.

        Examples:
            >>> await future_void  # Waits for the completion of the async operation.
        
        Returns:
            None: Indicates the operation has completed.
        """
        ...

    def is_completed(self) -> bool:
        """
        Checks if the FutureVoid has been resolved.

        Returns:
            bool: True if the FutureVoid is resolved, False otherwise.
        """
        ...

class ReadOnlyMetadata:
    """
    ReadOnlyMetadata is a key-value store.
    """

    def __getitem__(self, key: str) -> typing.Any:
        """
        Get the value for the given key
        """
        ...

    def keys(self) -> list[str]:
        """
        Return a list of all keys in the metadata
        """
        ...

class Metadata(ReadOnlyMetadata):
    """
    Metadata is a key-value store.
    """

    def __setitem__(self, key: str, value: typing.Any) -> None:
        """
        Set the value for the given key. Setting the value will immediately persist the change without requiring a commit().
        """
        ...


def query(query: str, token: str | None = None) -> DatasetView:
    """
    Executes the given TQL query and returns a DatasetView.

    Compared to [deeplake.Dataset.query][], this version of query can join multiple datasets together
    or query a single dataset without opening it first.

    Examples:
        >>> r = deeplake.query("select * from \\"al://my_org/dataset\\" where id > 30")
    """

    ...

def query_async(query: str, token: str | None = None) -> Future:
    """
    Asynchronously executes the given TQL query and returns a Future that will resolve into DatasetView.

    Examples:
            >>> future = deeplake.query_async("select * where category == 'active'")
            >>> result = future.result()
            >>> for row in result:
            >>>     print("Id is: ", row["id"])

            >>> # or use the Future in an await expression
            >>> future = deeplake.query_async("select * where category == 'active'")
            >>> result = await future
            >>> for row in result:
            >>>     print("Id is: ", row["id"])
    """
    ...

class Client:
    endpoint: str


class Tag:
    """
    Describes a tag within the dataset.

    Tags are created using [deeplake.Dataset.tag][].
    """

    @property
    def id(self) -> str:
        """
        The unique identifier of the tag
        """

    @property
    def name(self) -> str:
        """
        The name of the tag
        """

    @property
    def version(self) -> str:
        """
        The version that has been tagged
        """

    def delete(self) -> None:
        """
        Deletes the tag from the dataset
        """
        ...

    def rename(self, new_name: str) -> None:
        """
        Renames the tag within the dataset
        """
        ...

    def open(self) -> ReadOnlyDataset:
        """
        Fetches the dataset corresponding to the tag
        """
        ...

    def open_async(self) -> Future:
        """
        Asynchronously fetches the dataset corresponding to the tag and returns a Future object.
        """
        ...

    def __repr__(self) -> str: ...

class TagView:
    """
    Describes a read-only tag within the dataset.

    Tags are created using [deeplake.Dataset.tag][].
    """

    @property
    def id(self) -> str:
        """
        The unique identifier of the tag
        """

    @property
    def name(self) -> str:
        """
        The name of the tag
        """

    @property
    def version(self) -> str:
        """
        The version that has been tagged
        """

    def open(self) -> ReadOnlyDataset:
        """
        Fetches the dataset corresponding to the tag
        """
        ...

    def open_async(self) -> Future:
        """
        Asynchronously fetches the dataset corresponding to the tag and returns a Future object.
        """
        ...

    def __repr__(self) -> str: ...

class TagNotFoundError(Exception):
    pass

class TagExistsError(Exception):
    pass

class CannotTagUncommittedDatasetError(Exception):
    pass

class PushError(Exception):
    pass

class Tags:
    """
    Provides access to the tags within a dataset.

    It is returned by the [deeplake.Dataset.tags][] property.
    """

    def __getitem__(self, name: str) -> Tag:
        """
        Return a tag by name
        """
        ...

    def __len__(self) -> int:
        """
        The total number of tags in the dataset
        """
        ...

    def __repr__(self) -> str: ...

    def names(self) -> list[str]:
        """
        Return a list of tag names
        """

    ...

class TagsView:
    """
    Provides access to the tags within a dataset.

    It is returned by the [deeplake.Dataset.tags][] property on a [deeplake.ReadOnlyDataset][].
    """

    def __getitem__(self, name: str) -> TagView:
        """
        Return a tag by name
        """
        ...

    def __len__(self) -> int:
        """
        The total number of tags in the dataset
        """
        ...

    def __repr__(self) -> str: ...

    def names(self) -> list[str]:
        """
        Return a list of tag names
        """

    ...


class ColumnDefinition:
    def __repr__(self) -> str: ...

    @property
    def name(self) -> str:
        """
        The name of the column
        """
        ...

    @property
    def dtype(self) -> types.Type:
        """
        The column datatype
        """
        ...

    def drop(self) -> None:
        """
        Drops the column from the dataset.
        """
        ...

    def rename(self, new_name: str) -> None:
        """
        Renames the column

        Args:
            new_name: The new name for the column

        """
        ...


class ColumnDefinitionView:
    """
    A read-only view of a [deeplake.ColumnDefinition][]
    """

    def __repr__(self) -> str: ...

    @property
    def name(self) -> str:
        """
        The name of the column
        """
        ...

    @property
    def dtype(self) -> types.Type:
        """
        The column datatype
        """
        ...


class ColumnView:
    """
    Provides access to a column in a dataset.
    """

    def __getitem__(self, index: int | slice) -> typing.Any: ...

    def get_async(self, index: int | slice) -> Future: ...

    def __len__(self) -> int: ...

    def __repr__(self) -> str: ...

    @property
    def metadata(self) -> ReadOnlyMetadata: ...

    @property
    def name(self) -> str: ...


class Column(ColumnView):
    def __setitem__(self, index: int | slice, value: typing.Any) -> None: ...
    def set_async(self, index: int | slice, value: typing.Any) -> FutureVoid: ...

    @property
    def metadata(self) -> Metadata: ...


class Version:
    """
    An atomic change within [deeplake.Dataset][]'s history
    """

    def __repr__(self) -> str: ...

    @property
    def client_timestamp(self) -> datetime.datetime:
        """
        When the version was created, according to the writer's local clock.

        This timestamp is not guaranteed to be accurate, and [deeplake.Version.timestamp][] should generally be used instead.
        """

    @property
    def message(self) -> str | None:
        """
        The description of the version provided at commit time.
        """

    @property
    def timestamp(self) -> datetime.datetime:
        """
        The version timestamp.

        This is based on the storage provider's clock, and so generally more accurate than [deeplake.Version.client_timestamp][].
        """

    @property
    def id(self) -> str:
        """
        The unique version identifier
        """

    def open(self) -> ReadOnlyDataset:
        """
        Fetches the dataset corresponding to the version
        """
        ...


class Row:
    """
    Provides mutable access to a particular row in a dataset.
    """

    def __getitem__(self, column: str) -> typing.Any:
        """
        The value for the given column
        """

    def get_async(self, column: str) -> Future:
        """
        Asynchronously retrieves data for the specified column and returns a Future object.
    
        Args:
            column (str): The name of the column to retrieve data for.

        Returns:
            Future: A Future object that will resolve to the value containing the column data.

        Examples:
            >>> future = row.get_async("column_name")
            >>> column = future.result()  # Blocking call to get the result when it's ready.
        
        Notes:
            - The Future will resolve asynchronously, meaning the method will not block execution 
            while the data is being retrieved.
            - You can either wait for the result using `future.result()` (a blocking call) 
            or use the Future in an `await` expression.
        """

    def __setitem__(self, column: str, value: typing.Any) -> None:
        """
        Change the value for the given column
        """

    def set_async(self, column: str, value: typing.Any) -> FutureVoid:
        """
        Asynchronously sets a value for the specified column and returns a FutureVoid object.

        Args:
            column (str): The name of the column to update.
            value (typing.Any): The value to set for the column.

        Returns:
            FutureVoid: A FutureVoid object that will resolve when the operation is complete.

        Examples:
            >>> future_void = row.set_async("column_name", new_value)
            >>> future_void.wait()  # Blocks until the operation is complete.
        
        Notes:
            - The method sets the value asynchronously and immediately returns a FutureVoid.
            - You can either block and wait for the operation to complete using `wait()` 
            or await the FutureVoid object in an asynchronous context.
        """

    def __repr__(self) -> str: ...

    @property
    def row_id(self) -> int:
        """
        The row_id of the row
        """


class RowRange:
    """
    Provides mutable access to a row range in a dataset.
    """

    def __iter__(self) -> typing.Iterator[Row]:
        """
        Iterate over the row range
        """

    def __len__(self) -> int:
        """
        The number of rows in the row range
        """

    def __getitem__(self, column: str) -> typing.Any:
        """
        The value for the given column
        """
    
    def get_async(self, column: str) -> Future:
        """
        Asynchronously retrieves data for the specified column and returns a Future object.
    
        Args:
            column (str): The name of the column to retrieve data for.

        Returns:
            Future: A Future object that will resolve to the value containing the column data.

        Examples:
            >>> future = row_range.get_async("column_name")
            >>> column = future.result()  # Blocking call to get the result when it's ready.
        
        Notes:
            - The Future will resolve asynchronously, meaning the method will not block execution 
            while the data is being retrieved.
            - You can either wait for the result using `future.result()` (a blocking call) 
            or use the Future in an `await` expression.
        """

    def __setitem__(self, column: str, value: typing.Any) -> None:
        """
        Change the value for the given column
        """

    def set_async(self, column: str, value: typing.Any) -> FutureVoid:
        """
        Asynchronously sets a value for the specified column and returns a FutureVoid object.

        Args:
            column (str): The name of the column to update.
            value (typing.Any): The value to set for the column.

        Returns:
            FutureVoid: A FutureVoid object that will resolve when the operation is complete.

        Examples:
            >>> future_void = row_range.set_async("column_name", new_value)
            >>> future_void.wait()  # Blocks until the operation is complete.
        
        Notes:
            - The method sets the value asynchronously and immediately returns a FutureVoid.
            - You can either block and wait for the operation to complete using `wait()` 
            or await the FutureVoid object in an asynchronous context.
        """

    def __repr__(self) -> str: ...


class RowRangeView:
    """
    Provides access to a row range in a dataset.
    """

    def __iter__(self) -> typing.Iterator[RowView]:
        """
        Iterate over the row range
        """

    def __len__(self) -> int:
        """
        The number of rows in the row range
        """

    def __getitem__(self, column: str) -> typing.Any:
        """
        The value for the given column
        """

    def get_async(self, column: str) -> Future:
        """
        Asynchronously retrieves data for the specified column and returns a Future object.
    
        Args:
            column (str): The name of the column to retrieve data for.

        Returns:
            Future: A Future object that will resolve to the value containing the column data.

        Examples:
            >>> future = row_range_view.get_async("column_name")
            >>> column = future.result()  # Blocking call to get the result when it's ready.
        
        Notes:
            - The Future will resolve asynchronously, meaning the method will not block execution 
            while the data is being retrieved.
            - You can either wait for the result using `future.result()` (a blocking call) 
            or use the Future in an `await` expression.
        """

    def __repr__(self) -> str: ...


class RowView:
    """
    Provides access to a particular row in a dataset.
    """

    def __getitem__(self, column: str) -> typing.Any:
        """
        The value for the given column
        """

    def get_async(self, column: str) -> Future:
        """
        Asynchronously retrieves data for the specified column and returns a Future object.
    
        Args:
            column (str): The name of the column to retrieve data for.

        Returns:
            Future: A Future object that will resolve to the value containing the column data.

        Examples:
            >>> future = row_view.get_async("column_name")
            >>> column = future.result()  # Blocking call to get the result when it's ready.
        
        Notes:
            - The Future will resolve asynchronously, meaning the method will not block execution 
            while the data is being retrieved.
            - You can either wait for the result using `future.result()` (a blocking call) 
            or use the Future in an `await` expression.
        """

    def __repr__(self) -> str: ...

    @property
    def row_id(self) -> int:
        """
        The row_id of the row
        """


class DatasetView:
    """
    A DatasetView is a dataset-like structure. It has a defined schema and contains data which can be queried.
    """

    def __repr__(self) -> str: ...

    @typing.overload
    def __getitem__(self, offset: int) -> RowView:
        """
        Get a row by offset within the DatasetView.
        """
        ...

    @typing.overload
    def __getitem__(self, range: slice) -> RowRangeView:
        """
        Get a range of rows by offset within the DatasetView.
        """
        ...

    @typing.overload
    def __getitem__(self, column: str) -> ColumnView:
        """
        Get a column by name within the DatasetView.
        """
        ...

    def __getitem__(
            self, input: int | slice | str
    ) -> RowView | RowRangeView | ColumnView:
        """
        Returns a subset of data from the DatasetView.

        The result will depend on the type of value passed to the `[]` operator.

        - `int`: The zero-based offset of the single row to return. Returns a [deeplake.RowView][]
        - `slice`: A slice specifying the range of rows to return. Returns a [deeplake.RowRangeView][]
        - `str`: A string specifying column to return all values from. Returns a [deeplake.ColumnView][]

        Examples:
            >>> ds = deeplake.create("mem://")
            >>> ds.add_column("id", int)
            >>> ds.add_column("name", str)
            >>> ds.append({"id": [1,2,3], "name": ["Mary", "Joe", "Bill"]})
            >>>
            >>> row = ds[1]
            >>> print("Id:", row["id"], "Name:", row["name"])
            Id: 2 Name: Joe
            >>> rows = ds[1:2]
            >>> print(rows["id"])

            >>> column_data = ds["id"]

        """

    def __iter__(self) -> typing.Iterator[RowView]:
        """
        Row based iteration over the dataset.

        Examples:
            >>> for row in ds:
            >>>     # process row
            >>>     pass

        """
        ...

    def __len__(self) -> int:
        """
        The number of rows in the dataset
        """
        ...

    def summary(self) -> None:
        """
        Prints a summary of the dataset.

        Examples:
            >>> ds.summary()
            Dataset(columns=(id,title,embedding), length=51611356)
            +---------+-------------------------------------------------------+
            | column  |                         type                          |
            +---------+-------------------------------------------------------+
            |   id    |               kind=generic, dtype=int32               |
            +---------+-------------------------------------------------------+
            |  title  |                         text                          |
            +---------+-------------------------------------------------------+
            |embedding|kind=embedding, dtype=array(dtype=float32, shape=[768])|
            +---------+-------------------------------------------------------+

        """
        ...

    def query(self, query: str) -> DatasetView:
        """
        Executes the given TQL query against the dataset and return the results as a [deeplake.DatasetView][].

        Examples:
            >>> result = ds.query("select * where category == 'active'")
            >>> for row in result:
            >>>     print("Id is: ", row["id"])

        """
        ...

    def query_async(self, query: str) -> Future:
        """
        Asynchronously executes the given TQL query against the dataset and return a future that will resolve into [deeplake.DatasetView][].

        Examples:
            >>> future = ds.query_async("select * where category == 'active'")
            >>> result = future.result()
            >>> for row in result:
            >>>     print("Id is: ", row["id"])

            >>> # or use the Future in an await expression
            >>> future = ds.query_async("select * where category == 'active'")
            >>> result = await future
            >>> for row in result:
            >>>     print("Id is: ", row["id"])
        """
        ...

    @property
    def schema(self) -> SchemaView:
        """
        The schema of the dataset.
        """


    def tensorflow(self) -> typing.Any:
        """
        Returns a TensorFlow `tensorflow.data.Dataset` wrapper around this DatasetView.

        Raises:
            ImportError: If TensorFlow is not installed

        Examples:
            >>> ds = deeplake.open("path/to/dataset")
            >>> dl = ds.tensorflow().shuffle(500).batch(32).
            >>> for i_batch, sample_batched in enumerate(dataloader):
            >>>      process_batch(sample_batched)

        """
        ...


    def pytorch(self, transform: typing.Callable[[typing.Any], typing.Any] = None):
        """
        Returns a PyTorch `torch.utils.data. Dataset` wrapper around this dataset.

        By default, no transformations are applied and each row is returned as a `dict` with keys of column names.

        Parameters:
             transform: A custom function to apply to each sample before returning it

        Raises:
            ImportError: If pytorch is not installed

        Examples:
            >>> from torch.utils.data import DataLoader
            >>>
            >>> ds = deeplake.open("path/to/dataset")
            >>> dataloader = DataLoader(ds.pytorch(), batch_size=60,
            >>>                             shuffle=True, num_workers=10)
            >>> for i_batch, sample_batched in enumerate(dataloader):
            >>>      process_batch(sample_batched)

        """
        ...

    def batches(self, batch_size: int, drop_last: bool = False) -> Prefetcher:
        """
        Return a [deeplake.Prefetcher][] for this DatasetView

        Parameters:
            batch_size: Number of rows in each batch
            drop_last: Whether to drop the final batch if it is incomplete
        """
        ...

class Prefetcher:
    """
    The Prefetcher can be used to more efficiently stream large amounts of data from a DeepLake dataset, such as to the DataLoader then to the training framework.

    Examples:
        >>> ds = deeplake.open("al://my_org/dataset")
        >>> fetcher = deeplake.Prefetcher(view, batch_size=2000)
        >>> for batch in dl:
        >>>     process_batch(batch["images"])

    """

    def __init__(
            self,
            dataset: DatasetView,
            batch_size: int = 1,
            drop_last: bool = False,
    ) -> None:
        """
        Parameters:
             dataset: The [deeplake.DatasetView][] to stream from
             batch_size: The numer of rows to return in each iteration
             drop_last: If true, do not return a non-full final batch
        """
        ...

    def __iter__(self) -> Prefetcher:
        """
        Iterate over the dataset view
        """
        ...

    def __len__(self) -> int:
        """
        The number of batches in the Prefetcher
        """
        ...

    def __next__(self) -> dict:
        """
        Returns the next batch of dataset
        """
        ...

    def reset(self) -> None:
        """
        Reset the iterator
        """
        ...


class Dataset(DatasetView):
    """
    Datasets are the primary data structure used in DeepLake. They are used to store and manage data for searching, training, evaluation.

    Unlike [deeplake.ReadOnlyDataset][], instances of `Dataset` can be modified.
    """

    def __repr__(self) -> str: ...

    def tag(self, name: str, version: str | None = None) -> Tag:
        """
        Tags a version of the dataset. If no version is given, the current version is tagged.

        Parameters:
            name: The name of the tag
            version: The version of the dataset to tag
        """
        ...

    @property
    def tags(self) -> Tags:
        """
        The collection of [deeplake.Tag][]s within the dataset
        """

    name: str
    """
    The name of the dataset. Setting the value will immediately persist the change without requiring a commit().
    """

    description: str
    """
    The description of the dataset.  Setting the value will immediately persist the change without requiring a commit().
    """

    @property
    def metadata(self) -> Metadata:
        """
        The metadata of the dataset.
        """

    @property
    def created_time(self) -> datetime.datetime:
        """
        When the dataset was created. The value is auto-generated at creation time.
        """

    @property
    def version(self) -> str:
        """
        The currently checked out version of the dataset
        """

    @property
    def history(self) -> History:
        """
        The history of the dataset.
        """

    @property
    def _reader(self) -> storage.Reader:
        """
        A [reader][deeplake.storage.Reader] that can be used to directly access files from the dataset's storage.

        Primarily used for debugging purposes.
        """
        ...

    @property
    def id(self) -> str:
        """
        The unique identifier of the dataset. Value is auto-generated at creation time.
        """

    @typing.overload
    def __getitem__(self, offset: int) -> Row:
        """
        Get a row by offset within the dataset.
        """
        ...

    @typing.overload
    def __getitem__(self, range: slice) -> RowRange:
        """
        Get a range of rows by offset within the dataset.
        """
        ...

    @typing.overload
    def __getitem__(self, column: str) -> Column:
        """
        Get a column by name within the dataset.
        """
        ...

    def __getitem__(self, input: int | slice | str) -> Row | RowRange | Column:
        """
        Returns a subset of data from the Dataset

        The result will depend on the type of value passed to the `[]` operator.

        - `int`: The zero-based offset of the single row to return. Returns a [deeplake.Row][]
        - `slice`: A slice specifying the range of rows to return. Returns a [deeplake.RowRange][]
        - `str`: A string specifying column to return all values from. Returns a [deeplake.Column][]

        Examples:
            >>> row = ds[318]

            >>> rows = ds[931:1038]

            >>> column_data = ds["id"]

        """

    ...

    def __iter__(self) -> typing.Iterator[Row]:
        """
        Row based iteration over the dataset.

        Examples:
            >>> for row in ds:
            >>>     # process row
            >>>     pass

        """
        ...

    def __getstate__(self) -> tuple:
        """Returns a dict that can be pickled and used to restore this dataset.

        Note:
            Pickling a dataset does not copy the dataset, it only saves attributes that can be used to restore the dataset.
        """

    def __setstate__(self, arg0: tuple) -> None:
        """Restores dataset from a pickled state.

        Args:
            arg0 (dict): The pickled state used to restore the dataset.
        """

    def add_column(
            self,
            name: str,
            dtype: types.DataType | str | types.Type | type | typing.Callable,
            format: formats.DataFormat | None = None,
    ) -> None:
        """
        Add a new column to the dataset.

        Any existing rows in the dataset will have a `None` value for the new column

        Args:
            name: The name of the column
            dtype: The type of the column. Possible values include:

                - Values from `deeplake.types` such as "[deeplake.types.Int32][]()"
                - Python types: `str`, `int`, `float`
                - Numpy types: such as `np.int32`
                - A function reference that returns one of the above types
            format (DataFormat, optional): The format of the column, if applicable. Only required when the dtype is [deeplake.types.DataType][].

        Examples:
            >>> ds.add_column("labels", deeplake.types.Int32)

            >>> ds.add_column("labels", "int32")

            >>> ds.add_column("name", deeplake.types.Text())

            >>> ds.add_column("json_data", deeplake.types.Dict())

            >>> ds.add_column("images", deeplake.types.Image(dtype=deeplake.types.UInt8(), sample_compression="jpeg"))

            >>> ds.add_column("embedding", deeplake.types.Embedding(dtype=deeplake.types.Float32(), dimensions=768))

        Raises:
            deeplake.ColumnAlreadyExistsError: If a column with the same name already exists.
        """

    def remove_column(
            self,
            name: str
    ) -> None:
        """
        Remove the existing column from the dataset.

        Args:
            name: The name of the column to remove

        Examples:
            >>> ds.remove_column("name")

        Raises:
            deeplake.ColumnDoesNotExistsError: If a column with the specified name does not exists.
        """

    def rename_column(
            self,
            name: str,
            new_name: str
    ) -> None:
        """
        Renames the existing column in the dataset.

        Args:
            name: The name of the column to rename
            new_name: The new name to set to column

        Examples:
            >>> ds.rename_column("old_name", "new_name")

        Raises:
            deeplake.ColumnDoesNotExistsError: If a column with the specified name does not exists.
            deeplake.ColumnAlreadyExistsError: If a column with the specified new name already exists.
        """

    @typing.overload
    def append(self, data: list[dict[str, typing.Any]]) -> None: ...

    @typing.overload
    def append(self, data: dict[str, typing.Any]) -> None: ...

    @typing.overload
    def append(self, data: DatasetView) -> None: ...

    def append(
            self, data: list[dict[str, typing.Any]] | dict[str, typing.Any] | DatasetView
    ) -> None:
        """
        Adds data to the dataset.

        The data can be in a variety of formats:

        - A list of dictionaries, each value in the list is a row, with the dicts containing the column name and its value for the row.
        - A dictionary, the keys are the column names and the values are array-like (list or numpy.array) objects corresponding to the column values.
        - A DatasetView that was generated through any mechanism

        Args:
            data: The data to insert into the dataset.

        Examples:
            >>> ds.append({"name": ["Alice", "Bob"], "age": [25, 30]})

            >>> ds.append([{"name": "Alice", "age": 25}, {"name": "Bob", "age": 30}])

            >>> ds.append({
            >>>     "embedding": np.random.rand(4, 768),
            >>>     "text": ["Hello World"] * 4})

            >>> ds.append([{"embedding": np.random.rand(768), "text": "Hello World"}] * 4)

            >>> ds.append(deeplake.from_parquet("./file.parquet"))

        Raises:
            deeplake.ColumnMissingAppendValueError: If any column is missing from the input data.
            deeplake.UnevenColumnsError: If the input data columns are not the same length.
            deeplake.InvalidTypeDimensions: If the input data does not match the column's dimensions.
        """
        ...

    def delete(self, offset: int) -> None:
        """
        Delete a row from the dataset.

        Parameters:
            offset: The offset of the row within the dataset to delete
        """

    def commit(self, message: str | None = None) -> None:
        """
        Atomically commits changes you have made to the dataset. After commit, other users will see your changes to the dataset.

        Parameters:
            message (str, optional): A message to store in history describing the changes made in the version

        Examples:
            >>> ds.commit()

            >>> ds.commit("Added data from updated documents")

        """

    def commit_async(self, message: str | None = None) -> FutureVoid:
        """
        Asynchronously commits changes you have made to the dataset.

        See [deeplake.Dataset.commit][] for more information.

        Parameters:
            message (str, optional): A message to store in history describing the changes made in the commit

        Examples:
            >>> ds.commit_async().wait()

            >>> ds.commit_async("Added data from updated documents").wait()

            >>> await ds.commit_async()

            >>> await ds.commit_async("Added data from updated documents")

            >>> future = ds.commit_async() # then you can check if the future is completed using future.is_completed()

        """

    def rollback(self) -> None:
        """
        Reverts any in-progress changes to the dataset you have made. Does not revert any changes that have been committed.
        """

    def rollback_async(self) -> FutureVoid:
        """
        Asynchronously reverts any in-progress changes to the dataset you have made. Does not revert any changes that have been committed.
        """

    def push(self, url: str, creds: dict[str, str] | None = None, token: str | None = None) -> None:
        """
        Pushes any new history from this dataset to the dataset at the given url

        Similar to [deeplake.Dataset.pull][] but the other direction.

        Parameters:
            url: The URL of the destination dataset
            creds: Optional credentials needed to connect to the dataset
            token: Optional deeplake token
        """
        ...
    def push_async(self, url: str, creds: dict[str, str] | None = None, token: str | None = None) -> FutureVoid:
        """
        Asynchronously Pushes new any history from this dataset to the dataset at the given url

        Similar to [deeplake.Dataset.pull_async][] but the other direction.

        Parameters:
            url: The URL of the destination dataset
            creds: Optional credentials needed to connect to the dataset
            token: Optional deeplake token
        """
        ...

    def pull(self, url: str, creds: dict[str, str] | None = None, token: str | None = None) -> None:
        """
        Pulls any new history from the dataset at the passed url into this dataset.

        Similar to [deeplake.Dataset.push][] but the other direction.

        Parameters:
            url: The URL of the destination dataset
            creds: Optional credentials needed to connect to the dataset
            token: Optional deeplake token
        """
        ...
    def pull_async(self, url: str, creds: dict[str, str] | None = None, token: str | None = None) -> FutureVoid:
        """
        Asynchronously pulls any new history from the dataset at the passed url into this dataset.

        Similar to [deeplake.Dataset.push_async][] but the other direction.

        Parameters:
            url: The URL of the destination dataset
            creds: Optional credentials needed to connect to the dataset
            token: Optional deeplake token
        """
        ...


    @property
    def history(self) -> History:
        """
        This dataset's version history
        """

    @property
    def schema(self) -> Schema:
        """
        The schema of the dataset.
        """
        ...


class ReadOnlyDataset(DatasetView):
    @typing.overload
    def __getitem__(self, offset: int) -> RowView:
        """
        Get a row by offset within the dataset.
        """
        ...

    @typing.overload
    def __getitem__(self, range: slice) -> RowRangeView:
        """
        Get a range of rows by offset within the dataset.
        """
        ...

    @typing.overload
    def __getitem__(self, column: str) -> ColumnView:
        """
        Get a column by name within the dataset.
        """
        ...

    def __getitem__(
            self, input: int | slice | str
    ) -> RowView | RowRangeView | ColumnView:
        """
        Returns a subset of data from the dataset.

        The result will depend on the type of value passed to the `[]` operator.

        - `int`: The zero-based offset of the single row to return. Returns a [deeplake.RowView][]
        - `slice`: A slice specifying the range of rows to return. Returns a [deeplake.RowRangeView][]
        - `str`: A string specifying column to return all values from. Returns a [deeplake.ColumnView][]

        Examples:
            >>> row = ds[318]

            >>> rows = ds[931:1038]

            >>> column_data = ds["id"]

        """
        ...

    def __iter__(self) -> typing.Iterator[RowView]:
        """
        Row based iteration over the dataset.

        Examples:
            >>> for row in ds:
            >>>     # process row
            >>>     pass

        """
        ...

    def __repr__(self) -> str: ...

    @property
    def tags(self) -> TagsView:
        """
        The collection of [deeplake.TagView][]s within the dataset
        """
        ...

    @property
    def created_time(self) -> datetime.datetime:
        """
        When the dataset was created. The value is auto-generated at creation time.
        """
        ...

    @property
    def description(self) -> str:
        """
        The description of the dataset
        """
        ...

    @property
    def metadata(self) -> ReadOnlyMetadata:
        """
        The metadata of the dataset.
        """
        ...

    @property
    def version(self) -> str:
        """
        The currently checked out version of the dataset
        """

    @property
    def history(self) -> History:
        """
        The history of the overall dataset configuration.
        """
        ...

    @property
    def _reader(self) -> storage.Reader:
        """
        A [reader][deeplake.storage.Reader] that can be used to directly access files from the dataset's storage.

        Primarily used for debugging purposes.
        """
        ...

    @property
    def id(self) -> str:
        """
        The unique identifier of the dataset. Value is auto-generated at creation time.
        """
        ...

    @property
    def name(self) -> str:
        """
        The name of the dataset.
        """
        ...

    @property
    def schema(self) -> SchemaView:
        """
        The schema of the dataset.
        """
        ...

    def push(self, url: str, creds: dict[str, str] | None = None, token: str | None = None) -> None:
        """
        Pushes any history from this dataset to the dataset at the given url

        Parameters:
            url: The URL of the destination dataset
            creds: Optional credentials needed to connect to the dataset
            token: Optional deeplake token
        """
        ...
    def push_async(self, url: str, creds: dict[str, str] | None = None, token: str | None = None) -> FutureVoid:
        """
        Asynchronously Pushes any history from this dataset to the dataset at the given url

        Parameters:
            url: The URL of the destination dataset
            creds: Optional credentials needed to connect to the dataset
            token: Optional deeplake token
        """
        ...

    def __getstate__(self) -> tuple:
        """Returns a dict that can be pickled and used to restore this dataset.

        Note:
            Pickling a dataset does not copy the dataset, it only saves attributes that can be used to restore the dataset.
        """

    def __setstate__(self, state: tuple) -> None:
        """Restores dataset from a pickled state.

        Args:
            state (dict): The pickled state used to restore the dataset.
        """

class ExpiredTokenError(Exception):
    pass


class FormatNotSupportedError(Exception):
    pass


class UnevenColumnsError(Exception):
    pass


class UnevenUpdateError(Exception):
    pass


class ColumnMissingAppendValueError(Exception):
    pass


class ColumnAlreadyExistsError(Exception):
    pass


class ColumnDoesNotExistError(Exception):
    pass


class InvalidColumnValueError(Exception):
    pass


class GcsStorageProviderFailed(Exception):
    pass


class History:
    """
    The version history of a [deeplake.Dataset][].
    """

    @typing.overload
    def __getitem__(self, offset: int) -> Version:
        ...
    @typing.overload
    def __getitem__(self, version: str) -> Version:
        ...

    def __getitem__(self, input: int | str) -> Version:
        ...

    def __iter__(self) -> typing.Iterator[Version]:
        """
        Iterate over the history, starting at the initial version
        """
        ...

    def __len__(self) -> int:
        """
        The number of versions within the history
        """
        ...

    def __repr__(self) -> str: ...


class InvalidType(Exception):
    pass


class LogExistsError(Exception):
    pass


class LogNotexistsError(Exception):
    pass


class IncorrectDeeplakePathError(Exception):
    pass


class AuthenticationError(Exception):
    pass


class AuthorizationError(Exception):
    pass


class NotFoundError(Exception):
    pass


class AgreementError(Exception):
    pass


class AgreementNotAcceptedError(Exception):
    pass


class NotLoggedInAgreementError(Exception):
    pass


class JSONKeyNotFound(Exception):
    pass


class JSONIndexNotFound(Exception):
    pass


class UnknownFormat(Exception):
    pass


class UnknownStringType(Exception):
    pass


class InvalidChunkStrategyType(Exception):
    pass


class InvalidSequenceOfSequence(Exception):
    pass


class InvalidTypeAndFormatPair(Exception):
    pass


class UnknownType(Exception):
    pass


class InvalidTextType(Exception):
    pass


class UnsupportedPythonType(Exception):
    pass


class UnsupportedSampleCompression(Exception):
    pass


class UnsupportedChunkCompression(Exception):
    pass


class InvalidImageCompression(Exception):
    pass


class InvalidMaskCompression(Exception):
    pass


class DtypeMismatch(Exception):
    pass


class UnspecifiedDtype(Exception):
    pass


class DimensionsMismatch(Exception):
    pass


class ShapeIndexOutOfChunk(Exception):
    pass


class BytePositionIndexOutOfChunk(Exception):
    pass


class TensorAlreadyExists(Exception):
    pass


class CanNotCreateTensorWithProvidedCompressions(Exception):
    pass


class WrongChunkCompression(Exception):
    pass


class WrongSampleCompression(Exception):
    pass


class UnknownBoundingBoxCoordinateFormat(Exception):
    pass


class UnknownBoundingBoxPixelFormat(Exception):
    pass


class InvalidTypeDimensions(Exception):
    pass


class SchemaView:
    """
    A read-only view of a [deeplake.Dataset][] [deeplake.Schema][].
    """

    def __getitem__(self, column: str) -> ColumnDefinitionView:
        """
        Return the column definition by name
        """
        ...

    def __len__(self) -> int:
        """
        The number of columns within the schema
        """
        ...

    @property
    def columns(self) -> list[ColumnDefinitionView]:
        """
        A list of all columns within the schema
        """
        ...

    def __repr__(self) -> str: ...


class Schema:
    """
    The schema of a [deeplake.Dataset][].
    """

    def __getitem__(self, column: str) -> ColumnDefinition:
        """
        Return the column definition by name
        """
        ...

    def __len__(self) -> int:
        """
        The number of columns within the schema
        """
        ...

    @property
    def columns(self) -> list[ColumnDefinition]:
        """
        A list of all columns within the schema
        """
        ...

    def __repr__(self) -> str: ...


class StorageAccessDenied(Exception):
    pass


class StorageKeyAlreadyExists(Exception):
    pass


class StorageKeyNotFound(Exception):
    pass


class StorageNetworkConnectionError(Exception):
    pass


class StorageInternalError(Exception):
    pass


class WriteFailedError(Exception):
    pass


def create(url: str, creds: dict[str, str] | None = None, token: str | None = None,
           schema: schemas.SchemaTemplate | None = None) -> Dataset:
    """
    Creates a new dataset at the given URL.

    To open an existing dataset, use [deeplake.open][]

    Args:
        url: The URL of the dataset.
            URLs can be specified using the following protocols:

            - `file://path` local filesystem storage
            - `al://org_id/dataset_name` A dataset on app.activeloop.ai
            - `azure://bucket/path` or `az://bucket/path` Azure storage
            - `gs://bucket/path` or `gcs://bucket/path` or `gcp://bucket/path` Google Cloud storage
            - `s3://bucket/path` S3 storage
            - `mem://name` In-memory storage that lasts the life of the process

            A URL without a protocol is assumed to be a file:// URL

        creds (dict, str, optional): The string ``ENV`` or a dictionary containing credentials used to access the dataset at the path.

            - If 'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token' are present, these take precedence over credentials present in the environment or in credentials file. Currently only works with s3 paths.
            - It supports 'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token', 'endpoint_url', 'aws_region', 'profile_name' as keys.
            - To use credentials managed in your Activeloop organization, use they key 'creds_key': 'managed_key_name'. This requires the org_id dataset argument to be set.
            - If nothing is given is, credentials are fetched from the environment variables. This is also the case when creds is not passed for cloud datasets
        token (str, optional): Activeloop token, used for fetching credentials to the dataset at path if it is a Deep Lake dataset. This is optional, tokens are normally autogenerated.
        schema (dict): The initial schema to use for the dataset. See `deeplake.schema` such as [deeplake.schemas.TextEmbeddings][] for common starting schemas.

    Examples:
        >>> import deeplake
        >>> from deeplake import types
        >>>
        >>> # Create a dataset in your local filesystem:
        >>> ds = deeplake.create("directory_path")
        >>> ds.add_column("id", types.Int32())
        >>> ds.add_column("url", types.Text())
        >>> ds.add_column("embedding", types.Embedding(768))
        >>> ds.commit()
        >>> ds.summary()
        Dataset(columns=(id,url,embedding), length=0)
        +---------+-------------------------------------------------------+
        | column  |                         type                          |
        +---------+-------------------------------------------------------+
        |   id    |               kind=generic, dtype=int32               |
        +---------+-------------------------------------------------------+
        |   url   |                         text                          |
        +---------+-------------------------------------------------------+
        |embedding|kind=embedding, dtype=array(dtype=float32, shape=[768])|
        +---------+-------------------------------------------------------+


        >>> # Create dataset in your app.activeloop.ai organization:
        >>> ds = deeplake.create("al://organization_id/dataset_name")

        >>> # Create a dataset stored in your cloud using specified credentials:
        >>> ds = deeplake.create("s3://mybucket/my_dataset",
        >>>     creds = {"aws_access_key_id": ..., ...})

        >>> # Create dataset stored in your cloud using app.activeloop.ai managed credentials.
        >>> ds = deeplake.create("s3://mybucket/my_dataset",
        >>>     creds = {"creds_key": "managed_creds_key"}, org_id = "my_org_id")

        >>> # Create dataset stored in your cloud using app.activeloop.ai managed credentials.
        >>> ds = deeplake.create("azure://bucket/path/to/dataset")

        >>> ds = deeplake.create("gcs://bucket/path/to/dataset")

        >>> ds = deeplake.create("mem://in-memory")


    Raises:
        ValueError: if a dataset already exists at the given URL
    """


def create_async(url: str, creds: dict[str, str] | None = None, token: str | None = None,
                 schema: schemas.SchemaTemplate | None = None) -> Future:
    """
    Asynchronously creates a new dataset at the given URL.

    See [deeplake.create][] for more information.

    To open an existing dataset, use [deeplake.open_async][].
    
    Examples:
        >>> import deeplake
        >>> from deeplake import types
        >>>
        >>> # Asynchronously create a dataset in your local filesystem:
        >>> ds = await deeplake.create_async("directory_path")
        >>> await ds.add_column("id", types.Int32())
        >>> await ds.add_column("url", types.Text())
        >>> await ds.add_column("embedding", types.Embedding(768))
        >>> await ds.commit()
        >>> await ds.summary()  # Example of usage in an async context

        >>> # Alternatively, create a dataset using .result().
        >>> future_ds = deeplake.create_async("directory_path")
        >>> ds = future_ds.result()  # Blocks until the dataset is created

        >>> # Create a dataset in your app.activeloop.ai organization:
        >>> ds = await deeplake.create_async("al://organization_id/dataset_name")

        >>> # Create a dataset stored in your cloud using specified credentials:
        >>> ds = await deeplake.create_async("s3://mybucket/my_dataset",
        >>>     creds={"aws_access_key_id": ..., ...})

        >>> # Create dataset stored in your cloud using app.activeloop.ai managed credentials.
        >>> ds = await deeplake.create_async("s3://mybucket/my_dataset",
        >>>     creds={"creds_key": "managed_creds_key"}, org_id="my_org_id")

        >>> # Create dataset stored in your cloud using app.activeloop.ai managed credentials.
        >>> ds = await deeplake.create_async("azure://bucket/path/to/dataset")

        >>> ds = await deeplake.create_async("gcs://bucket/path/to/dataset")

        >>> ds = await deeplake.create_async("mem://in-memory")

    Raises:
        ValueError: if a dataset already exists at the given URL (will be raised when the future is awaited)
    """

def copy(src: str, dst: str, src_creds: dict[str, str] | None = None, dst_creds: dict[str, str] | None = None,
             token: str | None = None,) -> None:
    """
    Copies the dataset at the source URL to the destination URL.

    NOTE: Currently private due to potential issues in file timestamp handling

    Args:
        src (str): The URL of the source dataset.
        dst (str): The URL of the destination dataset.
        src_creds (dict, str, optional): The string ``ENV`` or a dictionary containing credentials used to access the source dataset at the path.
        dst_creds (dict, str, optional): The string ``ENV`` or a dictionary containing credentials used to access the destination dataset at the path.
        token (str, optional): Activeloop token, used for fetching credentials to the dataset at path if it is a Deep Lake dataset. This is optional, tokens are normally autogenerated.

    Examples:
        >>> deeplake.copy("al://organization_id/source_dataset", "al://organization_id/destination_dataset")

    """


def delete(
        url: str,
        creds: dict[str, str] | None = None,
        token: str | None = None) -> None:
    """
    Deletes an existing dataset.

    !!! warning
        This operation is irreversible. All data will be lost.

        If concurrent processes are attempting to write to the dataset while it's being deleted, it may lead to data inconsistency.
        It's recommended to use this operation with caution.
    """

def exists(
        url: str, creds: dict[str, str] | None = None, token: str | None = None
) -> bool:
    """
    Check if a dataset exists at the given URL

    Args:
        url: URL of the dataset
        creds: The string ``ENV`` or a dictionary containing credentials used to access the dataset at the path.
        token: Activeloop token, used for fetching credentials to the dataset at path if it is a Deep Lake dataset. This is optional, tokens are normally autogenerated.
    """

def open(
        url: str, creds: dict[str, str] | None = None, token: str | None = None
) -> Dataset:
    """
    Opens an existing dataset, potenitally for modifying its content.

    See [deeplake.open_read_only][] for opening the dataset in read only mode

    To create a new dataset, see [deeplake.open][]

    Args:
        url: The URL of the dataset. URLs can be specified using the following protocols:

            - `file://path` local filesystem storage
            - `al://org_id/dataset_name` A dataset on app.activeloop.ai
            - `azure://bucket/path` or `az://bucket/path` Azure storage
            - `gs://bucket/path` or `gcs://bucket/path` or `gcp://bucket/path` Google Cloud storage
            - `s3://bucket/path` S3 storage
            - `mem://name` In-memory storage that lasts the life of the process

             A URL without a protocol is assumed to be a file:// URL

        creds (dict, str, optional): The string ``ENV`` or a dictionary containing credentials used to access the dataset at the path.

            - If 'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token' are present, these take precedence over credentials present in the environment or in credentials file. Currently only works with s3 paths.
            - It supports 'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token', 'endpoint_url', 'aws_region', 'profile_name' as keys.
            - To use credentials managed in your Activeloop organization, use they key 'creds_key': 'managed_key_name'. This requires the org_id dataset argument to be set.
            - If nothing is given is, credentials are fetched from the environment variables. This is also the case when creds is not passed for cloud datasets
        token (str, optional): Activeloop token, used for fetching credentials to the dataset at path if it is a Deep Lake dataset. This is optional, tokens are normally autogenerated.

    Examples:
        >>> # Load dataset managed by Deep Lake.
        >>> ds = deeplake.open("al://organization_id/dataset_name")

        >>> # Load dataset stored in your cloud using your own credentials.
        >>> ds = deeplake.open("s3://bucket/my_dataset",
        >>>     creds = {"aws_access_key_id": ..., ...})

        >>> # Load dataset stored in your cloud using Deep Lake managed credentials.
        >>> ds = deeplake.open("s3://bucket/my_dataset",
        >>>     ...creds = {"creds_key": "managed_creds_key"}, org_id = "my_org_id")

        >>> ds = deeplake.open("s3://bucket/path/to/dataset")

        >>> ds = deeplake.open("azure://bucket/path/to/dataset")

        >>> ds = deeplake.open("gcs://bucket/path/to/dataset")

    """

def open_async(
    url: str, creds: dict[str, str] | None = None, token: str | None = None
) -> Future:
    """
    Asynchronously opens an existing dataset, potentially for modifying its content.

    See [deeplake.open][] for opening the dataset synchronously.

    Examples:
        >>> # Asynchronously load dataset managed by Deep Lake using await.
        >>> ds = await deeplake.open_async("al://organization_id/dataset_name")

        >>> # Asynchronously load dataset stored in your cloud using your own credentials.
        >>> ds = await deeplake.open_async("s3://bucket/my_dataset",
        >>>     creds={"aws_access_key_id": ..., ...})

        >>> # Asynchronously load dataset stored in your cloud using Deep Lake managed credentials.
        >>> ds = await deeplake.open_async("s3://bucket/my_dataset",
        >>>     creds={"creds_key": "managed_creds_key"}, org_id="my_org_id")

        >>> ds = await deeplake.open_async("s3://bucket/path/to/dataset")

        >>> ds = await deeplake.open_async("azure://bucket/path/to/dataset")

        >>> ds = await deeplake.open_async("gcs://bucket/path/to/dataset")

        >>> # Alternatively, load the dataset using .result().
        >>> future_ds = deeplake.open_async("al://organization_id/dataset_name")
        >>> ds = future_ds.result()  # Blocks until the dataset is loaded
    """

def like(
        src: DatasetView,
        dest: str,
        creds: dict[str, str] | None = None,
        token: str | None = None,
) -> Dataset:
    """
    Creates a new dataset by copying the ``source`` dataset's structure to a new location.

    !!! note
        No data is copied.

    Args:
        src: The dataset to copy the structure from.
        dest: The URL to create the new dataset at.
                creds (dict, str, optional): The string ``ENV`` or a dictionary containing credentials used to access the dataset at the path.

            - If 'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token' are present, these take precedence over credentials present in the environment or in credentials file. Currently only works with s3 paths.
            - It supports 'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token', 'endpoint_url', 'aws_region', 'profile_name' as keys.
            - To use credentials managed in your Activeloop organization, use they key 'creds_key': 'managed_key_name'. This requires the org_id dataset argument to be set.
            - If nothing is given is, credentials are fetched from the environment variables. This is also the case when creds is not passed for cloud datasets
        token (str, optional): Activeloop token, used for fetching credentials to the dataset at path if it is a Deep Lake dataset. This is optional, tokens are normally autogenerated.

    Examples:
        >>> ds = deeplake.like(src="az://bucket/existing/to/dataset",
        >>>     dest="s3://bucket/new/dataset")

    """


def connect(
        src: str,
        dest: str | None = None,
        org_id: str | None = None,
        creds_key: str | None = None,
        token: str | None = None,
) -> Dataset:
    """
    Connects an existing dataset your [app.activeloop.ai](https://app.activeloop.ai) account.

    Either `dest` or `org_id` is required but not both.

    See [deeplake.disconnect][]

    Args:
        src: The URL to the existing dataset.
        dest (str, optional): Desired Activeloop url for the dataset entry. Example: `al://my_org/dataset`
        org_id (str, optional): The id of the organization to store the dataset under. The dataset name will be based on the source dataset's name.
        creds_key (str, optional): The creds_key of the managed credentials that will be used to access the source path. If not set, use the organization's default credentials.
        token (str, optional): Activeloop token used to fetch the managed credentials.

    Examples:
        >>> ds = deeplake.connect("s3://bucket/path/to/dataset",
        >>>     "al://my_org/dataset")

        >>> ds = deeplake.connect("s3://bucket/path/to/dataset",
        >>>     "al://my_org/dataset", creds_key="my_key")

        >>> # Connect the dataset as al://my_org/dataset
        >>> ds = deeplake.connect("s3://bucket/path/to/dataset",
        >>>     org_id="my_org")

        >>> ds = deeplake.connect("az://bucket/path/to/dataset",
        >>>     "al://my_org/dataset", creds_key="my_key")

        >>> ds = deeplake.connect("gcs://bucket/path/to/dataset",
        >>>     "al://my_org/dataset", creds_key="my_key")

    """


def disconnect(url: str, token: str | None = None) -> None:
    """
    Disconnect the dataset your Activeloop account.

    See [deeplake.connect][]

    !!! note
        Does not delete the stored data, it only removes the connection from the activeloop organization

    Args:
        url: The URL of the dataset.
        token (str, optional): Activeloop token to authenticate user.

    Examples:
        >>> deeplake.disconnect("al://my_org/dataset_name")

    """


def open_read_only(
        url: str, creds: dict[str, str] | None = None, token: str | None = None
) -> ReadOnlyDataset:
    """
    Opens an existing dataset in read-only mode.

    See [deeplake.open][] for opening datasets for modification.

    Args:
        url: The URL of the dataset.

            URLs can be specified using the following protocols:

            - `file://path` local filesystem storage
            - `al://org_id/dataset_name` A dataset on app.activeloop.ai
            - `azure://bucket/path` or `az://bucket/path` Azure storage
            - `gs://bucket/path` or `gcs://bucket/path` or `gcp://bucket/path` Google Cloud storage
            - `s3://bucket/path` S3 storage
            - `mem://name` In-memory storage that lasts the life of the process

            A URL without a protocol is assumed to be a file:// URL

        creds (dict, str, optional): The string ``ENV`` or a dictionary containing credentials used to access the dataset at the path.

            - If 'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token' are present, these take precedence over credentials present in the environment or in credentials file. Currently only works with s3 paths.
            - It supports 'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token', 'endpoint_url', 'aws_region', 'profile_name' as keys.
            - To use credentials managed in your Activeloop organization, use they key 'creds_key': 'managed_key_name'. This requires the org_id dataset argument to be set.
            - If nothing is given is, credentials are fetched from the environment variables. This is also the case when creds is not passed for cloud datasets
        token (str, optional): Activeloop token to authenticate user.

    Examples:
        >>> ds = deeplake.open_read_only("directory_path")
        >>> ds.summary()
        Dataset(columns=(id,url,embedding), length=0)
        +---------+-------------------------------------------------------+
        | column  |                         type                          |
        +---------+-------------------------------------------------------+
        |   id    |               kind=generic, dtype=int32               |
        +---------+-------------------------------------------------------+
        |   url   |                         text                          |
        +---------+-------------------------------------------------------+
        |embedding|kind=embedding, dtype=array(dtype=float32, shape=[768])|
        +---------+-------------------------------------------------------+


        >>> ds = deeplake.open_read_only("file:///path/to/dataset")

        >>> ds = deeplake.open_read_only("s3://bucket/path/to/dataset")

        >>> ds = deeplake.open_read_only("azure://bucket/path/to/dataset")

        >>> ds = deeplake.open_read_only("gcs://bucket/path/to/dataset")

        >>> ds = deeplake.open_read_only("mem://in-memory")

    """

def open_read_only_async(
    url: str, creds: dict[str, str] | None = None, token: str | None = None
) -> Future:
    """
    Asynchronously opens an existing dataset in read-only mode.

    See [deeplake.open_async][] for opening datasets for modification and [deeplake.open_read_only][] for sync open.

    Examples:
        >>> # Asynchronously open a dataset in read-only mode:
        >>> ds = await deeplake.open_read_only_async("directory_path")

        >>> # Alternatively, open the dataset using .result().
        >>> future_ds = deeplake.open_read_only_async("directory_path")
        >>> ds = future_ds.result()  # Blocks until the dataset is loaded

        >>> ds = await deeplake.open_read_only_async("file:///path/to/dataset")

        >>> ds = await deeplake.open_read_only_async("s3://bucket/path/to/dataset")

        >>> ds = await deeplake.open_read_only_async("azure://bucket/path/to/dataset")

        >>> ds = await deeplake.open_read_only_async("gcs://bucket/path/to/dataset")

        >>> ds = await deeplake.open_read_only_async("mem://in-memory")
    """

def from_parquet(url: str) -> ReadOnlyDataset:
    """
    Opens a Parquet dataset in the deeplake format.

    Args:
        url: The URL of the Parquet dataset. If no protocol is specified, it assumes `file://`
    """


def __child_atfork() -> None:
    ...


def __parent_atfork() -> None:...



def __prepare_atfork() -> None:...
