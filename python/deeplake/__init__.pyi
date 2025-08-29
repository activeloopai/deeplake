import datetime
import typing
import pathlib

import storage
from . import schemas
from . import types

T = typing.TypeVar('T')
DatasetType = typing.TypeVar('DatasetType', bound='Dataset')

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

class Future(typing.Generic[T]):
    """
    A future representing an asynchronous operation result in ML pipelines.

    The Future class enables non-blocking operations for data loading and processing,
    particularly useful when working with large ML datasets or distributed training.
    Once resolved, the Future holds the operation result which can be accessed either
    synchronously or asynchronously.

    Methods:
        result() -> typing.Any:
            Blocks until the Future resolves and returns the result.

        __await__() -> typing.Any:
            Enables using the Future in async/await syntax.

        cancel() -> None:
            Cancels the Future if it is still pending.

        is_completed() -> bool:
            Checks if the Future has resolved without blocking.
    <!-- test-context
    ```python
    import deeplake
    ds = deeplake.create("mem://ml-data/embeddings")
    ds = deeplake.create("mem://ml-data/images")
    ds.add_column("images", "int32")
    ds.append({"images": [0] * 300})
    deeplake.open_async = lambda x: deeplake._deeplake.open_async(x.replace("s3://", "mem://"))
    ```
    -->

    Examples:
        Loading ML dataset asynchronously:
        ```python
        future = deeplake.open_async("s3://ml-data/embeddings")

        # Check status without blocking
        if not future.is_completed():
            print("Still loading...")

        # Block until ready
        ds = future.result()
        ```

        Using with async/await:
        ```python
        async def load_data():
            ds = await deeplake.open_async("s3://ml-data/images")
            batch = await ds["images"].get_async(slice(0, 32))
            return batch
        ```
    """

    def result(self) -> T:
        """
        Blocks until the Future resolves and returns the result.

        Returns:
            typing.Any: The operation result once resolved.

        <!-- test-context
        ```python
        import deeplake
        import numpy as np
        ds = deeplake.create("tmp://")
        ds.add_column("images", "int32")
        ds.append({"images": [0] * 300})
        ```
        -->

        Examples:
        ```python
        future = ds["images"].get_async(slice(0, 32))
        batch = future.result()  # Blocks until batch is loaded
        ```
        """
        ...

    def __await__(self) -> typing.Any:
        """
        Makes the Future compatible with async/await syntax.

        <!-- test-context
        ```python
        import deeplake
        import numpy as np
        ds = deeplake.create("tmp://")
        ds.add_column("images", "int32")
        ds.append({"images": [0] * 300})
        ```
        -->

        Examples:
        ```python
        async def load_batch():
            batch = await ds["images"].get_async(slice(0, 32))
        ```

        Returns:
            typing.Any: The operation result once resolved.
        """
        ...

    def is_completed(self) -> bool:
        """
        Checks if the Future has resolved without blocking.

        Returns:
            bool: True if resolved, False if still pending.

        <!-- test-context
        ```python
        import deeplake
        import numpy as np
        ds = deeplake.create("tmp://")
        ds.add_column("label", "int32")
        ds.append({"label": [0] * 300})
        ds["label"].metadata["class_names"] = ["car", "truck", "bus"]
        ```
        -->

        Examples:
        ```python
        future = ds.query_async("SELECT * WHERE label = 'car'")
        if future.is_completed():
            results = future.result()
        else:
            print("Query still running...")
        ```
        """
        ...

    def cancel(self) -> None:
        """
        Cancels the Future if it is still pending.
        """

class FutureVoid:
    """
    A Future representing a void async operation in ML pipelines.

    Similar to Future but for operations that don't return values, like saving
    or committing changes. Useful for non-blocking data management operations.

    Methods:
        wait() -> None:
            Blocks until operation completes.

        __await__() -> typing.Any:
            Enables using with async/await syntax.

        is_completed() -> bool:
            Checks completion status without blocking.

        cancel() -> None:
            Cancels the Future if still pending.

    <!-- test-context
    ```python
    import deeplake
    ds = deeplake.create("tmp://")
    ds.add_column("embeddings", "float32")
    ds.append({"embeddings": [0.1] * 100})
    new_embeddings = [0.2] * 32
    def process_other_data():
        pass
    ```
    -->

    Examples:
        Asynchronous dataset updates:
        ```python
        # Update embeddings without blocking
        future = ds["embeddings"].set_async(slice(0, 32), new_embeddings)

        # Do other work while update happens
        process_other_data()

        # Wait for update to complete
        future.wait()
        ```

        Using with async/await:
        ```python
        async def update_dataset():
            await ds.commit_async()
            print("Changes saved")
        ```
    """

    def wait(self) -> None:
        """
        Blocks until the operation completes.

        <!-- test-context
        ```python
        import deeplake
        ds = deeplake.create("tmp://")
        ```
        -->

        Examples:
            ```python
            future = ds.commit_async()
            future.wait()  # Blocks until commit finishes
            ```
        """
        ...

    def __await__(self) -> typing.Any:
        """
        Makes the FutureVoid compatible with async/await syntax.

        <!-- test-context
        ```python
        import deeplake
        ds = deeplake.create("tmp://")
        ```
        -->

        Examples:
        ```python
        async def save_changes():
            await ds.commit_async()
        ```
        """
        ...

    def cancel(self) -> None:
        """
        Cancels the Future if it is still pending.
        """

    def is_completed(self) -> bool:
        """
        Checks if the operation has completed without blocking.

        Returns:
            bool: True if completed, False if still running.

        <!-- test-context
        ```python
        import deeplake
        ds = deeplake.create("tmp://")
        ```
        -->

        Examples:
            ```python
            future = ds.commit_async()
            if future.is_completed():
                print("Commit finished")
            else:
                print("Commit still running...")
            ```
        """
        ...

class Array:
    """
    Wrapper around n dimensional array
    """

    @property
    def dtype(self) -> numpy.dtype[typing.Any]:
        """
        Returns the data type of the array
        """
        ...

    @property
    def shape(self) -> tuple:
        """
        Returns the shape of the array
        """
        ...

    def __getitem__(self, index: int | slice | list | tuple ) -> typing.Any:
        """
        Returns the value at the given index or slice
        """
        ...

    def get_async(self, index: int | slice | list | tuple) -> Future[typing.Any]:
        """
        Returns the value at the given index or slice asynchronously
        """
        ...

    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

class ReadOnlyMetadata:
    """
    Read-only access to dataset and column metadata for ML workflows.

    Stores important information about datasets like:
    - Model parameters and hyperparameters
    - Preprocessing statistics (mean, std, etc.)
    - Data splits and fold definitions
    - Version and training information

    <!-- test-context
    ```python
    import deeplake
    ds = deeplake.create("tmp://")
    ds.add_column("images", "int32")
    ds.metadata["model_name"] = "resnet50"
    ds.metadata["hyperparameters"] = {"learning_rate": 0.001, "batch_size": 32}
    ds["images"].metadata["mean"] = [0.485, 0.456, 0.406]
    ds["images"].metadata["std"] = [0.229, 0.224, 0.225]
    ```
    -->

    Examples:
        Accessing model metadata:
        ```python
        metadata = ds.metadata
        model_name = metadata["model_name"]
        model_params = metadata["hyperparameters"]
        ```

        Reading preprocessing stats:
        ```python
        mean = ds["images"].metadata["mean"]
        std = ds["images"].metadata["std"]
        ```
    """

    def __getitem__(self, key: str) -> typing.Any:
        """
        Gets metadata value for the given key.

        Args:
            key: Metadata key to retrieve

        Returns:
            The stored metadata value

        <!-- test-context
        ```python
        import deeplake
        ds = deeplake.create("tmp://")
        ds.add_column("images", "int32")
        ds.metadata["model_name"] = "resnet50"
        ds.metadata["hyperparameters"] = {"learning_rate": 0.001, "batch_size": 32}
        ds["images"].metadata["mean"] = [0.485, 0.456, 0.406]
        ds["images"].metadata["std"] = [0.229, 0.224, 0.225]
        ```
        -->

        Examples:
            ```python
            mean = ds["images"].metadata["mean"]
            std = ds["images"].metadata["std"]
            ```
        """
        ...

    def keys(self) -> list[str]:
        """
        Lists all available metadata keys.

        Returns:
            list[str]: List of metadata key names

        <!-- test-context
        ```python
        import deeplake
        ds = deeplake.create("tmp://")
        ds.add_column("images", "int32")
        metadata = ds.metadata
        ```
        -->

        Examples:
            ```python
            # Print all metadata
            for key in metadata.keys():
                print(f"{key}: {metadata[key]}")
            ```
        """
        ...

    def __contains__(self, key: str) -> bool:
        """
        Checks if the metadata contains the given key.
        """
        ...

class Metadata(ReadOnlyMetadata):
    """
    Writable access to dataset and column metadata for ML workflows.

    Stores important information about datasets like:

    - Model parameters and hyperparameters
    - Preprocessing statistics
    - Data splits and fold definitions
    - Version and training information

    Changes are persisted immediately without requiring `commit()`.

    <!-- test-context
    ```python
    import deeplake
    ds = deeplake.create("tmp://")
    ds.add_column("images", "int32")
    ```
    -->

    Examples:
        Storing model metadata:
        ```python
        ds.metadata["model_name"] = "resnet50"
        ds.metadata["hyperparameters"] = {
            "learning_rate": 0.001,
            "batch_size": 32
        }
        ```

        Setting preprocessing stats:
        ```python
        ds["images"].metadata["mean"] = [0.485, 0.456, 0.406]
        ds["images"].metadata["std"] = [0.229, 0.224, 0.225]
        ```
    """

    def __setitem__(self, key: str, value: typing.Any) -> None:
        """
        Sets metadata value for given key. Changes are persisted immediately.

        Args:
            key: Metadata key to set
            value: Value to store

        Examples:
            ```python
            ds.metadata["train_split"] = 0.8
            ds.metadata["val_split"] = 0.1
            ds.metadata["test_split"] = 0.1
            ```
        """
        ...

class ExplainQueryResult:
    def __str__(self) -> str:
        ...
    def to_dict(self) -> typing.Any:
        ...

def prepare_query(query: str, token: str | None = None, creds: dict[str, str] | None = None) -> Executor:
    """
    Prepares a TQL query for execution with optional authentication.

    Args:
        query: TQL query string to execute
        token: Optional Activeloop authentication token
        creds (dict, optional): Dictionary containing credentials used to access the dataset at the path.

    Returns:
        Executor: An executor object to run the query.

    <!-- test-context
    ```python
    import deeplake
    ds = deeplake.create("mem://parametriized")
    ds.add_column("category", "text")
    ds.append({"category": ["active", "inactive", "not sure"]})
    ds.commit()
    ```
    -->

    Examples:
        Running a parametrized batch query:
        ```python
        ex = deeplake.prepare_query('SELECT * FROM "mem://parametriized" WHERE category = ?')
        results = ex.run_batch([["active"], ["inactive"]])
        assert len(results) == 2
        ```
    """
    ...

def query(query: str, token: str | None = None, creds: dict[str, str] | None = None) -> DatasetView:
    """
    Executes TQL queries optimized for ML data filtering and search.

    TQL is a SQL-like query language designed for ML datasets, supporting:

      - Vector similarity search
      - Text semantic search
      - Complex data filtering
      - Joining across datasets
      - Efficient sorting and pagination

    Args:
        query: TQL query string supporting:

          - Vector similarity: COSINE_SIMILARITY, L2_NORM
          - Text search: BM25_SIMILARITY, CONTAINS
          - MAXSIM similarity for ColPali embeddings: MAXSIM
          - Filtering: WHERE clauses
          - Sorting: ORDER BY
          - Joins: JOIN across datasets

        token: Optional Activeloop authentication token
        creds (dict, optional): Dictionary containing credentials used to access the dataset at the path.

          - If 'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token' are present, these take precedence over credentials present in the environment or in credentials file. Currently only works with s3 paths.
          - It supports 'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token', 'endpoint_url', 'aws_region', 'profile_name' as keys.
          - If nothing is given is, credentials are fetched from the environment variables. This is also the case when creds is not passed for cloud datasets

    Returns:
        DatasetView: Query results that can be:

          - Used directly in ML training
          - Further filtered with additional queries
          - Converted to PyTorch/TensorFlow dataloaders
          - Materialized into a new dataset

    <!-- test-context
    ```python
    import deeplake
    ds = deeplake.create("mem://embeddings")
    ds.add_column("vector", deeplake.types.Array("float32", 1))
    ds.commit()
    ds = deeplake.create("mem://documents")
    ds.add_column("text", "text")
    ds.commit()
    ds = deeplake.create("mem://dataset")
    ds.add_column("train_split", "text")
    ds.add_column("confidence", "float32")
    ds.add_column("label", "text")
    ds.commit()
    ds = deeplake.create("mem://images")
    ds.add_column("id", "int32")
    ds.add_column("image", "int32")
    ds.add_column("embedding", deeplake.types.Array("float32", 1))
    ds.commit()
    ds = deeplake.create("mem://metadata")
    ds.add_column("image_id", "int32")
    ds.add_column("labels", "text")
    ds.add_column("metadata", "text")
    ds.add_column("verified", "bool")
    ds.commit()
    ```
    -->

    Examples:
        Vector similarity search:
        ```python
        # Find similar embeddings
        similar = deeplake.query('''
            SELECT * FROM "mem://embeddings"
            ORDER BY COSINE_SIMILARITY(vector, ARRAY[0.1, 0.2, 0.3]) DESC
            LIMIT 100
        ''')

        # Use results in training
        dataloader = similar.pytorch()
        ```

        Text semantic search:
        ```python
        # Search documents using BM25
        relevant = deeplake.query('''
            SELECT * FROM "mem://documents"
            ORDER BY BM25_SIMILARITY(text, 'machine learning') DESC
            LIMIT 10
        ''')
        ```

        Complex filtering:
        ```python
        # Filter training data
        train = deeplake.query('''
            SELECT * FROM "mem://dataset"
            WHERE "train_split" = 'train'
            AND confidence > 0.9
            AND label IN ('cat', 'dog')
        ''')
        ```

        Joins for feature engineering:
        ```python
        # Combine image features with metadata
        features = deeplake.query('''
            SELECT i.image, i.embedding, m.labels, m.metadata
            FROM "mem://images" AS i
            JOIN "mem://metadata" AS m ON i.id = m.image_id
            WHERE m.verified = true
        ''')
        ```
    """
    ...

def query_async(query: str, token: str | None = None, creds: dict[str, str] | None = None) -> Future[DatasetView]:
    """
    Asynchronously executes TQL queries optimized for ML data filtering and search.

    Non-blocking version of `query()` for better performance with large datasets.
    Supports the same TQL features including vector similarity search, text search,
    filtering, and joins.

    Args:
        query: TQL query string supporting:
            - Vector similarity: COSINE_SIMILARITY, EUCLIDEAN_DISTANCE
            - Text search: BM25_SIMILARITY, CONTAINS
            - Filtering: WHERE clauses
            - Sorting: ORDER BY
            - Joins: JOIN across datasets
        token: Optional Activeloop authentication token
        creds (dict, optional): Dictionary containing credentials used to access the dataset at the path.

          - If 'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token' are present, these take precedence over credentials present in the environment or in credentials file. Currently only works with s3 paths.
          - It supports 'aws_access_key_id', 'aws_secret_access_key', 'aws_session_token', 'endpoint_url', 'aws_region', 'profile_name' as keys.
          - If nothing is given is, credentials are fetched from the environment variables. This is also the case when creds is not passed for cloud datasets

    Returns:
        Future: Resolves to DatasetView that can be:
            - Used directly in ML training
            - Further filtered with additional queries
            - Converted to PyTorch/TensorFlow dataloaders
            - Materialized into a new dataset

    <!-- test-context
    ```python
    import deeplake
    def prepare_training():
        pass
    ```
    -->

    Examples:
        Basic async query:
        ```python
        # Run query asynchronously
        future = deeplake.query_async('''
            SELECT * FROM "mem://embeddings"
            ORDER BY COSINE_SIMILARITY(vector, ARRAY[0.1, 0.2, 0.3]) DESC
        ''')

        # Do other work while query runs
        prepare_training()

        # Get results when needed
        results = future.result()
        ```

        With async/await:
        ```python
        async def search_similar():
            results = await deeplake.query_async('''
                SELECT * FROM "mem://images"
                ORDER BY COSINE_SIMILARITY(embedding, ARRAY[0.1, 0.2, 0.3]) DESC
                LIMIT 100
            ''')
            return results

        async def main():
            similar = await search_similar()
        ```

        Non-blocking check:
        ```python
        future = deeplake.query_async(
            "SELECT * FROM dataset WHERE train_split = 'train'"
        )

        if future.is_completed():
            train_data = future.result()
        else:
            print("Query still running...")
        ```
    """
    ...

def explain_query(query: str, token: str | None = None, creds: dict[str, str] | None = None) -> ExplainQueryResult:
    """
    Explains TQL query with optional authentication.

    Args:
        query: TQL query string to explain
        token: Optional Activeloop authentication token
        creds (dict, optional): Dictionary containing credentials used to access the dataset at the path.

    Returns:
        ExplainQueryResult: An explain result object to analyze the query.

    <!-- test-context
    ```python
    import deeplake
    ds = deeplake.create("mem://explain_query")
    ds.add_column("category", "text")
    ds.append({"category": ["active", "inactive", "not sure"]})
    ds.commit()
    ```
    -->

    Examples:
        Explaining a query:
        ```python
        explain_result = deeplake.explain_query('SELECT * FROM "mem://explain_query" WHERE category == \'active\'')
        print(explain_result)
        ```
    """
    ...

class Client:
    """
    Client for connecting to Activeloop services.
    Handles authentication and API communication.
    """
    endpoint: str

class Random:
    """
    A pseudorandom number generator class that allows for deterministic random number generation
    through seed control.
    """

    seed: int | None

class Branch:
    """
    Describes a branch within the dataset.

    Branches are created using [deeplake.Dataset.branch][].
    """

    __hash__: typing.ClassVar[None] = None
    def __eq__(self, other: Branch) -> bool:
        ...

    def __str__(self) -> str:
        ...

    def delete(self) -> None:
        """
        Deletes the branch from the dataset
        """
        ...

    def open(self) -> Dataset:
        """
        Opens corresponding branch of the dataset
        """
        ...

    def open_async(self) -> Future[Dataset]:
        """
        Asynchronously fetches the dataset corresponding to the branch and returns a Future object.
        """
        ...

    def rename(self, new_name: str) -> None:
        """
        Renames the branch within the dataset
        """
        ...

    @property
    def base(self) -> tuple[str, str] | None:
        """
        The base branch id and version
        """
        ...

    @property
    def id(self) -> str:
        """
        The unique identifier of the branch
        """
        ...

    @property
    def name(self) -> str:
        """
        The name of the branch
        """
        ...

    @property
    def timestamp(self) -> datetime.datetime:
        """
        The branch creation timestamp
        """
        ...

class BranchView:
    """
    Describes a read-only branch within the dataset.
    """

    __hash__: typing.ClassVar[None] = None
    def __eq__(self, other: BranchView) -> bool:
        ...

    def __str__(self) -> str:
        ...

    @property
    def base(self) -> tuple[str, str] | None:
        """
        The base branch id and version
        """
        ...
    @property
    def id(self) -> str:
        """
        The unique identifier of the branch
        """
        ...

    @property
    def name(self) -> str:
        """
        The name of the branch
        """
        ...

    @property
    def timestamp(self) -> datetime.datetime:
        """
        The branch creation timestamp
        """
        ...

    def open(self) -> ReadOnlyDataset:
        """
        Opens corresponding branch of the dataset
        """
        ...

    def open_async(self) -> Future[ReadOnlyDataset]:
        """
        Asynchronously fetches the dataset corresponding to the branch and returns a Future object.
        """
        ...

class Branches:
    """
    Provides access to the branches within a dataset.

    It is returned by the [deeplake.Dataset.branches][] property.
    """
    def __getitem__(self, name: str) -> Branch:
        """
        Return a branch by name or id
        """
        ...
    def __len__(self) -> int:
        """
        The number of branches in the dataset
        """
        ...
    def __str__(self) -> str:
        ...
    def names(self) -> list[str]:
        """
        Return a list of branch names
        """
        ...

class BranchesView:
    """
    Provides access to the read-only branches within a dataset view.

    It is returned by the [deeplake.ReadOnlyDataset.branches][] property.
    """
    def __getitem__(self, name: str) -> BranchView:
        """
        Return a branch by name or id
        """
        ...
    def __len__(self) -> int:
        """
        The number of branches in the dataset
        """
        ...
    def __str__(self) -> str:
        ...
    def names(self) -> list[str]:
        """
        Return a list of branch names
        """
        ...

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
    def message(self) -> str:
        """
        The message of the tag
        """

    @property
    def timestamp(self) -> datetime.datetime:
        """
        The tag creation timestamp
        """
        ...

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

    def open(self) -> DatasetView:
        """
        Fetches the dataset corresponding to the tag
        """
        ...

    def open_async(self) -> Future[DatasetView]:
        """
        Asynchronously fetches the dataset corresponding to the tag and returns a Future object.
        """
        ...

    def __str__(self) -> str:
        ...

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
    def message(self) -> str:
        """
        The message of the tag
        """

    @property
    def version(self) -> str:
        """
        The version that has been tagged
        """

    @property
    def timestamp(self) -> datetime.datetime:
        """
        The tag creation timestamp
        """
        ...

    def open(self) -> DatasetView:
        """
        Fetches the dataset corresponding to the tag
        """
        ...

    def open_async(self) -> Future[DatasetView]:
        """
        Asynchronously fetches the dataset corresponding to the tag and returns a Future object.
        """
        ...

    def __str__(self) -> str: ...

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

    def __str__(self) -> str: ...
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

    def __str__(self) -> str: ...
    def names(self) -> list[str]:
        """
        Return a list of tag names
        """
        ...

class ColumnDefinition:
    def __str__(self) -> str: ...
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

    def __str__(self) -> str: ...
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
    Provides read-only access to a column in a dataset. ColumnView is designed for efficient
    data access in ML workflows, supporting both synchronous and asynchronous operations.

    The ColumnView class allows you to:

      - Access column data using integer indices, slices, or lists of indices
      - Retrieve data asynchronously for better performance in ML pipelines
      - Access column metadata and properties
      - Get information about linked data if the column contains references

    <!-- test-context
    ```python
    import deeplake
    ds = deeplake.create("tmp://")
    ds.add_column("images", "int32")
    ds.append({"images": [0] * 300})
    ds.add_column("embeddings", deeplake.types.Array("float32", 1))
    ```
    -->

    Examples:
        Load image data from a column for training:
        ```python
        # Access a single image
        image = ds["images"][0]

        # Load a batch of images
        batch = ds["images"][0:32]

        # Async load for better performance
        images_future = ds["images"].get_async(slice(0, 32))
        images = images_future.result()
        ```

        Access embeddings for similarity search:
        ```python
        # Get all embeddings
        embeddings = ds["embeddings"][:]

        # Get specific embeddings by indices
        selected = ds["embeddings"][[1, 5, 10]]
        ```

        Check column properties:
        ```python
        # Get column name
        name = ds["images"].name

        # Access metadata
        if "mean" in ds["images"].metadata.keys():
            mean = dataset["images"].metadata["mean"]
        ```
    """

    def __getitem__(self, index: int | slice | list | tuple) -> numpy.ndarray | list | core.Dict | str | bytes | None | Array:
        """
        Retrieve data from the column at the specified index or range.

        Parameters:
            index: Can be:

              - int: Single item index
              - slice: Range of indices (e.g., 0:10)
              - list/tuple: Multiple specific indices

        Returns:
            The data at the specified index/indices. Type depends on the column's data type.

        <!-- test-context
        ```python
        import deeplake
        ds = deeplake.create("tmp://")
        ds.add_column("images", "int32")
        ds.append({"images": [0] * 300})
        column = ds["images"]
        ```
        -->

        Examples:
            ```python
            # Get single item
            image = column[0]

            # Get range
            batch = column[0:32]

            # Get specific indices
            items = column[[1, 5, 10]]
            ```
        """
        ...

    def get_async(self, index: int | slice | list | tuple) -> Future[typing.Any]:
        """
        Asynchronously retrieve data from the column. Useful for large datasets or when
        loading multiple items in ML pipelines.

        Parameters:
            index: Can be:

              - int: Single item index
              - slice: Range of indices
              - list/tuple: Multiple specific indices

        Returns:
            Future: A Future object that resolves to the requested data.

        <!-- test-context
        ```python
        import deeplake
        ds = deeplake.create("tmp://")
        ds.add_column("images", "int32")
        ds.append({"images": [0] * 300})
        column = ds["images"]
        ```
        -->

        Examples:
            ```python
            # Async batch load
            future = column.get_async(slice(0, 32))
            batch = future.result()

            # Using with async/await
            async def load_batch():
                batch = await column.get_async(slice(0, 32))
                return batch
            ```
        """
        ...

    def get_bytes(self, index: int | slice | list | tuple) -> bytes | list:
        ...

    def get_bytes_async(self, index: int | slice | list | tuple) -> Future[bytes | list]:
        ...

    def __len__(self) -> int:
        """
        Get the number of items in the column.

        Returns:
            int: Number of items in the column.
        """
        ...

    def __str__(self) -> str: ...

    def _links_info(self) -> dict:
        """
        Get information about linked data if this column contains references to other datasets.

        Internal method used primarily for debugging and advanced operations.

        Returns:
            dict: Information about linked data.
        """
        ...

    @property
    def indexes(self) -> list[types.IndexType]:
        """
        Get a list of indexes on the column.

        <!-- test-context
        ```python
        import deeplake
        ds = deeplake.create("tmp://")
        ```
        -->

        Examples:
            ```python
            ds.add_column("A", deeplake.types.Text(deeplake.types.BM25))
            print([str(element) for element in ds["A"].indexes])
            ```
        """
        ...

    @property
    def metadata(self) -> ReadOnlyMetadata:
        """
        Access the column's metadata. Useful for storing statistics, preprocessing parameters,
        or other information about the column data.

        Returns:
            ReadOnlyMetadata: A ReadOnlyMetadata object for reading metadata.

        <!-- test-context
        ```python
        import deeplake
        ds = deeplake.create("tmp://")
        ds.add_column("images", "int32")
        ds["images"].metadata["mean"] = [0.485, 0.456, 0.406]
        ds["images"].metadata["std"] = [0.229, 0.224, 0.225]
        column = ds["images"]
        ```
        -->

        Examples:
            ```python
            # Access preprocessing parameters
            mean = column.metadata["mean"]
            std = column.metadata["std"]

            # Check available metadata
            for key in column.metadata.keys():
                print(f"{key}: {column.metadata[key]}")
            ```
        """
        ...

    @property
    def name(self) -> str:
        """
        Get the name of the column.

        Returns:
            str: The column name.
        """
        ...


class Column(ColumnView):
    """
    Provides read-write access to a column in a dataset. Column extends ColumnView with
    methods for modifying data, making it suitable for dataset creation and updates in
    ML workflows.

    The Column class allows you to:

      - Read and write data using integer indices, slices, or lists of indices
      - Modify data asynchronously for better performance
      - Access and modify column metadata
      - Handle various data types common in ML: images, embeddings, labels, etc.

    <!-- test-context
    ```python
    import deeplake
    ds = deeplake.create("tmp://")
    ds.add_column("images", "int32")
    ds.append({"images": [0] * 300})
    ds.add_column("embeddings", deeplake.types.Array("float32", 1))
    ds.add_column("labels", "int32")
    new_labels = [1] * 32
    images = [0] * 32
    class Model:
        def encode(self, images):
            return [[0.1]] * len(images)
    model = Model()
    ```
    -->

    Examples:
        Update training labels:
        ```python
        # Update single label
        ds["labels"][0] = 1

        # Update batch of labels
        ds["labels"][0:32] = new_labels

        # Async update for better performance
        future = ds["labels"].set_async(slice(0, 32), new_labels)
        future.wait()
        ```

        Store image embeddings:
        ```python
        # Generate and store embeddings
        embeddings = model.encode(images)
        ds["embeddings"][0:len(embeddings)] = embeddings
        ```

        Manage column metadata:
        ```python
        # Store preprocessing parameters
        ds["images"].metadata["mean"] = [0.485, 0.456, 0.406]
        ds["images"].metadata["std"] = [0.229, 0.224, 0.225]
        ```
    """

    def __setitem__(self, index: int | slice, value: typing.Any) -> None:
        """
        Set data in the column at the specified index or range.

        Parameters:
            index: Can be:

              - int: Single item index
              - slice: Range of indices (e.g., 0:10)
            value: The data to store. Must match the column's data type.

        <!-- test-context
        ```python
        import deeplake
        ds = deeplake.create("tmp://")
        ds.add_column("images", "int32")
        ds.append({"images": [0] * 300})
        column = ds["images"]
        new_image = 1
        new_batch = [1] * 32
        ```
        -->

        Examples:
            ```python
            # Update single item
            column[0] = new_image

            # Update range
            column[0:32] = new_batch
            ```
        """
        ...
    def create_index(self, index_type: types.IndexType) -> None:
        """
        Create an index on the column.

        Parameters:
            index_type: Can be:

              - TextIndexType: Index for text columns
              - EmbeddingIndexType: Index for embedding columns

        <!-- test-context
        ```python
        import deeplake
        ds = deeplake.create("tmp://")
        ```
        -->

        Examples:
            ```python
            ds.add_column("A", deeplake.types.Text)
            ds.append({"A": ["Test"]})
            ds["A"].create_index(deeplake.types.TextIndex(deeplake.types.BM25))
            ```
        """
        ...

    def drop_index(self, index_type: types.IndexType) -> None:
        """
        Drop an index on the column.

        Parameters:
            index_type: Can be:

              - TextIndexType: Index for text columns
              - EmbeddingIndexType: Index for embedding columns

        <!-- test-context
        ```python
        import deeplake
        ds = deeplake.create("tmp://")
        ```
        -->

        Examples:
            ```python
            ds.add_column("A", deeplake.types.Text)
            ds["A"].create_index(deeplake.types.TextIndex(deeplake.types.BM25))
            ds["A"].drop_index(deeplake.types.TextIndex(deeplake.types.BM25))
            ```
        """
        ...


    def set_async(self, index: int | slice, value: typing.Any) -> FutureVoid:
        """
        Asynchronously set data in the column. Useful for large updates or when
        modifying multiple items in ML pipelines.

        Parameters:
            index: Can be:

              - int: Single item index
              - slice: Range of indices
            value: The data to store. Must match the column's data type.

        Returns:
            FutureVoid: A FutureVoid that completes when the update is finished.

        <!-- test-context
        ```python
        import deeplake
        ds = deeplake.create("tmp://")
        ds.add_column("images", "int32")
        ds.append({"images": [0] * 300})
        column = ds["images"]
        new_image = 1
        new_batch = [1] * 32
        ```
        -->

        Examples:
            ```python
            # Async batch update
            future = column.set_async(slice(0, 32), new_batch)
            future.wait()

            # Using with async/await
            async def update_batch():
                await column.set_async(slice(0, 32), new_batch)
            ```
        """
        ...

    @property
    def metadata(self) -> Metadata: ...


class Version:
    """
    An atomic change within [deeplake.Dataset][]'s history
    """

    def __str__(self) -> str: ...
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

    def open_async(self) -> Future[ReadOnlyDataset]:
        """
        Asynchronously fetches the dataset corresponding to the version and returns a Future object.
        """
        ...

class Row:
    """
    Provides mutable access to a particular row in a dataset.
    """

    def __getitem__(self, column: str) ->  numpy.ndarray | list | core.Dict | str | bytes | None | Array:
        """
        The value for the given column
        """

    def get_async(self, column: str) -> Future[numpy.ndarray | list | core.Dict | str | bytes | None | Array]:
        """
        Asynchronously retrieves data for the specified column and returns a Future object.

        Args:
            column (str): The name of the column to retrieve data for.

        Returns:
            Future: A Future object that will resolve to the value containing the column data.

        <!-- test-context
        ```python
        import deeplake
        ds = deeplake.create("tmp://")
        ds.add_column("column_name", "int32")
        ds.append({"column_name": [0] * 300})
        row = ds[0]
        ```
        -->

        Examples:
            ```python
            future = row.get_async("column_name")
            column = future.result()  # Blocking call to get the result when it's ready.
            ```

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

        <!-- test-context
        ```python
        import deeplake
        ds = deeplake.create("tmp://")
        ds.add_column("column_name", "int32")
        ds.append({"column_name": [0] * 300})
        row = ds[0]
        new_value = 1
        ```
        -->

        Examples:
            ```python
            future_void = row.set_async("column_name", new_value)
            future_void.wait()  # Blocks until the operation is complete.
            ```

        Notes:
            - The method sets the value asynchronously and immediately returns a FutureVoid.
            - You can either block and wait for the operation to complete using `wait()`
            or await the FutureVoid object in an asynchronous context.
        """

    def get_bytes(self, column: str) ->  bytes | list:
        ...

    def get_bytes_async(self, column: str) -> Future[bytes | list]:
        ...

    def to_dict(self) -> dict:
        """
        Converts the row to a dictionary.
        """

    def __str__(self) -> str: ...
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

    def __getitem__(self, column: str) ->  numpy.ndarray | list | core.Dict | str | bytes | None | Array:
        """
        The value for the given column
        """

    def get_async(self, column: str) -> Future[numpy.ndarray | list | core.Dict | str | bytes | None | Array]:
        """
        Asynchronously retrieves data for the specified column and returns a Future object.

        Args:
            column (str): The name of the column to retrieve data for.

        Returns:
            Future: A Future object that will resolve to the value containing the column data.

        <!-- test-context
        ```python
        import deeplake
        ds = deeplake.create("tmp://")
        ds.add_column("column_name", "int32")
        ds.append({"column_name": [0] * 300})
        row_range = ds[0:30]
        ```
        -->

        Examples:
            ```python
            future = row_range.get_async("column_name")
            column = future.result()  # Blocking call to get the result when it's ready.
            ```

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

        <!-- test-context
        ```python
        import deeplake
        ds = deeplake.create("tmp://")
        ds.add_column("column_name", "int32")
        ds.append({"column_name": [0] * 300})
        row_range = ds[0:30]
        new_value = [1] * 30
        ```
        -->

        Examples:
            ```python
            future_void = row_range.set_async("column_name", new_value)
            future_void.wait()  # Blocks until the operation is complete.
            ```

        Notes:
            - The method sets the value asynchronously and immediately returns a FutureVoid.
            - You can either block and wait for the operation to complete using `wait()`
            or await the FutureVoid object in an asynchronous context.
        """

    def get_bytes(self, column: str) ->  bytes | list:
        ...

    def get_bytes_async(self, column: str) -> Future[bytes | list]:
        ...

    def summary(self) -> None:
        """
        Prints a summary of the RowRange.
        """

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

    def __getitem__(self, column: str) -> numpy.ndarray | list | core.Dict | str | bytes | None | Array:
        """
        The value for the given column
        """

    def summary(self) -> None:
        """
        Prints a summary of the RowRange.
        """

    def get_async(self, column: str) -> Future[numpy.ndarray | list | core.Dict | str | bytes | None | Array]:
        """
        Asynchronously retrieves data for the specified column and returns a Future object.

        Args:
            column (str): The name of the column to retrieve data for.

        Returns:
            Future: A Future object that will resolve to the value containing the column data.

        <!-- test-context
        ```python
        import deeplake
        ds = deeplake.create("tmp://")
        ds.add_column("column_name", "int32")
        ds.append({"column_name": [0] * 300})
        row_range_view = ds[0:30]
        ```
        -->

        Examples:
            ```python
            future = row_range_view.get_async("column_name")
            column = future.result()  # Blocking call to get the result when it's ready.
            ```

        Notes:
            - The Future will resolve asynchronously, meaning the method will not block execution
            while the data is being retrieved.
            - You can either wait for the result using `future.result()` (a blocking call)
            or use the Future in an `await` expression.
        """

    def get_bytes(self, column: str) -> bytes | list:
        ...

    def get_bytes_async(self, column: str) -> Future[bytes | list]:
        ...

class RowView:
    """
    Provides access to a particular row in a dataset.
    """

    def __getitem__(self, column: str) -> numpy.ndarray | list | core.Dict | str | bytes | None | Array:
        """
        The value for the given column
        """

    def get_async(self, column: str) -> Future[numpy.ndarray | list | core.Dict | str | bytes | None | Array]:
        """
        Asynchronously retrieves data for the specified column and returns a Future object.

        Args:
            column (str): The name of the column to retrieve data for.

        Returns:
            Future: A Future object that will resolve to the value containing the column data.

        <!-- test-context
        ```python
        import deeplake
        ds = deeplake.create("tmp://")
        ds.add_column("column_name", "int32")
        ds.append({"column_name": [0] * 300})
        row_view = ds[0]
        ```
        -->

        Examples:
            ```python
            future = row_view.get_async("column_name")
            column = future.result()  # Blocking call to get the result when it's ready.
            ```

        Notes:
            - The Future will resolve asynchronously, meaning the method will not block execution
            while the data is being retrieved.
            - You can either wait for the result using `future.result()` (a blocking call)
            or use the Future in an `await` expression.
        """

    def get_bytes(self, column: str) -> bytes | list:
        ...

    def get_bytes_async(self, column: str) -> Future[bytes | list]:
        ...

    def to_dict(self) -> dict:
        """
        Converts the row to a dictionary.
        """

    def __str__(self) -> str: ...
    @property
    def row_id(self) -> int:
        """
        The row_id of the row
        """

class DatasetView:
    """
    A DatasetView is a dataset-like structure. It has a defined schema and contains data which can be queried.
    """

    def __str__(self) -> str: ...
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
    def __getitem__(self, indices: list) -> RowRangeView:
        """
        Get a range of rows by the given list of indices within the DatasetView.
        """
        ...

    @typing.overload
    def __getitem__(self, indices: tuple) -> RowRangeView:
        """
        Get a range of rows by the given tuple of indices within the DatasetView.
        """
        ...

    @typing.overload
    def __getitem__(self, column: str) -> ColumnView:
        """
        Get a column by name within the DatasetView.
        """
        ...

    def __getitem__(
        self, input: int | slice | list | tuple | str
    ) -> RowView | RowRangeView | ColumnView:
        """
        Returns a subset of data from the DatasetView.

        The result will depend on the type of value passed to the `[]` operator.

          - `int`: The zero-based offset of the single row to return. Returns a [deeplake.RowView][]
          - `slice`: A slice specifying the range of rows to return. Returns a [deeplake.RowRangeView][]
          - `list`: A list of indices specifying the rows to return. Returns a [deeplake.RowRangeView][]
          - `tuple`: A tuple of indices specifying the rows to return. Returns a [deeplake.RowRangeView
          - `str`: A string specifying column to return all values from. Returns a [deeplake.ColumnView][]

        Examples:
            ```python
            ds = deeplake.create("mem://")
            ds.add_column("id", int)
            ds.add_column("name", str)
            ds.append({"id": [1,2,3], "name": ["Mary", "Joe", "Bill"]})

            row = ds[1]
            print("Id:", row["id"], "Name:", row["name"]) # Output: 2 Name: Joe
            rows = ds[1:2]
            print(rows["id"])

            column_data = ds["id"]
            ```
        """

    def __getstate__(self) -> tuple: ...
    def __setstate__(self, arg0: tuple) -> None: ...
    def __iter__(self) -> typing.Iterator[RowView]:
        """
        Row based iteration over the dataset.

        Examples:
            ```python
            for row in ds:
                # process row
                pass
            ```
        """
        ...

    def __len__(self) -> int:
        """
        The number of rows in the dataset
        """
        ...

    def explain_query(self, query: str) -> ExplainQueryResult:
        """
        Explains a query.

        Parameters:
            query: The query to explain

        Returns:
            ExplainQueryResult: The result of the explanation

        <!-- test-context
        ```python
        import deeplake
        ds = deeplake.create("tmp://")
        ds.add_column("category", "text")
        ds.append({"category": ["active", "inactive", "not sure"]})
        ```
        -->

        Examples:
            ```python
            explain_result = ds.explain_query("select * where category == 'inactive'")
            print(explain_result)
            ```
        """
        ...

    def summary(self) -> None:
        """
        Prints a summary of the dataset.

        Examples:
            ```python
            ds.summary()
            ```
        """
        ...

    def prepare_query(self, query: str) -> Executor:
        """
        Prepares a query for execution.

        Parameters:
            query: The query to prepare

        Returns:
            Executor: The prepared query

        <!-- test-context
        ```python
        import deeplake
        ds = deeplake.create("tmp://")
        ds.add_column("category", "text")
        ds.append({"category": ["active", "inactive", "not sure"]})
        ```
        -->

        Examples:
            ```python
            executor = ds.prepare_query("select * where category == ?")
            results = executor.run_batch([['active'], ['inactive'], ['not sure']])
            for row in results:
                print("Id is: ", row["category"])
            ```
        """
        ...

    def query(self, query: str) -> DatasetView:
        """
        Executes the given TQL query against the dataset and return the results as a [deeplake.DatasetView][].

        <!-- test-context
        ```python
        import deeplake
        ds = deeplake.create("tmp://")
        ds.add_column("category", "text")
        ```
        -->

        Examples:
            ```python
            result = ds.query("select * where category == 'active'")
            for row in result:
                print("Id is: ", row["id"])
            ```
        """
        ...

    def query_async(self, query: str) -> Future[DatasetView]:
        """
        Asynchronously executes the given TQL query against the dataset and return a future that will resolve into [deeplake.DatasetView][].

        <!-- test-context
        ```python
        import deeplake
        ds = deeplake.create("tmp://")
        ds.add_column("category", "text")
        ```
        -->

        Examples:
            ```python
            future = ds.query_async("select * where category == 'active'")
            result = future.result()
            for row in result:
                print("Id is: ", row["id"])

            async def query_and_process():
                # or use the Future in an await expression
                future = ds.query_async("select * where category == 'active'")
                result = await future
                for row in result:
                    print("Id is: ", row["id"])
            ```
        """
        ...

    def tag(self, name: str | None = None, message: str | None = None) -> Tag:
        """
        Saves the current view as a tag to its source dataset and returns the tag.
        """
        ...

    @property
    def schema(self) -> SchemaView:
        """
        The schema of the dataset.
        """

    def to_csv(self, stream: typing.Any) -> None:
        """
        Exports the dataset to a stream in CSV format.

        <!-- test-context
        ```python
        import io
        import deeplake
        ds = deeplake.create("tmp://")
        ds.add_column("A", "text")
        ds.add_column("B", "int32")
        ds.append({"A": ["Alice", "Bob"], "B": [25, 30]})
        ```
        -->

        Examples:
            ```python
            output = io.StringIO()
            ds.to_csv(output)
            print(output.getvalue())
            ```
        """
        ...

    def tensorflow(self) -> typing.Any:
        """
        Returns a TensorFlow `tensorflow.data.Dataset` wrapper around this DatasetView.

        Raises:
            ImportError: If TensorFlow is not installed

        <!-- test-context
        ```python
        import deeplake
        ds = deeplake.create("tmp://")
        ds.add_column("images", "int32")
        ds.append({"images": [0] * 300})
        def process_batch(batch):
            pass
        ```
        -->

        Examples:
            ```python
            dl = ds.tensorflow().shuffle(500).batch(32)
            for i_batch, sample_batched in enumerate(dl):
                 process_batch(sample_batched)
            ```
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

        <!-- test-context
        ```python
        import deeplake
        import multiprocessing as mp
        try:
            mp.set_start_method('fork')
        except RuntimeError:
            pass  # method has already been set


        ds = deeplake.create("tmp://")
        ds.add_column("images", "int32")
        ds.append({"images": [0] * 300})

        def process_batch(batch):
            pass
        ```
        -->

        Examples:
            ```python
            from torch.utils.data import DataLoader

            dl = DataLoader(ds.pytorch(), batch_size=60,
                                        shuffle=True, num_workers=8)
            for i_batch, sample_batched in enumerate(dl):
                process_batch(sample_batched)
            ```
        """
        ...

    def batches(self, batch_size: int, drop_last: bool = False) -> typing.Iterable:
        """
        The batches can be used to more efficiently stream large amounts of data from a DeepLake dataset, such as to the DataLoader then to the training framework.

        Parameters:
            batch_size: Number of rows in each batch
            drop_last: Whether to drop the final batch if it is incomplete

        <!-- test-context
        ```python
        ds = deeplake.create("tmp://")
        def process_batch(*args, **kwargs):
            pass
        ```
        -->

        Examples:
            ```python
            batches = ds.batches(batch_size=2000, drop_last=True)
            for batch in batches:
                process_batch(batch["images"])
            ```
        """
        ...


class IndexingMode:
    """
    Enumeration of available indexing modes in deeplake.

    Members:
        Always: Indices are always updated at commit.
        Automatic: Deeplake automatically detects when to update the indices.
        Off: Index updates are disabled during the session.
    """
    Always: typing.ClassVar[IndexingMode]
    Automatic: typing.ClassVar[IndexingMode]
    Off: typing.ClassVar[IndexingMode]
    __members__: typing.ClassVar[Dict[str, IndexingMode]]

    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...


class Dataset(DatasetView):
    """
    Datasets are the primary data structure used in DeepLake. They are used to store and manage data for searching, training, evaluation.

    Unlike [deeplake.ReadOnlyDataset][], instances of `Dataset` can be modified.
    """

    def __str__(self) -> str:
        ...

    def branch(self, name: str, version: str | None = None) -> Branch:
        """
        Creates a branch with the given version of the current branch. If no version is given, the current version will be picked up.

        Parameters:
            name: The name of the branch
            version: The version of the dataset
        """
        ...

    def merge(self, branch_name: str, version: str | None = None) -> None:
        """
        Merge the given branch into the current branch. If no version is given, the current version will be picked up.

        Parameters:
            name: The name of the branch
            version: The version of the dataset

        <!-- test-context
        ```python
        import deeplake
        ```
        -->

        Examples:
            ```python
            ds = deeplake.create("mem://merge_branch")
            ds.add_column("c1", deeplake.types.Int64())
            ds.append({"c1": [1, 2, 3]})
            ds.commit()

            b = ds.branch("Branch1")
            branch_ds = b.open()
            branch_ds.append({"c1": [4, 5, 6]})
            branch_ds.commit()

            ds.merge("Branch1")
            print(len(ds))
            ```
        """
        ...

    def tag(self, name: str, message: str | None = None, version: str | None = None) -> Tag:
        """
        Tags a version of the dataset. If no version is given, the current version is tagged.

        Parameters:
            name: The name of the tag
            version: The version of the dataset to tag
        """
        ...

    @property
    def current_branch(self) -> Branch:
        """
        The current active branch
        """
        ...

    @property
    def branches(self) -> Branches:
        """
        The collection of [deeplake.Branch][]s within the dataset
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

    auto_commit_enabled: bool
    """
    This property controls whether the dataset will perform time-based auto-commits.

    <!-- test-context
    ```python
    import deeplake
    ds = deeplake.create("mem://auto_commit_ds")
    ds.auto_commit_enabled = False
    ds.add_column("column_name", deeplake.types.Text(deeplake.types.BM25))
    a = ['a']*10_000
    ds.append({"column_name":a})
    ds.commit()
    ```
    -->

    Examples:
        ```python
        ds = deeplake.open("mem://auto_commit_ds")
        ds.auto_commit_enabled = True
        ```
    """

    indexing_mode: IndexingMode
    """
    The indexing mode of the dataset. This property can be set to change the indexing mode of the dataset for the current session,
    other sessions will not be affected.

    <!-- test-context
    ```python
    import deeplake
    ds = deeplake.create("mem://indexing_mode_ds")
    ds.indexing_mode = deeplake.IndexingMode.Off
    ds.add_column("column_name", deeplake.types.Text(deeplake.types.BM25))
    a = ['a']*10_000
    ds.append({"column_name":a})
    ds.commit()
    ```
    -->

    Examples:
        ```python
        ds = deeplake.open("mem://indexing_mode_ds")
        ds.indexing_mode = deeplake.IndexingMode.Automatic
        ds.commit()
        ```
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
    def __getitem__(self, indices: list) -> RowRange:
        """
        Get a range of rows by the given list of indices within the dataset.
        """
        ...

    @typing.overload
    def __getitem__(self, indices: tuple) -> RowRange:
        """
        Get a range of rows by the given tuple of indices within the dataset.
        """
        ...

    @typing.overload
    def __getitem__(self, column: str) -> Column:
        """
        Get a column by name within the dataset.
        """
        ...

    def __getitem__(self, input: int | slice | list | tuple | str) -> Row | RowRange | Column:
        """
        Returns a subset of data from the Dataset

        The result will depend on the type of value passed to the `[]` operator.

        - `int`: The zero-based offset of the single row to return. Returns a [deeplake.Row][]
        - `slice`: A slice specifying the range of rows to return. Returns a [deeplake.RowRange][]
        - `list`: A list of indices specifying the rows to return. Returns a [deeplake.RowRange][]
        - `tuple`: A tuple of indices specifying the rows to return. Returns a [deeplake.RowRange][]
        - `str`: A string specifying column to return all values from. Returns a [deeplake.Column][]

        <!-- test-context
        ```python
        import deeplake
        ds = deeplake.create("tmp://")
        ds.add_column("id", int)
        ds.append({"id": [3] * 3000})
        ```
        -->

        Examples:
            ```python
            row = ds[318]

            rows = ds[931:1038]

            rows = ds[931:1038:3]

            rows = ds[[1, 3, 5, 7]]

            rows = ds[(1, 3, 5, 7)]

            column_data = ds["id"]
            ```
        """
    ...

    def __iter__(self) -> typing.Iterator[Row]:
        """
        Row based iteration over the dataset.

        Examples:
            ```python
            for row in ds:
                # process row
                pass
            ```
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

    def add_column(self, name: str, dtype: typing.Any, default_value: typing.Any = None) -> None:
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
            ```python
            ds.add_column("labels", deeplake.types.Int32)

            ds.add_column("categories", "int32")

            ds.add_column("name", deeplake.types.Text())

            ds.add_column("json_data", deeplake.types.Dict())

            ds.add_column("images", deeplake.types.Image(dtype=deeplake.types.UInt8(), sample_compression="jpeg"))

            ds.add_column("embedding", deeplake.types.Embedding(size=768))
            ```

        Raises:
            deeplake.ColumnAlreadyExistsError: If a column with the same name already exists.
        """

    def remove_column(self, name: str) -> None:
        """
        Remove the existing column from the dataset.

        Args:
            name: The name of the column to remove

        <!-- test-context
        ```python
        import deeplake
        ds = deeplake.create("tmp://")
        ds.add_column("name", "text")
        ```
        -->

        Examples:
            ```python
            ds.remove_column("name")
            ```

        Raises:
            deeplake.ColumnDoesNotExistsError: If a column with the specified name does not exists.
        """

    def rename_column(self, name: str, new_name: str) -> None:
        """
        Renames the existing column in the dataset.

        Args:
            name: The name of the column to rename
            new_name: The new name to set to column

        <!-- test-context
        ```python
        import deeplake
        ds = deeplake.create("tmp://")
        ds.add_column("old_name", "text")
        ```
        -->

        Examples:
            ```python
            ds.rename_column("old_name", "new_name")
            ```

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

        <!-- test-context
        ```python
        import deeplake
        import numpy as np
        ds = deeplake.create("tmp://")
        ds.add_column("name", "text")
        ds.add_column("age", "int32")
        ds2 = deeplake.create("tmp://")
        ds2.add_column("text", "text")
        ds2.add_column("embedding", deeplake.types.Embedding(size=768))
        deeplake.from_parquet = lambda x: ds2
        ```
        -->

        Examples:
            ```python
            ds.append({"name": ["Alice", "Bob"], "age": [25, 30]})

            ds.append([{"name": "Alice", "age": 25}, {"name": "Bob", "age": 30}])
            ```

            ```python
            ds2.append({
                "embedding": np.random.rand(4, 768),
                "text": ["Hello World"] * 4})

            ds2.append([{"embedding": np.random.rand(768), "text": "Hello World"}] * 4)
            ```

            ```python
            ds2.append(deeplake.from_parquet("./file.parquet"))
            ```

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
            ```python
            ds.commit()

            ds.commit("Added data from updated documents")
            ```
        """

    def commit_async(self, message: str | None = None) -> FutureVoid:
        """
        Asynchronously commits changes you have made to the dataset.

        See [deeplake.Dataset.commit][] for more information.

        Parameters:
            message (str, optional): A message to store in history describing the changes made in the commit

        Examples:
            ```python
            ds.commit_async().wait()

            ds.commit_async("Added data from updated documents").wait()

            async def do_commit():
                await ds.commit_async()

            future = ds.commit_async() # then you can check if the future is completed using future.is_completed()
            ```
        """

    def rollback(self) -> None:
        """
        Reverts any in-progress changes to the dataset you have made. Does not revert any changes that have been committed.
        """

    def rollback_async(self) -> FutureVoid:
        """
        Asynchronously reverts any in-progress changes to the dataset you have made. Does not revert any changes that have been committed.
        """

    def set_creds_key(self, key: str, token: str | None = None) -> None:
        """
        Sets the key used to store the credentials for the dataset.
        """
        pass

    @property
    def creds_key(self) -> str | None:
        """
        The key used to store the credentials for the dataset.
        """

    def push(
        self, url: str, creds: dict[str, str] | None = None, token: str | None = None
    ) -> None:
        """
        Pushes any new history from this dataset to the dataset at the given url

        Similar to [deeplake.Dataset.pull][] but the other direction.

        Parameters:
            url: The URL of the destination dataset
            creds: Optional credentials needed to connect to the dataset
            token: Optional deeplake token
        """
        ...

    def push_async(
        self, url: str, creds: dict[str, str] | None = None, token: str | None = None
    ) -> FutureVoid:
        """
        Asynchronously Pushes new any history from this dataset to the dataset at the given url

        Similar to [deeplake.Dataset.pull_async][] but the other direction.

        Parameters:
            url: The URL of the destination dataset
            creds: Optional credentials needed to connect to the dataset
            token: Optional deeplake token
        """
        ...

    def pull(
        self, url: str, creds: dict[str, str] | None = None, token: str | None = None
    ) -> None:
        """
        Pulls any new history from the dataset at the passed url into this dataset.

        Similar to [deeplake.Dataset.push][] but the other direction.

        Parameters:
            url: The URL of the destination dataset
            creds: Optional credentials needed to connect to the dataset
            token: Optional deeplake token
        """
        ...

    def pull_async(
        self, url: str, creds: dict[str, str] | None = None, token: str | None = None
    ) -> FutureVoid:
        """
        Asynchronously pulls any new history from the dataset at the passed url into this dataset.

        Similar to [deeplake.Dataset.push_async][] but the other direction.

        Parameters:
            url: The URL of the destination dataset
            creds: Optional credentials needed to connect to the dataset
            token: Optional deeplake token
        """
        ...

    def refresh(
        self
    ) -> None:
        """
        Refreshes any new info from the dataset.

        Similar to [deeplake.Dataset.open_read_only][] but the lightweight way.
        """
        ...

    def refresh_async(
        self
    ) -> FutureVoid:
        """
        Asynchronously refreshes any new info from the dataset.

        Similar to [deeplake.Dataset.open_read_only_async][] but the lightweight way.
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
    def __iter__(self) -> typing.Iterator[RowView]:
        """
        Row based iteration over the dataset.

        Examples:
            ```python
            for row in ds:
                # process row
                pass
            ```
        """
        ...

    def __str__(self) -> str:
        ...

    @property
    def current_branch(self) -> BranchView:
        """
        The current active branch
        """
        ...

    @property
    def branches(self) -> BranchesView:
        """
        The collection of [deeplake.BranchView][]s within the dataset
        """
        ...

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

    def push(
        self, url: str, creds: dict[str, str] | None = None, token: str | None = None
    ) -> None:
        """
        Pushes any history from this dataset to the dataset at the given url

        Parameters:
            url: The URL of the destination dataset
            creds: Optional credentials needed to connect to the dataset
            token: Optional deeplake token
        """
        ...

    def push_async(
        self, url: str, creds: dict[str, str] | None = None, token: str | None = None
    ) -> FutureVoid:
        """
        Asynchronously Pushes any history from this dataset to the dataset at the given url

        Parameters:
            url: The URL of the destination dataset
            creds: Optional credentials needed to connect to the dataset
            token: Optional deeplake token
        """
        ...

    def refresh(
        self
    ) -> None:
        """
        Refreshes any new info from the dataset.

        Similar to [deeplake.Dataset.open_read_only][] but the lightweight way.
        """
        ...

    def refresh_async(
        self
    ) -> FutureVoid:
        """
        Asynchronously refreshes any new info from the dataset.

        Similar to [deeplake.Dataset.open_read_only_async][] but the lightweight way.
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

class Executor:
    def get_query_string(self) -> str:
        ...
    def run_single(self) -> DatasetView:
        ...
    def run_single_async(self) -> Future[DatasetView]:
        ...
    def run_batch(self, parameters: list = None) -> list:
        ...
    def run_batch_async(self, parameters: list = None) -> Future[list[DatasetView]]:
        ...

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

class InvalidPolygonShapeError(Exception):
    pass

class InvalidLinkDataError(Exception):
    pass

class InvalidIndexCreationError(Exception):
    pass

class DimensionsMismatchError(Exception):
    pass

class IndexAlreadyExistsError(Exception):
    pass

class EmbeddingSizeMismatch(Exception):
    pass

class EmptyColumnNameError(Exception):
    pass

class InvalidCredsKeyAssignmentError(Exception):
    pass

class CredsKeyAlreadyAssignedError(Exception):
    pass

class GcsStorageProviderFailed(Exception):
    pass

class History:
    """
    The version history of a [deeplake.Dataset][].
    """

    @typing.overload
    def __getitem__(self, offset: int) -> Version: ...
    @typing.overload
    def __getitem__(self, version: str) -> Version: ...
    def __getitem__(self, input: int | str) -> Version: ...
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

    def __str__(self) -> str: ...


class ReadOnlyDatasetModificationError(Exception):
    pass

class DatasetUnavailableError(Exception):
    pass

class CannotDeleteMainBranchError(Exception):
    pass

class CannotRenameMainBranchError(Exception):
    pass

class BranchExistsError(Exception):
    pass

class BranchNotFoundError(Exception):
    pass

class TagNotFoundError(Exception):
    pass

class TagExistsError(Exception):
    pass

class PermissionDeniedError(Exception):
    pass

class PushError(Exception):
    pass

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

class BadRequestError(Exception):
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

class InvalidLinkType(Exception):
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

class InvalidSegmentMaskCompression(Exception):
    pass

class InvalidMedicalCompression(Exception):
    pass

class UnexpectedMedicalTypeInputData(Exception):
    pass

class UnexpectedInputDataForDicomColumn(Exception):
    pass

class InvalidBinaryMaskCompression(Exception):
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

class HTTPRequestFailedError(Exception):
    pass

class HTTPBodyIsMissingError(Exception):
    pass

class HTTPBodyIsNotJSONError(Exception):
    pass

class InvalidURIError(Exception):
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

    def __str__(self) -> str: ...

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

    def __str__(self) -> str: ...

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

class VersionNotFoundError(Exception):
    pass

class WriteFailedError(Exception):
    pass

def create(
    url: str,
    creds: dict[str, str] | None = None,
    token: str | None = None,
    schema: dict[str, typing.Any] | typing.Any | None = None,
) -> Dataset:
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

    <!-- test-context
    ```python
    import deeplake
    from deeplake import types
    ds = deeplake.create("tmp://")
    def create(path, creds = None, token = None, org_id = None):
        return ds
    deeplake.create = create
    key = ''
    id = ''
    ```
    -->

    Examples:
        ```python
        # Create a dataset in your local filesystem:
        ds = deeplake.create("directory_path")
        ds.add_column("id", types.Int32())
        ds.add_column("url", types.Text())
        ds.add_column("embedding", types.Embedding(768))
        ds.commit()
        ds.summary()
        ```

        ```python
        # Create dataset in your app.activeloop.ai organization:
        ds = deeplake.create("al://organization_id/dataset_name")

        # Create a dataset stored in your cloud using specified credentials:
        ds = deeplake.create("s3://mybucket/my_dataset",
            creds = {"aws_access_key_id": id, "aws_secret_access_key": key})

        # Create dataset stored in your cloud using app.activeloop.ai managed credentials.
        ds = deeplake.create("s3://mybucket/my_dataset",
            creds = {"creds_key": "managed_creds_key"}, org_id = "my_org_id")

        ds = deeplake.create("azure://bucket/path/to/dataset")

        ds = deeplake.create("gcs://bucket/path/to/dataset")

        ds = deeplake.create("mem://in-memory")
        ```

    Raises:
        LogExistsError: if a dataset already exists at the given URL
    """

def create_async(
    url: str,
    creds: dict[str, str] | None = None,
    token: str | None = None,
    schema: dict[str, typing.Any] | typing.Any | None = None,
) -> Future[Dataset]:
    """
    Asynchronously creates a new dataset at the given URL.

    See [deeplake.create][] for more information.

    To open an existing dataset, use [deeplake.open_async][].

    <!-- test-context
    ```python
    import deeplake
    from deeplake import types
    ds = deeplake.create_async("tmp://")
    def create(path, creds = None, token = None, org_id = None):
        return ds
    deeplake.create_async = create
    key = ''
    id = ''
    ```
    -->

    Examples:
        ```python
        async def create_dataset():
            # Asynchronously create a dataset in your local filesystem:
            ds = await deeplake.create_async("directory_path")
            await ds.add_column("id", types.Int32())
            await ds.add_column("url", types.Text())
            await ds.add_column("embedding", types.Embedding(768))
            await ds.commit()
            await ds.summary()  # Example of usage in an async context

            # Alternatively, create a dataset using .result().
            future_ds = deeplake.create_async("directory_path")
            ds = future_ds.result()  # Blocks until the dataset is created

            # Create a dataset in your app.activeloop.ai organization:
            ds = await deeplake.create_async("al://organization_id/dataset_name")

            # Create a dataset stored in your cloud using specified credentials:
            ds = await deeplake.create_async("s3://mybucket/my_dataset",
                creds={"aws_access_key_id": id, "aws_secret_access_key": key})

            # Create dataset stored in your cloud using app.activeloop.ai managed credentials.
            ds = await deeplake.create_async("s3://mybucket/my_dataset",
                creds={"creds_key": "managed_creds_key"}, org_id="my_org_id")

            ds = await deeplake.create_async("azure://bucket/path/to/dataset")

            ds = await deeplake.create_async("gcs://bucket/path/to/dataset")

            ds = await deeplake.create_async("mem://in-memory")
        ```

    Raises:
        RuntimeError: if a dataset already exists at the given URL (will be raised when the future is awaited)
    """

def _create_global_cache(
    size: int = None,
) -> None:
    """
    Creates a global cache with the given size.
    Args:
        size (int, optional): The size of the global cache in bytes. If not specified, a default size of 1GB is used.
    """

def copy(
    src: str,
    dst: str,
    src_creds: dict[str, str] | None = None,
    dst_creds: dict[str, str] | None = None,
    token: str | None = None,
) -> None:
    """
    Copies the dataset at the source URL to the destination URL.

    NOTE: Currently private due to potential issues in file timestamp handling

    Args:
        src (str): The URL of the source dataset.
        dst (str): The URL of the destination dataset.
        src_creds (dict, str, optional): The string ``ENV`` or a dictionary containing credentials used to access the source dataset at the path.
        dst_creds (dict, str, optional): The string ``ENV`` or a dictionary containing credentials used to access the destination dataset at the path.
        token (str, optional): Activeloop token, used for fetching credentials to the dataset at path if it is a Deep Lake dataset. This is optional, tokens are normally autogenerated.

    <!-- test-context
    ```python
    import deeplake
    from deeplake import types
    deeplake.copy = lambda src, dst: None
    ```
    -->

    Examples:
        ```python
        deeplake.copy("al://organization_id/source_dataset", "al://organization_id/destination_dataset")
        ```
    """

def delete(
    url: str, creds: dict[str, str] | None = None, token: str | None = None
) -> None:
    """
    Deletes an existing dataset.

    !!! warning
        This operation is irreversible. All data will be lost.

        If concurrent processes are attempting to write to the dataset while it's being deleted, it may lead to data inconsistency.
        It's recommended to use this operation with caution.
    """

def delete_async(
    url: str, creds: dict[str, str] | None = None, token: str | None = None
) -> FutureVoid:
    """
    Asynchronously deletes an existing dataset.

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

    To create a new dataset, see [deeplake.create][]

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

    <!-- test-context
    ```python
    import deeplake
    deeplake.open = lambda url, creds = None, token = None, org_id = None: None
    id = ''
    key = ''
    ```
    -->

    Examples:
        ```python
        # Load dataset managed by Deep Lake.
        ds = deeplake.open("al://organization_id/dataset_name")

        # Load dataset stored in your cloud using your own credentials.
        ds = deeplake.open("s3://bucket/my_dataset",
            creds = {"aws_access_key_id": id, "aws_secret_access_key": key})

        # Load dataset stored in your cloud using Deep Lake managed credentials.
        ds = deeplake.open("s3://bucket/my_dataset",
            creds = {"creds_key": "managed_creds_key"}, org_id = "my_org_id")

        ds = deeplake.open("s3://bucket/path/to/dataset")

        ds = deeplake.open("azure://bucket/path/to/dataset")

        ds = deeplake.open("gcs://bucket/path/to/dataset")
        ```
    """

def open_async(
    url: str, creds: dict[str, str] | None = None, token: str | None = None
) -> Future[Dataset]:
    """
    Asynchronously opens an existing dataset, potentially for modifying its content.

    See [deeplake.open][] for opening the dataset synchronously.

    Examples:
        ```python
        async def async_open():
            # Asynchronously load dataset managed by Deep Lake using await.
            ds = await deeplake.open_async("al://organization_id/dataset_name")

            # Asynchronously load dataset stored in your cloud using your own credentials.
            ds = await deeplake.open_async("s3://bucket/my_dataset",
                creds={"aws_access_key_id": id, "aws_secret_access_key": key})

            # Asynchronously load dataset stored in your cloud using Deep Lake managed credentials.
            ds = await deeplake.open_async("s3://bucket/my_dataset",
                creds={"creds_key": "managed_creds_key"}, org_id="my_org_id")

            ds = await deeplake.open_async("s3://bucket/path/to/dataset")

            ds = await deeplake.open_async("azure://bucket/path/to/dataset")

            ds = await deeplake.open_async("gcs://bucket/path/to/dataset")

            # Alternatively, load the dataset using .result().
            future_ds = deeplake.open_async("al://organization_id/dataset_name")
            ds = future_ds.result()  # Blocks until the dataset is loaded
        ```
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

    <!-- test-context
    ```python
    import deeplake
    deeplake.like = lambda src, dest, creds = None, token = None: None
    ```
    -->

    Examples:
        ```python
        ds = deeplake.like(src="az://bucket/existing/to/dataset",
           dest="s3://bucket/new/dataset")
        ```
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

    <!-- test-context
    ```python
    import deeplake
    deeplake.connect = lambda src, dest = None, org_id = None, creds_key = None, token = None: None
    ```
    -->

    Examples:
        ```python
        ds = deeplake.connect("s3://bucket/path/to/dataset",
            "al://my_org/dataset")

        ds = deeplake.connect("s3://bucket/path/to/dataset",
            "al://my_org/dataset", creds_key="my_key")

        # Connect the dataset as al://my_org/dataset
        ds = deeplake.connect("s3://bucket/path/to/dataset",
            org_id="my_org")

        ds = deeplake.connect("az://bucket/path/to/dataset",
            "al://my_org/dataset", creds_key="my_key")

        ds = deeplake.connect("gcs://bucket/path/to/dataset",
            "al://my_org/dataset", creds_key="my_key")
        ```
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
        ```python
        deeplake.disconnect("al://my_org/dataset_name")
        ```
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

        ds = deeplake.open_read_only("directory_path")
        ds.summary()

        Example Output:
        Dataset length: 5
        Columns:
          id       : int32
          url      : text
          embedding: embedding(768)

        ds = deeplake.open_read_only("file:///path/to/dataset")

        ds = deeplake.open_read_only("s3://bucket/path/to/dataset")

        ds = deeplake.open_read_only("azure://bucket/path/to/dataset")

        ds = deeplake.open_read_only("gcs://bucket/path/to/dataset")

        ds = deeplake.open_read_only("mem://in-memory")
    """

def open_read_only_async(
    url: str, creds: dict[str, str] | None = None, token: str | None = None
) -> Future:
    """
    Asynchronously opens an existing dataset in read-only mode.

    See [deeplake.open_async][] for opening datasets for modification and [deeplake.open_read_only][] for sync open.

    Examples:

        # Asynchronously open a dataset in read-only mode:
        ds = await deeplake.open_read_only_async("directory_path")

        # Alternatively, open the dataset using .result().
        future_ds = deeplake.open_read_only_async("directory_path")
        ds = future_ds.result()  # Blocks until the dataset is loaded

        ds = await deeplake.open_read_only_async("file:///path/to/dataset")

        ds = await deeplake.open_read_only_async("s3://bucket/path/to/dataset")

        ds = await deeplake.open_read_only_async("azure://bucket/path/to/dataset")

        ds = await deeplake.open_read_only_async("gcs://bucket/path/to/dataset")

        ds = await deeplake.open_read_only_async("mem://in-memory")
    """

def convert(
    src: str,
    dst: str,
    dst_creds: dict[str, str] | None = None,
    token: str | None = None
) -> None:
    """
    Converts a Deep Lake v3 dataset to the new v4 format while preserving data and metadata.
    Optimized for ML workloads with efficient handling of large datasets and linked data.

    Args:
        src: URL of the source v3 dataset to convert
        dst: Destination URL for the new v4 dataset. Supports:
            - `file://path` local storage
            - `s3://bucket/path` S3 storage
            - `gs://bucket/path` Google Cloud storage
            - `azure://bucket/path` Azure storage
        dst_creds: Optional credentials for accessing the destination storage.
            Supports cloud provider credentials like access keys
        token: Optional Activeloop authentication token

    <!-- test-context
    ```python
    import deeplake
    deeplake.convert = lambda src, dst, dst_creds = None, token = None: None
    ```
    -->

    Examples:
        ```python
        # Convert local dataset
        deeplake.convert("old_dataset/", "new_dataset/")

        # Convert cloud dataset with credentials
        deeplake.convert(
            "s3://old-bucket/dataset",
            "s3://new-bucket/dataset",
            dst_creds={"aws_access_key_id": "key",
                      "aws_secret_access_key": "secret"}
        )
        ```

    Notes:
        - You can open v3 dataset without converting it to v4 using `deeplake.query('SELECT * FROM "old_dataset/"')`
    """

def from_parquet(url_or_bytes: bytes | str) -> ReadOnlyDataset:
    """
    Opens a Parquet dataset in the deeplake format.

    Args:
        url_or_bytes: The URL of the Parquet dataset or bytes of the Parquet file. If no protocol is specified, it assumes `file://`
    """

def from_csv(url_or_bytes: bytes | str) -> ReadOnlyDataset:
    """
    Opens a CSV dataset in the deeplake format.

    Args:
        url_or_bytes: The URL of the CSV dataset or bytes of the CSV file. If no protocol is specified, it assumes `file://`
    """

def from_coco(
    images_directory: typing.Union[str, pathlib.Path],
    annotation_files: typing.Dict[str, typing.Union[str, pathlib.Path]],
    dest: typing.Union[str, pathlib.Path],
    dest_creds: typing.Optional[Dict[str, str]] = None,
) -> dp.Dataset:
    """Ingest images and annotations in COCO format to a Deep Lake Dataset. The source data can be stored locally or in the cloud.

    Args:
        images_directory (str, pathlib.Path): The path to the directory containing images.
        annotation_files Dict(str, Union[str, pathlib.Path]): dictionary from key to path to JSON annotation file in COCO format.
            - the required keys are the following `instances`, `keypoints` and `stuff`
        dest (str, pathlib.Path):
            - The full path to the dataset. Can be:
            - a Deep Lake cloud path of the form ``al://org_id/datasetname``. To write to Deep Lake cloud datasets, ensure that you are authenticated to Deep Lake (pass in a token using the 'token' parameter).
            - an s3 path of the form ``s3://bucketname/path/to/dataset``. Credentials are required in either the environment or passed to the creds argument.
            - a local file system path of the form ``./path/to/dataset`` or ``~/path/to/dataset`` or ``path/to/dataset``.
            - a memory path of the form ``mem://path/to/dataset`` which doesn't save the dataset but keeps it in memory instead. Should be used only for testing as it does not persist.
        dest_creds (Optional[Dict[str, str]]): The dictionary containing credentials used to access the destination path of the dataset.

    Returns:
        Dataset: The Dataset created from images and COCO annotations.

    Raises:
        CocoAnnotationMissingError: If one or many annotation key is missing from file.
    """

def __prepare_atfork() -> None: ...

class TelemetryClient:
    """
    Client for logging deeplake messages to telemetry.
    """
    endpoint: str
    api_key: str
