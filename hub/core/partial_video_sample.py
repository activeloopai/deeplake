from typing import Optional, Union, Dict, Tuple
from hub.core.sample import Sample
from hub.core.storage import StorageProvider
import numpy as np

try:
    from hub.core.storage.gcs import GCSProvider
except ImportError:
    GCSProvider = None  # type: ignore


class PartialVideoSample(Sample):
    def __init__(
        self,
        path: Optional[str] = None,
        array: np.ndarray = None,
        buffer: Union[bytes, memoryview] = None,
        compression: str = None,
        verify: bool = False,
        shape: Tuple[int] = None,
        dtype: Optional[str] = None,
        creds: Optional[Dict] = None,
        storage: Optional[StorageProvider] = None,
        start: int = 0,
        stop: Optional[int] = None,
        duration: Optional[int] = None,
    ):
        """Represents a single sample for a tensor. Provides all important meta information in one place.

        Note:
            If `self.is_lazy` is True, this `Sample` doesn't actually have any data loaded. To read this data,
                simply try to read it into a numpy array (`sample.array`)

        Args:
            path (str): Path to a sample stored on the local file system that represents a single sample. If `path` is provided, `array` should not be.
                Implicitly makes `self.is_lazy == True`.
            array (np.ndarray): Array that represents a single sample. If `array` is provided, `path` should not be. Implicitly makes `self.is_lazy == False`.
            buffer: (bytes): Byte buffer that represents a single sample. If compressed, `compression` argument should be provided.
            compression (str): Specify in case of byte buffer.
            verify (bool): If a path is provided, verifies the sample if True.
            shape (Tuple[int]): Shape of the sample.
            dtype (str, optional): Data type of the sample.
            creds (optional, Dict): Credentials for s3 and gcp for urls.
            storage (optional, StorageProvider): Storage provider.

        Raises:
            ValueError: Cannot create a sample from both a `path` and `array`.
        """
        super().__init__(
            path, array, buffer, compression, verify, shape, dtype, creds, storage
        )
        self._start = start
        self._stop = stop
        self._duration = duration

    def _read_from_gcs(self):
        if GCSProvider is None:
            raise Exception(
                "GCP dependencies not installed. Install them with pip install hub[gcs]"
            )
        root, key = self._get_root_and_key(path)
        if self.storage is not None:
            url = self.storage.get_presigned_url(key)
        else:
            path = self.path.replace("gcp://", "").replace("gcs://", "")  # type: ignore
            gcs = GCSProvider(root, token=self._creds)
            url = gcs.get_presigned_url(key)
