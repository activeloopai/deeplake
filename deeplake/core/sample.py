import requests
from deeplake.core.compression import (
    compress_array,
    decompress_array,
    verify_compressed_file,
    read_meta_from_compressed_file,
    get_compression,
    _open_video,
    _read_metadata_from_vstream,
    _read_audio_meta,
    _read_3d_data_meta,
)
from deeplake.compression import (
    get_compression_type,
    AUDIO_COMPRESSION,
    IMAGE_COMPRESSION,
    VIDEO_COMPRESSION,
    POINT_CLOUD_COMPRESSION,
    MESH_COMPRESSION,
)
from deeplake.util.exceptions import UnableToReadFromUrlError
from deeplake.util.exif import getexif
from deeplake.core.storage.provider import StorageProvider
from deeplake.util.path import get_path_type, is_remote_path
import numpy as np
from typing import Optional, Tuple, Union, Dict

from PIL import Image  # type: ignore
from PIL.ExifTags import TAGS  # type: ignore
from io import BytesIO

from deeplake.core.storage.s3 import S3Provider
from deeplake.core.storage.google_drive import GDriveProvider

try:
    from deeplake.core.storage.gcs import GCSProvider
except ImportError:
    GCSProvider = None  # type: ignore

import warnings


class Sample:
    path: Optional[str]

    def __init__(
        self,
        path: Optional[str] = None,
        array: Optional[np.ndarray] = None,
        buffer: Optional[Union[bytes, memoryview]] = None,
        compression: Optional[str] = None,
        verify: bool = False,
        shape: Optional[Tuple[int]] = None,
        dtype: Optional[str] = None,
        creds: Optional[Dict] = None,
        storage: Optional[StorageProvider] = None,
    ):
        """Represents a single sample for a tensor. Provides all important meta information in one place.

        Note:
            If ``self.is_lazy`` is ``True``, this :class:`Sample` doesn't actually have any data loaded. To read this data, simply try to read it into a numpy array (`sample.array`)

        Args:
            path (str): Path to a sample stored on the local file system that represents a single sample. If ``path`` is provided, ``array`` should not be.
                Implicitly makes ``self.is_lazy == True``.
            array (np.ndarray): Array that represents a single sample. If ``array`` is provided, ``path`` should not be. Implicitly makes ``self.is_lazy == False``.
            buffer: (bytes): Byte buffer that represents a single sample. If compressed, ``compression`` argument should be provided.
            compression (str): Specify in case of byte buffer.
            verify (bool): If a path is provided, verifies the sample if ``True``.
            shape (Tuple[int]): Shape of the sample.
            dtype (optional, str): Data type of the sample.
            creds (optional, Dict): Credentials for s3, gcp and http urls.
            storage (optional, StorageProvider): Storage provider.

        Raises:
            ValueError: Cannot create a sample from both a ``path`` and ``array``.
        """
        if path is None and array is None and buffer is None:
            raise ValueError("Must pass one of `path`, `array` or `buffer`.")

        self._compressed_bytes = {}
        self._uncompressed_bytes = None

        self._array = None
        self._pil = None
        self._typestr = None
        self._shape = shape or None
        self._dtype = dtype or None

        self.path = None
        self.storage = storage
        self._buffer = None
        self._creds = creds or {}
        self._verify = verify

        if path is not None:
            self.path = path
            self._compression = compression
            if self._verify:
                if self._compression is None:
                    self._compression = get_compression(path=self.path)
                compressed_bytes = self._read_from_path()
                if self._compression is None:
                    self._compression = get_compression(header=compressed_bytes[:32])
                self._shape, self._typestr = verify_compressed_file(compressed_bytes, self._compression)  # type: ignore

        if array is not None:
            self._array = array
            self._shape = array.shape  # type: ignore
            self._typestr = array.__array_interface__["typestr"]
            self._dtype = np.dtype(self._typestr).name
            self._compression = None

        if buffer is not None:
            self._compression = compression
            self._buffer = buffer
            if compression is None:
                self._uncompressed_bytes = buffer
            else:
                self._compressed_bytes[compression] = buffer
                if self._verify:
                    self._shape, self._typestr = verify_compressed_file(buffer, self._compression)  # type: ignore

        self.htype = None

    @property
    def buffer(self):
        if self._buffer is None and self.path is not None:
            self._read_from_path()
        if self._buffer is not None:
            return self._buffer
        return self.compressed_bytes(self.compression)

    @property
    def is_text_like(self):
        return self.htype in {"text", "list", "json"}

    @property
    def dtype(self):
        if self._dtype is None:
            self._read_meta()
            self._dtype = np.dtype(self._typestr).name
        return self._dtype

    @property
    def shape(self):
        self._read_meta()
        return self._shape

    @property
    def compression(self):
        if self._compression is None and self.path:
            self._read_meta()
        return self._compression

    def _load_dicom(self):
        if self._array is not None:
            return
        try:
            from pydicom import dcmread
        except ImportError:
            raise ModuleNotFoundError(
                "Pydicom not found. Install using `pip install pydicom`"
            )
        if self.path and get_path_type(self.path) == "local":
            dcm = dcmread(self.path)
        else:
            dcm = dcmread(BytesIO(self.buffer))
        self._array = dcm.pixel_array
        self._shape = self._array.shape
        self._typestr = self._array.__array_interface__["typestr"]

    def _read_meta(self, f=None):
        if self._shape is not None:
            return
        store = False
        if self._compression is None and self.path:
            self._compression = get_compression(path=self.path)
        if f is None:
            if self.path:
                if is_remote_path(self.path):
                    f = self._read_from_path()
                    self._buffer = f
                    store = True
                else:
                    f = self.path
            else:
                f = self._buffer
        self._compression, self._shape, self._typestr = read_meta_from_compressed_file(
            f, compression=self._compression
        )
        if store:
            self._compressed_bytes[self._compression] = f

    def _get_dicom_meta(self) -> dict:
        try:
            from pydicom import dcmread
            from pydicom.dataelem import RawDataElement
        except ImportError:
            raise ModuleNotFoundError(
                "Pydicom not found. Install using `pip install pydicom`"
            )
        if self.path and get_path_type(self.path) == "local":
            dcm = dcmread(self.path)
        else:
            dcm = dcmread(BytesIO(self.buffer))

        meta = {
            x.keyword: {
                "name": x.name,
                "tag": str(x.tag),
                "value": x.value
                if isinstance(x.value, (str, int, float))
                else x.to_json_dict(None, None).get("Value", ""),  # type: ignore
                "vr": x.VR,
            }
            for x in dcm
            if not isinstance(x.value, bytes)
        }
        return meta

    def _get_video_meta(self) -> dict:
        if self.path and get_path_type(self.path) == "local":
            container, vstream = _open_video(self.path)
        else:
            container, vstream = _open_video(self.buffer)
        _, duration, fps, timebase = _read_metadata_from_vstream(container, vstream)
        return {"duration": duration, "fps": fps, "timebase": timebase}

    def _get_audio_meta(self) -> dict:
        if self.path and get_path_type(self.path) == "local":
            info = _read_audio_meta(self.path)
        else:
            info = _read_audio_meta(self.buffer)
        return info

    def _get_point_cloud_meta(self) -> dict:
        if self.path and get_path_type(self.path) == "local":
            info = _read_3d_data_meta(self.path)
        else:
            info = _read_3d_data_meta(self.buffer)
        return info

    @property
    def is_lazy(self) -> bool:
        return self._array is None

    @property
    def is_empty(self) -> bool:
        return 0 in self.shape

    def _recompress(self, buffer: bytes, compression: str) -> bytes:
        if get_compression_type(self._compression) != IMAGE_COMPRESSION:
            raise ValueError(
                "Recompression with different format is only supported for images."
            )
        img = Image.open(BytesIO(buffer))
        if img.mode == "1":
            self._uncompressed_bytes = img.tobytes("raw", "L")
        else:
            self._uncompressed_bytes = img.tobytes()
        return compress_array(self.array, compression)

    def compressed_bytes(self, compression: Optional[str]) -> bytes:
        """Returns this sample as compressed bytes.

        Note:
            If this sample is pointing to a path and the requested ``compression`` is the same as it's stored in, the data is returned without re-compressing.

        Args:
            compression (Optional[str]): ``self.array`` will be compressed into this format. If ``compression`` is ``None``, return :meth:`uncompressed_bytes`.

        Returns:
            bytes: Bytes for the compressed sample. Contains all metadata required to decompress within these bytes.

        Raises:
            ValueError: On recompression of unsupported formats.
        """

        if compression is None:
            return self.uncompressed_bytes()  # type: ignore

        compressed_bytes = self._compressed_bytes.get(compression)
        if compressed_bytes is None:
            if self.path is not None:
                if self._compression is None:
                    self._compression = get_compression(path=self.path)
                compressed_bytes = self._read_from_path()
                if self._compression is None:
                    self._compression = get_compression(header=compressed_bytes[:32])
                if self._compression == compression:
                    if self._shape is None:
                        _, self._shape, self._typestr = read_meta_from_compressed_file(
                            compressed_bytes, compression=self._compression
                        )
                else:
                    compressed_bytes = self._recompress(compressed_bytes, compression)
            elif self._buffer is not None:
                compressed_bytes = self._recompress(self._buffer, compression)
            else:
                compressed_bytes = compress_array(self.array, compression)
            self._compressed_bytes[compression] = compressed_bytes
        return compressed_bytes

    def _decompress(self, to_pil: bool = False):
        if not to_pil and self._array is not None:
            if self._uncompressed_bytes is None:
                self._uncompressed_bytes = self._array.tobytes()
            return
        compression = self.compression
        if compression is None and self._buffer is not None:
            if self.is_text_like:
                from deeplake.core.serialize import bytes_to_text

                buffer = bytes(self._buffer)
                self._array = bytes_to_text(buffer, self.htype)
            else:
                self._array = np.frombuffer(self._buffer, dtype=self.dtype).reshape(
                    self.shape
                )

        else:
            if self.path and get_path_type(self.path) == "local":
                compressed = self.path
            else:
                compressed = self.buffer

            if to_pil:
                self._pil = decompress_array(
                    compressed,
                    compression=compression,
                    shape=self.shape,
                    dtype=self.dtype,
                    to_pil=True,
                )  # type: ignore
            else:
                self._array = decompress_array(
                    compressed,
                    compression=compression,
                    shape=self.shape,
                    dtype=self.dtype,
                )
                self._uncompressed_bytes = self._array.tobytes()
                self._typestr = self._array.__array_interface__["typestr"]
                self._dtype = np.dtype(self._typestr).name

    def uncompressed_bytes(self) -> Optional[bytes]:
        """Returns uncompressed bytes."""
        self._decompress()
        return self._uncompressed_bytes

    @property
    def array(self) -> np.ndarray:  # type: ignore
        """Return numpy array corresponding to the sample. Decompresses the sample if necessary.

        Example:

            >>> sample = deeplake.read("./images/dog.jpg")
            >>> arr = sample.array
            >>> arr.shape
            (323, 480, 3)
        """
        arr = self._array
        if arr is not None:
            return arr
        self._decompress()
        return self._array  # type: ignore

    @property
    def pil(self) -> Image.Image:  # type: ignore
        """Return PIL image corresponding to the sample. Decompresses the sample if necessary.

        Example:

            >>> sample = deeplake.read("./images/dog.jpg")
            >>> pil = sample.pil
            >>> pil.size
            (480, 323)
        """
        pil = self._pil
        if pil is not None:
            return pil
        self._decompress(to_pil=True)
        return self._pil

    def __str__(self):
        if self.is_lazy:
            return f"Sample(is_lazy=True, path={self.path})"

        return f"Sample(is_lazy=False, shape={self.shape}, compression='{self.compression}', dtype='{self.dtype}' path={self.path})"

    def __repr__(self):
        return str(self)

    def __array__(self, dtype=None):
        arr = self.array
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    def __eq__(self, other):
        if self.path is not None and other.path is not None:
            return self.path == other.path
        return self.buffer == other.buffer

    def _read_from_path(self) -> bytes:  # type: ignore
        if self._buffer is None:
            path_type = get_path_type(self.path)
            if path_type == "local":
                self._buffer = self._read_from_local()
            elif path_type == "gcs":
                self._buffer = self._read_from_gcs()
            elif path_type == "s3":
                self._buffer = self._read_from_s3()
            elif path_type == "gdrive":
                self._buffer = self._read_from_gdrive()
            elif path_type == "http":
                self._buffer = self._read_from_http()
        return self._buffer  # type: ignore

    def _read_from_local(self) -> bytes:
        with open(self.path, "rb") as f:  # type: ignore
            return f.read()

    def _get_root_and_key(self, path):
        split_path = path.split("/", 2)
        if len(split_path) > 2:
            root, key = "/".join(split_path[:2]), split_path[2]
        else:
            root, key = split_path
        return root, key

    def _read_from_s3(self) -> bytes:
        assert self.path is not None
        if self.storage is not None:
            assert isinstance(self.storage, S3Provider)
            return self.storage.get_object_from_full_url(self.path)
        path = self.path.replace("s3://", "")  # type: ignore
        root, key = self._get_root_and_key(path)
        s3 = S3Provider(root, **self._creds)
        return s3[key]

    def _read_from_gcs(self) -> bytes:
        assert self.path is not None
        if GCSProvider is None:
            raise Exception(
                "GCP dependencies not installed. Install them with pip install deeplake[gcs]"
            )
        if self.storage is not None:
            assert isinstance(self.storage, GCSProvider)
            return self.storage.get_object_from_full_url(self.path)
        path = self.path.replace("gcp://", "").replace("gcs://", "")  # type: ignore
        root, key = self._get_root_and_key(path)
        gcs = GCSProvider(root, token=self._creds)
        return gcs[key]

    def _read_from_gdrive(self) -> bytes:
        assert self.path is not None
        gdrive = GDriveProvider("gdrive://", token=self._creds, makemap=False)
        return gdrive.get_object_from_full_url(self.path)

    def _read_from_http(self) -> bytes:
        assert self.path is not None
        if "Authorization" in self._creds:
            headers = {"Authorization": self._creds["Authorization"]}
        else:
            headers = {}
        result = requests.get(self.path, headers=headers)
        if result.status_code != 200:
            raise UnableToReadFromUrlError(self.path, result.status_code)
        return result.content

    def _getexif(self) -> dict:
        if self.path and get_path_type(self.path) == "local":
            img = Image.open(self.path)
        else:
            img = Image.open(BytesIO(self.buffer))
        try:
            return getexif(img)
        except Exception as e:
            warnings.warn(
                f"Error while reading exif data, possibly due to corrupt exif: {e}"
            )
            return {}

    @property
    def meta(self) -> dict:
        meta: Dict[str, Union[Dict, str]] = {}
        compression = self.compression
        compression_type = get_compression_type(compression)
        if compression == "dcm":
            meta.update(self._get_dicom_meta())
        elif compression_type == IMAGE_COMPRESSION:
            meta["exif"] = self._getexif()
        elif compression_type == VIDEO_COMPRESSION:
            meta.update(self._get_video_meta())
        elif compression_type == AUDIO_COMPRESSION:
            meta.update(self._get_audio_meta())
        elif compression_type in [POINT_CLOUD_COMPRESSION, MESH_COMPRESSION]:
            meta.update(self._get_point_cloud_meta())
        meta["shape"] = self.shape
        meta["format"] = self.compression
        if self.path:
            meta["filename"] = str(self.path)
        return meta


SampleValue = Union[np.ndarray, int, float, bool, Sample]
