import re
from typing import List, Tuple, Union

import exiftool  # type: ignore
import numpy as np
from hub.core.chunk_engine.read import sample_from_index_entry
from hub.core.chunk_engine.write import write_bytes
from hub.core.flatten import row_wise_to_bytes
from hub.core.index import Index
from hub.core.meta.index_map import read_index_map, write_index_map
from hub.core.meta.tensor_meta import (
    read_tensor_meta,
    update_tensor_meta_with_array,
    validate_tensor_meta,
    write_tensor_meta,
)
from hub.core.typing import StorageProvider
from hub.util.array import normalize_and_batchify_array_shape
from hub.util.dataset import get_compressor
from hub.util.exceptions import (
    DynamicTensorNumpyError,
    ImageReadError,
    TensorAlreadyExistsError,
    TensorDoesNotExistError,
    TensorInvalidSampleShapeError,
    TensorMetaMismatchError,
    WrongMetadataError,
)
from hub.util.keys import get_index_map_key, get_tensor_meta_key
from PIL import Image  # type: ignore


def tensor_exists(key: str, storage: StorageProvider) -> bool:
    """A tensor exists if at the specified `key` and `storage` there is both a tensor meta file and index map."""

    meta_key = get_tensor_meta_key(key)
    index_map_key = get_index_map_key(key)
    return meta_key in storage and index_map_key in storage


def create_tensor(key: str, storage: StorageProvider, meta: dict):
    """If a tensor does not exist, create a new one with the provided meta.

    Args:
        key (str): Key for where the chunks, index_map, and meta will be located in `storage` relative to it's root.
        storage (StorageProvider): StorageProvider that all tensor data is written to.
        meta (dict): Meta for the tensor. For required properties, see `default_tensor_meta`.

    Raises:
        TensorAlreadyExistsError: If a tensor defined with `key` already exists.
    """

    if tensor_exists(key, storage):
        raise TensorAlreadyExistsError(key)

    validate_tensor_meta(meta)

    write_tensor_meta(key, storage, meta)
    write_index_map(key, storage, [])


def add_samples_to_tensor(
    array: np.ndarray,
    key: str,
    storage: StorageProvider,
    batched: bool = False,
):
    """Adds samples to a tensor that already exists. `array` is chunked and sent to `storage`.
        For more on chunking, see the `generate_chunks` method.

    Args:
        array (np.ndarray): Array to be chunked/written. Batch axis (`array.shape[0]`) is optional, if `array` does
            have a batch axis, you should pass the argument `batched=True`.
        key (str): Key for where the chunks, index_map, and meta will be located in `storage` relative to it's root.
        storage (StorageProvider): StorageProvider for storing the chunks, index_map, and meta.
        batched (bool): If True, the provided `array`'s first axis (`shape[0]`) will be considered it's batch axis.
            If False, a new axis will be created with a size of 1 (`array.shape[0] == 1`). default=False.

    Raises:
        TensorDoesNotExistError: If a tensor at `key` does not exist. A tensor must be created first using
            `create_tensor(...)`.
    """

    if not tensor_exists(key, storage):
        raise TensorDoesNotExistError(key)

    index_map = read_index_map(key, storage)
    tensor_meta = read_tensor_meta(key, storage)

    array = normalize_and_batchify_array_shape(array, batched=batched)
    if "min_shape" not in tensor_meta:
        tensor_meta = update_tensor_meta_with_array(tensor_meta, array, batched=True)

    _check_array_and_tensor_are_compatible(tensor_meta, array)

    # TODO: get the tobytes function from meta
    tobytes = row_wise_to_bytes

    array_length = array.shape[0]
    for i in range(array_length):
        sample = array[i]

        if 0 in sample.shape:
            # if sample has a 0 in the shape, no data will be written
            index_map_entry = {"chunk_names": []}  # type: ignore

        else:
            # TODO: we may want to call `tobytes` on `array` and call memoryview on that. this may depend on the access patterns we
            # choose to optimize for.
            b = tobytes(sample)
            if not tensor_meta.get("is_compressed", False):
                compressor = get_compressor(tensor_meta["compression"])
                if compressor:
                    b = compressor.encode(b)
            b = memoryview(b)
            # b = memoryview(tobytes(sample))
            index_map_entry = write_bytes(
                b, key, tensor_meta["chunk_size"], storage, index_map
            )
            index_map_entry["compression"] = tensor_meta["compression"]
            index_map_entry["is_compressed"] = True

        index_map_entry["shape"] = sample.shape
        _update_tensor_meta_shapes(sample.shape, tensor_meta)
        index_map.append(index_map_entry)

    tensor_meta["length"] += array_length

    write_tensor_meta(key, storage, tensor_meta)
    write_index_map(key, storage, index_map)


def add_index_map_to_tensor(
    index_map_dict: dict,
    key: str,
    storage: StorageProvider,
):
    """Adds samples to a tensor that already exists. bytes of index_map are chunked and sent to `storage`.
    For more on chunking, see the `generate_chunks` method.

    Args:
        index_map_dict (dict): Bytes of the image that should be chunked and written to the storage.
        key (str): Key for where the chunks, index_map, and meta will be located in `storage` relative to it's root.
        storage (StorageProvider): StorageProvider for storing the chunks, index_map, and meta.

    Raises:
        TensorDoesNotExistError: If a tensor at `key` does not exist. A tensor must be created first using
            `create_tensor(...)`.
    """
    if not tensor_exists(key, storage):
        raise TensorDoesNotExistError(key)

    index_map = read_index_map(key, storage)
    tensor_meta = read_tensor_meta(key, storage)
    shape = index_map_dict["shape"]
    if "min_shape" not in tensor_meta:
        tensor_meta["dtype"] = str(index_map_dict["dtype"])
        tensor_meta["min_shape"] = shape
        tensor_meta["max_shape"] = shape
    index_map_entry = {"chunk_names": []}  # type: ignore
    index_map_entry = write_bytes(
        index_map_dict["bytes"], key, tensor_meta["chunk_size"], storage, index_map
    )
    index_map_entry["compression"] = index_map_dict["compression"]
    index_map_entry["is_compressed"] = index_map_dict["is_compressed"]
    index_map_entry["dtype"] = index_map_dict["dtype"]
    index_map_entry["shape"] = shape
    _update_tensor_meta_shapes(shape, tensor_meta)
    index_map.append(index_map_entry)

    tensor_meta["length"] += 1
    write_tensor_meta(key, storage, tensor_meta)
    write_index_map(key, storage, index_map)


def read_samples_from_tensor(
    key: str,
    storage: StorageProvider,
    index: Index = Index(),
    aslist: bool = False,
) -> Union[np.ndarray, List[np.ndarray]]:
    """Read (and unpack) samples from a tensor as an np.ndarray.

    Args:
        key (str): Key for where the chunks, index_map, and meta are located in `storage` relative to it's root.
        storage (StorageProvider): StorageProvider for reading the chunks, index_map, and meta.
        index (Index): Index that represents which samples to read.
        aslist (bool): If True, a list of np.ndarrays will be returned. Helpful for dynamic tensors.
            If False, a single np.ndarray will be returned unless the samples are dynamically shaped, in which case
            an error is raised.

    Raises:
        DynamicTensorNumpyError: If reading a dynamically-shaped array slice without `aslist=True`.
        NotImplementedError: Empty samples (shape contains 0) not implemented.

    Returns:
        np.ndarray: Array containing the sample(s) in the `array_slice` slice.
    """

    meta = read_tensor_meta(key, storage)
    index_map = read_index_map(key, storage)
    index_entries = [index_map[i] for i in index.values[0].indices(len(index_map))]

    dtype = meta["dtype"]
    # TODO: read samples in parallel
    samples = []
    for i, index_entry in enumerate(index_entries):
        shape = index_entry["shape"]
        compressor = get_compressor(index_entry["compression"])
        # check if all samples are the same shape
        last_shape = index_entries[i - 1]["shape"]
        if not aslist and shape != last_shape:
            raise DynamicTensorNumpyError(key, index)

        if 0 in shape:
            # TODO: implement support for 0s in shape and update docstring
            raise NotImplementedError("0s in shapes are not supported yet.")

        array = sample_from_index_entry(key, storage, index_entry, dtype, compressor)
        samples.append(array)
    if aslist:
        if index.values[0].subscriptable():
            return samples
        else:
            return samples[0]

    return index.apply(np.array(samples))


def _check_array_and_tensor_are_compatible(tensor_meta: dict, array: np.ndarray):
    """An array is considered incompatible with a tensor if the `tensor_meta` entries don't match the `array` properties.

    Args:
        tensor_meta (dict): Tensor meta containing the expected properties of `array`.
        array (np.ndarray): Candidate array to check compatibility with `tensor_meta`.

    Raises:
        TensorMetaMismatchError: When `array` properties do not match the `tensor_meta`'s exactly. Also when
            `len(array.shape)` != len(tensor_meta max/min shapes).
        TensorInvalidSampleShapeError: All samples must have the same dimensionality (`len(sample.shape)`).
    """

    if tensor_meta["dtype"] != array.dtype.name:
        raise TensorMetaMismatchError("dtype", tensor_meta["dtype"], array.dtype.name)

    sample_shape = array.shape[1:]

    expected_shape_len = len(tensor_meta["min_shape"])
    actual_shape_len = len(sample_shape)
    if expected_shape_len != actual_shape_len:
        raise TensorInvalidSampleShapeError(
            "Sample shape length is expected to be {}, actual length is {}.".format(
                expected_shape_len, actual_shape_len
            ),
            sample_shape,
        )


def read(image_path: str, check_meta: bool = True):
    """
    Get image bytes and metadata.

    Args:
        image_path (str): Path to the image file.
        check_meta (bool): If True, check if image can be read and image metadata
            information corresponds to the actual image parameters.

    Raises:
        ImageReadError: If image can't be opened by PIL.Image.open()
        WrongMetadataError: If any parameter from metadata doesn't match the image.

    Returns:
        Dictionary containing image bytes, extension, dtype and shape.
    """
    with exiftool.ExifTool() as et:
        metadata = et.get_metadata(image_path)
    with open(image_path, "rb") as image_file:
        image_bytes = image_file.read()
    meta_size = tuple(map(int, re.split("x| ", metadata["Composite:ImageSize"])))
    for meta_key, meta_value in metadata.items():
        if meta_key.endswith("FileType"):
            meta_extension = meta_value
        elif "ColorComponents" in meta_key:
            meta_channels = meta_value
        elif "PNG:ColorType" in meta_key:
            color_type = int(meta_value)
            if color_type == 6:
                meta_channels = 4
            elif color_type == 4:
                meta_channels = 2
            elif color_type == 2:
                meta_channels = 3
            else:
                meta_channels = 1
        elif "BitDepth" in meta_key or "BitsPerSample" in meta_key:
            meta_dtype = "uint" + str(meta_value)
    if check_meta:
        try:
            image = Image.open(image_path)
        except Exception as e:
            raise ImageReadError(image_path, e)
        image_arr = np.asarray(image)
        # print(image_arr.shape)
        image_dtype = image_arr.dtype
        if image.mode == "RGB":
            image_channels = 3
        elif image.mode == "RGBA":
            image_channels = 4
        else:
            image_channels = 1
        image_size = image.size
        image_extension = image.format
        if (
            meta_size != image_size
            or meta_extension != image_extension
            or meta_channels != image_channels
            or meta_dtype != image_dtype
        ):
            raise WrongMetadataError(image_path)
    return {
        "bytes": image_bytes,
        "name": image_path.split("/")[-1],
        "dtype": meta_dtype,
        "shape": meta_size + (meta_channels,),
        "compression": meta_extension,
        "is_compressed": True,
    }


def _update_tensor_meta_shapes(shape: Tuple[int], tensor_meta: dict):
    tensor_meta["min_shape"] = tuple(np.minimum(tensor_meta["min_shape"], shape))
    tensor_meta["max_shape"] = tuple(np.maximum(tensor_meta["max_shape"], shape))
