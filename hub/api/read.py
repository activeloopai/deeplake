from hub.core.sample import Sample  # type: ignore


def read(path: str, verify: bool = False, convert_grayscale: bool = True) -> Sample:
    """Utility that reads raw data from a file into a `np.ndarray` in 1 line of code. Also provides access to all important metadata.

    Note:
        No data is actually loaded until you try to get a property of the returned `Sample`. This is useful for passing along to
            `tensor.append` and `tensor.extend`.

    Examples:
        >>> sample = hub.read("path/to/cat.jpeg")
        >>> type(sample.array)
        <class 'numpy.ndarray'>
        >>> sample.compression
        'jpeg'

    Supported File Types:
        image: png, jpeg, and all others supported by `PIL`: https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#fully-supported-formats

    Args:
        path (str): Path to a supported file.
        verify (bool):  If True, contents of the file are verified.
        convert_grayscale: If True, and if the rest of the dataset is in color (3D), then
                           reshape a grayscale image by appending a 1 to its shape.

    Returns:
        Sample: Sample object. Call `sample.array` to get the `np.ndarray`.
    """

    sample = Sample(path, verify=verify)
    sample._convert_grayscale = convert_grayscale
    return sample
