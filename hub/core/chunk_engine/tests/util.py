from copy import deepcopy

import numpy as np

# from hub.core.chunk_engine.read import read_data_from_chunks
# from hub.core.chunk_engine.tests.util import get_random_shaped_array
# from hub.core.chunk_engine.write import chunk_and_write_data


def get_random_shaped_array(random_max_shape, dtype, fixed=False):
    """
    Get a randomly shaped array that is between (1, 1...) & random_max_shape.

    Args:
        random_max_shape(tuple): The returned array will have the dimensionality of random_max_shape & each dim will be selected randomly between 1 & it's value. If `fixed=True`, dims are not random.
        dtype(str): Datatype for the random array.
        fixed(bool): If True, the random array will be of shape random_max_shape instead of a random selection between 1 & it's value.

    Returns:
        numpy array with the provided specifications.
    """

    if fixed:
        dims = random_max_shape
    else:
        dims = []
        if type(random_max_shape) != int:
            for max_dim in random_max_shape:
                dims.append(np.random.randint(1, max_dim + 1))

    if "int" in dtype:
        low = np.iinfo(dtype).min
        high = np.iinfo(dtype).max
        a = np.random.randint(low=low, high=high, size=dims, dtype=dtype)
    elif "float" in dtype:
        a = np.random.random_sample(size=dims).astype(dtype)
    elif "bool" in dtype:
        a = np.random.uniform(size=dims)
        a = a > 0.5

    return a


def make_dummy_byte_array(length):
    """Generate a random bytearray of the provided length."""
    content = bytearray()
    a = np.random.randint(128, size=length)
    content.extend(a.tolist())
    assert len(content) == length
    return content


def get_random_chunk_size():
    return np.random.choice([8, 256, 1024, 4096])


def get_random_compressor():
    return np.random.choice([None, "gzip"])  # TODO more compressors


def get_random_compressor_subject():
    return np.random.choice(["sample"])  # TODO chunk


def get_random_dtype():
    return np.random.choice(
        [
            "bool",
            # uint
            "uint",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            # int
            "int",
            "int8",
            "int16",
            "int32",
            "int64",
            # float
            "float",
            "float16",
            "float32",
            "float64",
        ]
    )


def get_random_num_samples():
    return np.random.randint(1, 300)


def get_random_partial(chunk_size):
    return np.random.randint(1, chunk_size - 1)


def get_random_data(n, dtype, max_shape, fixed=False):
    X = []
    for _ in range(n):
        x = get_random_shaped_array(max_shape, dtype, fixed=fixed)
        X.append(x)
    return X


def assert_read_and_write_is_valid(config, data=None, fixed=False):
    if type(config) == dict:
        config = deepcopy(config)  # make sure no duplicate references

        n = len(data)
        key = "test_tensor"
        backend = config["backend"]

        for X in data:
            chunk_and_write_data(key, X, **config)

        for i in range(n):
            original_sample = data[i]
            sample = read_data_from_chunks(key, i, **config)
            np.testing.assert_array_equal(original_sample, sample)

        imc = backend.get_index_map_count(key)
        assert imc == n, "%i != %i cfg: %s" % (imc, n, str(config))
    else:
        if data is not None:
            raise Exception("if config is a list, data is expected to be None")
        for cfg in config:
            data = get_random_data(
                cfg["n"], cfg["dtype"], cfg["random_max_shape"], fixed=fixed
            )
            assert_read_and_write_is_valid(cfg, data=data, fixed=fixed)
