import numpy as np
import mmh3
from hub.core.sample import Sample, SampleValue  # type: ignore
from hub.core.serialize import serialize_input_samples
from hub.core.meta.tensor_meta import TensorMeta

from typing import List, Sequence, Union


def generate_hashes(samples: Union[np.ndarray, Sequence[SampleValue]]):
    """Generate two unsigned 64-bit murmurhash3 of samples
    Args:
        samples (Union[np.ndarray, Sequence[SampleValue]): Name of tfds dataset.You can see a list of all tfds datasets here:
            https://www.tensorflow.org/datasets/catalog/overview
    
    Returns:
        A Python list containg two unsigned 64-bit hashes for each sample in numpy aray format
    """
    hashlist = []

    for sample in samples:

        if isinstance(sample, Sample):
            hashed_sample = mmh3.hash64(sample.uncompressed_bytes(), signed=False)
        else:
            hashed_sample = mmh3.hash64(sample.tobytes(), signed=False)

        hashlist.append(np.array(hashed_sample))

    return hashlist