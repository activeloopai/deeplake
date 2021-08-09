import numpy as np
import mmh3
from hub.core.sample import Sample, SampleValue  # type: ignore
from hub.core.serialize import serialize_input_samples
from hub.core.meta.tensor_meta import TensorMeta

from typing import List, Sequence, Union, Optional, Tuple, Any

def generate_hashes(samples: Union[np.ndarray, Sequence[SampleValue]]):
    """ Generate 128-bit murmurhash3 of samples """

    for sample in samples:
        
        hashed_sample = mmh3.hash_bytes(sample.uncompressed_bytes())
        sample = hashed_sample.hex()
        print("Hash: ", sample)

    return samples