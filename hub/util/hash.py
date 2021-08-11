import numpy as np
import mmh3
from hub.core.sample import Sample, SampleValue  # type: ignore
from hub.core.serialize import serialize_input_samples
from hub.core.meta.tensor_meta import TensorMeta

from typing import List, Sequence, Union

def generate_hashes(samples: Union[np.ndarray, Sequence[SampleValue]]):
    """ Generate two unsigned 64-bit murmurhash3 of samples """
    
    hashlist = []

    for sample in samples:
        
        hashed_sample = mmh3.hash64(sample.uncompressed_bytes(), signed=False)
        hashlist.append(np.array(hashed_sample))
        
    return hashlist