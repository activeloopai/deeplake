from deeplake.core.dataset import Dataset
import numpy as np


def check_rechunk(ds: "Dataset"):
    rechunk = []
    for key, tensor in ds.tensors.items():
        if not tensor.meta.sample_compression and not tensor.meta.chunk_compression:
            engine = tensor.chunk_engine
            num_chunks = engine.num_chunks
            if num_chunks > 1:
                num_samples = engine.num_samples
                max_shape = tensor.meta.max_shape
                if len(max_shape) > 0:
                    nbytes = (
                        np.prod([num_samples] + max_shape)
                        * np.dtype(tensor.dtype).itemsize
                    )
                    avg_chunk_size = nbytes / num_chunks

                    if avg_chunk_size < 0.1 * engine.min_chunk_size:
                        rechunk.append(key)
    return rechunk
