import deeplake
import numpy as np
from deeplake.tests.common import requires_libdeeplake


@requires_libdeeplake
def test_index_creation(local_ds_generator):
    deeplake_ds = local_ds_generator()
    with deeplake_ds:
        deeplake_ds.create_tensor("embedding", htype="embedding", dtype=np.float32)
        for i in range(200):
            deeplake_ds.embedding.append(
                np.random.random_sample(
                    384,
                )
            )

    deeplake_ds.embedding.create_index("hnsw_1")
