

from hub.util.tiles import num_tiles_for_sample
from hub.constants import MB
from hub.core.meta.tensor_meta import TensorMeta
from hub.core.tiling.optimize import optimize_tile_shape

import matplotlib.pyplot as plt


if __name__ == "__main__":
    # tensor_meta = TensorMeta(htype="image", sample_compression="png", dtype="int32")
    tensor_meta = TensorMeta(htype="image", sample_compression=None, dtype="int32")

    tensor_meta.max_chunk_size = 32 * MB
    tensor_meta.min_chunk_size = 16 * MB

    # sample_shape = (10_000, 5_000, )
    # sample_shape = (10_000, 3)
    # sample_shape = (100_000_000, )
    sample_shape = (100, 5000, 5000, 3, 5)

    tile_shape, history = optimize_tile_shape(sample_shape, tensor_meta, validate=False, return_history=True)
    

    plt.plot([state["cost"] for state in history])
    plt.show()

    for state in history:
        print(state)

    print(f"found tile_shape={tile_shape} for sample_shape={sample_shape}")
    print("num tiles", num_tiles_for_sample(tile_shape, sample_shape))