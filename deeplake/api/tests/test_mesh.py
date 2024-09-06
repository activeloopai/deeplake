import pytest

import deeplake
import numpy as np
from deeplake.util.exceptions import (
    DynamicTensorNumpyError,
    MeshTensorMetaMissingRequiredValue,
    UnsupportedCompressionError,
)


def test_mesh(local_ds, mesh_paths):
    for i, (encoding_type, path) in enumerate(mesh_paths.items()):
        if encoding_type == "ascii2":
            pass
        tensor = local_ds.create_tensor(
            f"mesh_{i}", htype="mesh", sample_compression="ply"
        )
        sample = deeplake.read(path)
        tensor.append(deeplake.read(path))
        tensor_numpy = tensor.numpy()
        assert tensor_numpy.shape[-1] == 3

        tensor_data = tensor.data()
        assert isinstance(tensor_data, dict)

        tensor.append(deeplake.read(path))
        tensor.append(deeplake.read(path))
        tensor.append(deeplake.read(path))
        if encoding_type == "bin":
            with pytest.raises(DynamicTensorNumpyError):
                tensor.numpy()
        tensor_list = tensor.numpy(aslist=True)

        assert len(tensor_list) == 4

        tensor_data = tensor.data()
        assert len(tensor_data) == 4


def test_stl_mesh(local_ds, stl_mesh_paths):
    tensor = local_ds.create_tensor("stl_mesh", htype="mesh", sample_compression="stl")

    with pytest.raises(UnsupportedCompressionError):
        local_ds.create_tensor("unsupported", htype="mesh", sample_compression=None)

    with pytest.raises(MeshTensorMetaMissingRequiredValue):
        local_ds.create_tensor("unsupported", htype="mesh")

    for i, (_, path) in enumerate(stl_mesh_paths.items()):
        sample = deeplake.read(path)
        tensor.append(sample)
        tensor.append(deeplake.read(path))

    tensor_numpy = tensor.numpy()
    assert tensor_numpy.shape == (4, 12, 3, 3)
    assert np.all(tensor_numpy[0] == tensor_numpy[1])
    assert np.all(tensor_numpy[1] == tensor_numpy[2])
    assert np.all(tensor_numpy[2] == tensor_numpy[3])

    tensor_data = tensor.data()
    tensor_0_data = tensor[0].data()
    assert np.all(tensor_data["vertices"][0] == tensor_0_data["vertices"])
