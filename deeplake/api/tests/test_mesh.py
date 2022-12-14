import pytest

import deeplake
from deeplake.util.exceptions import DynamicTensorNumpyError


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
