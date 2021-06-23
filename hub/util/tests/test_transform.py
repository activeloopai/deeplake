import numpy as np
import hub
from hub.util.transform import merge_tensor_metas
from hub.core.meta.tensor_meta import TensorMeta
from hub.core.tests.common import parametrize_all_dataset_storages


@parametrize_all_dataset_storages
def test_merge_tensor_metas(ds):
    ds.create_tensor("id")
    ds.create_tensor("image", htype="image")
    ds.flush()

    id_dict = {
        "htype": "generic",
        "chunk_size": 16000000,
        "min_shape": [11],
        "max_shape": [11],
        "length": 1,
        "sample_compression": "uncompressed",
        "chunk_compression": "uncompressed",
        "dtype": "int64",
        "custom_meta": {},
        "version": "2.0a3",
    }

    image_dict = {
        "htype": "image",
        "chunk_size": 16000000,
        "min_shape": [32, 32, 3],
        "max_shape": [32, 32, 3],
        "length": 1,
        "sample_compression": "png",
        "chunk_compression": "uncompressed",
        "dtype": "uint8",
        "custom_meta": {},
        "version": "2.0a3",
    }
    all_workers_metas = [{"id": id_dict, "image": image_dict}]
    new_id_dict = id_dict.copy()
    new_id_dict.update({"min_shape": [5], "max_shape": [5], "length": 3})
    all_workers_metas.append(
        {"id": new_id_dict, "image": image_dict},
    )
    merge_tensor_metas(all_workers_metas, ds.storage, ("id", "image"))
    assert ds.id.meta.length == 4
    assert ds.id.meta.min_shape == [5]
    assert ds.id.meta.max_shape == [11]
    assert ds.image.meta.sample_compression == "png"
    assert ds.image.meta.length == 2
    ds.delete()
