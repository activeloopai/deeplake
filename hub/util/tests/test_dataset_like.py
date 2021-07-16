import hub


def test_like(local_ds):
    local_ds.create_tensor("image", htype="image", sample_compression="png")
    local_ds.create_tensor("label", htype="class_label", dtype="uint8")

    new_ds_1 = hub.dataset_like(local_ds.path + "_test", like=local_ds)
    new_ds_2 = hub.dataset_like(local_ds.path + "_test_1", like=local_ds.path)
    assert local_ds.meta == new_ds_1.meta
    assert local_ds.meta == new_ds_2.meta

    assert len(new_ds_1.tensors) == len(new_ds_2.tensors)

    for tensor_name in new_ds_1.tensors.keys():
        assert local_ds[tensor_name].meta == new_ds_1[tensor_name].meta
    new_ds_1.delete()
    new_ds_2.delete()
