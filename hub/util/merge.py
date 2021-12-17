def merge(dataset, changes, target_commit_id):
    """Merge changes into dataset."""
    changed = False
    ds2 = dataset._copy()
    ds2.checkout(target_commit_id)
    with dataset as ds:
        ds_keys = set(ds.tensors.keys())
        for tensor, change in changes.items():
            if tensor not in ds_keys:
                tensor_meta = ds2.tensors[tensor].chunk_engine.tensor_meta
                dtype, htype = tensor_meta.dtype, tensor_meta.htype
                sample_compression = tensor_meta.sample_compression
                chunk_compression = tensor_meta.chunk_compression
                ds.create_tensor(
                    tensor,
                    htype=htype,
                    dtype=dtype,
                    sample_compression=sample_compression,
                    chunk_compression=chunk_compression,
                )
                changed = True

            inplace_transform = change.get("data_transformed_in_place", False)
            if inplace_transform:
                raise NotImplementedError(
                    "Merging commits that had inplace transforms is not implemented yet."
                )

            data_added = change.get("data_added")
            if data_added is not None:
                for idx in range(*data_added):
                    ds[tensor].append(ds2[tensor][idx])  # tensor pass through
                    changed = True
            data_updated = change.get("data_updated")
            if data_updated is not None:
                for idx in data_updated:
                    ds[tensor][idx] = ds2[tensor][idx]  # tensor pass through
                    changed = True
    return changed
