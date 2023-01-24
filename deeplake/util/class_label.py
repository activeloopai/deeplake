from collections import OrderedDict, defaultdict
from typing import List

from deeplake.util.hash import hash_str_to_int32
from deeplake.client.log import logger
import numpy as np
import deeplake


def convert_to_idx(samples, class_names: List[str]):
    class_idx = {class_names[i]: i for i in range(len(class_names))}

    def convert(samples):
        idxs = []
        additions = []
        for sample in samples:
            if isinstance(sample, np.ndarray):
                sample = sample.tolist()
            if isinstance(sample, str):
                idx = class_idx.get(sample)
                if idx is None:
                    idx = len(class_idx)
                    class_idx[sample] = idx
                    additions.append((sample, idx))
                idxs.append(idx)
            elif isinstance(sample, list):
                idxs_, additions_ = convert(sample)
                idxs.append(idxs_)
                additions.extend(additions_)
            else:
                idxs.append(sample)
        return idxs, additions

    return convert(samples)


def convert_to_hash(samples, hash_label_map):
    if isinstance(samples, np.ndarray):
        samples = samples.tolist()
    if isinstance(samples, list):
        return [convert_to_hash(sample, hash_label_map) for sample in samples]
    else:
        if isinstance(samples, str):
            hash_ = hash_str_to_int32(samples)
            hash_label_map[hash_] = samples
        else:
            hash_ = samples
        return hash_


def convert_hash_to_idx(hashes, hash_idx_map):
    if isinstance(hashes, list):
        return [convert_hash_to_idx(hash, hash_idx_map) for hash in hashes]
    else:
        try:
            return hash_idx_map[hashes]
        except KeyError:
            return hashes


def convert_to_text(inp, class_names: List[str], return_original=False):
    if isinstance(inp, np.integer):
        idx = int(inp)
        if idx < len(class_names):
            return class_names[idx]
        return idx if return_original else None
    return [convert_to_text(item, class_names) for item in inp]


def sync_labels(
    ds, label_temp_tensors, hash_label_maps, num_workers, scheduler, verbose=True
):
    hl_maps = defaultdict(OrderedDict)
    for map in hash_label_maps:
        for tensor in map:
            hl_maps[tensor].update(map[tensor])
    hash_label_maps = hl_maps

    @deeplake.compute
    def class_label_sync(
        hash_tensor_sample,
        samples_out,
        label_tensor: str,
        hash_idx_map,
    ):
        hashes = hash_tensor_sample.numpy().tolist()
        idxs = convert_hash_to_idx(hashes, hash_idx_map)
        samples_out[label_tensor].append(idxs)

    for tensor, temp_tensor in label_temp_tensors.items():
        if len(ds[temp_tensor]) == 0:
            ds.delete_tensor(temp_tensor, large_ok=True)
        else:
            try:
                target_tensor = ds[tensor]
                hash_label_map = hash_label_maps[temp_tensor]
                class_names = target_tensor.info.class_names
                new_labels = [
                    label
                    for label in hash_label_map.values()
                    if label not in class_names
                ]
                if verbose:
                    N = len(class_names)
                    for i in range(len(new_labels)):
                        logger.info(
                            f"'{new_labels[i]}' added to {tensor}.info.class_names at index {N + i}"
                        )
                class_names.extend(new_labels)
                label_idx_map = {class_names[i]: i for i in range(len(class_names))}
                hash_idx_map = {
                    hash: label_idx_map[hash_label_map[hash]] for hash in hash_label_map
                }
                target_tensor.info.is_dirty = True
                target_tensor.meta._disable_temp_transform = True
                target_tensor.meta.is_dirty = True

                logger.info("Synchronizing class labels...")
                class_label_sync(label_tensor=tensor, hash_idx_map=hash_idx_map).eval(
                    ds[temp_tensor],
                    ds,
                    num_workers=num_workers,
                    scheduler=scheduler,
                    progressbar=True,
                    check_lengths=False,
                    skip_ok=True,
                )
                target_tensor.meta._disable_temp_transform = False
            finally:
                ds.delete_tensor(temp_tensor, large_ok=True)
