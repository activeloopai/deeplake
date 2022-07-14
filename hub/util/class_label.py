from collections import defaultdict
from typing import List

from hub.util.hash import hash_str_to_int32
import numpy as np
import hub


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
    def convert(samples):
        hashes = []
        for sample in samples:
            if isinstance(sample, np.ndarray):
                sample = sample.tolist()
            if isinstance(sample, list):
                hashes_ = convert(sample)
                hashes.extend(hashes_)
            else:
                hash_ = hash_str_to_int32(sample) if isinstance(sample, str) else sample
                hash_label_map[hash_] = sample
                hashes.append(hash_)
        return hashes

    return convert(samples)


def convert_hash_to_idx(hashes, hash_label_map, label_idx_map, class_names):
    def convert(hashes):
        idxs = []
        for hash in hashes:
            if isinstance(hash, list):
                idxs_ = convert(hash)
                idxs.extend(idxs_)
            else:
                label = hash_label_map[hash]
                if isinstance(label, str):
                    idx = label_idx_map.get(label)
                    if idx is not None:
                        idxs.append(idx)
                    else:
                        idx = len(class_names)
                        idxs.append(idx)
                        class_names.append(label)
                        label_idx_map[label] = idx
                else:
                    idxs.append(label)
        return idxs

    return convert(hashes)


def convert_to_text(inp, class_names: List[str]):
    if isinstance(inp, np.integer):
        idx = int(inp)
        if idx < len(class_names):
            return class_names[idx]
        return None
    return [convert_to_text(item, class_names) for item in inp]


def sync_labels(ds, label_temp_tensors, hash_label_maps, num_workers, scheduler):
    hl_maps = defaultdict(dict)
    for map in hash_label_maps:
        for tensor in map:
            hl_maps[tensor].update(map[tensor])
    hash_label_maps = hl_maps

    @hub.compute
    def upload(items, ds, tensor):
        ds[tensor].append(items)
        return ds

    for tensor, temp_tensor in label_temp_tensors.items():
        target_tensor = ds[tensor]
        hash_label_map = hash_label_maps[temp_tensor]
        class_names = target_tensor.info.class_names
        label_idx_map = {class_names[i]: i for i in range(len(class_names))}
        start_idx = len(target_tensor)
        items = ds[temp_tensor][start_idx:].numpy().tolist()
        idxs = convert_hash_to_idx(items, hash_label_map, label_idx_map, class_names)
        target_tensor.info.is_dirty = True
        target_tensor.meta._disable_temp_transform = True
        target_tensor.meta.is_dirty = True

        upload(tensor=tensor).eval(
            idxs,
            ds,
            num_workers=num_workers,
            scheduler=scheduler,
            progressbar=False,
            check_lengths=False,
            skip_ok=True,
        )
        target_tensor.meta._disable_temp_transform = False
        target_tensor.meta.is_dirty = True
        ds.delete_tensor(temp_tensor)
