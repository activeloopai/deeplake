from typing import List

import numpy as np


def convert_to_idx(samples, class_names: List[str]):
    class_idx = {class_names[i]: i for i in range(len(class_names))}

    def convert(samples):
        idxs = []
        additions = []
        for sample in samples:
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


def convert_to_text(inp, class_names: List[str]):
    if isinstance(inp, np.integer):
        idx = int(inp)
        if idx < len(class_names):
            return class_names[idx]
        return None
    return [convert_to_text(item, class_names) for item in inp]
