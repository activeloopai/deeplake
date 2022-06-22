from typing import List

import numpy as np


def convert_to_idx(samples, class_names: List[str]):
    idxs = []
    additions = []
    for sample in samples:
        if isinstance(sample, str):
            for i in range(len(class_names)):
                if class_names[i] == sample:
                    idxs.append(i)
                    break
            else:
                class_names.append(sample)
                idx = len(class_names) - 1
                idxs.append(idx)
                additions.append((sample, idx))
        elif isinstance(sample, list):
            idxs_, additions_ = convert_to_idx(sample, class_names)
            idxs.append(idxs_)
            additions.extend(additions_)
        else:
            idxs.append(sample)
    return idxs, additions


def convert_to_text(inp, class_names: List[str]):
    if isinstance(inp, np.integer):
        idx = int(inp)
        if idx < len(class_names):
            return class_names[idx]
        return None
    return [convert_to_text(item, class_names) for item in inp]
