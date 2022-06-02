import numpy as np


def numeric_to_text(inp, class_names: list[str]):
    if isinstance(inp, np.integer):
        idx = int(inp)
        if idx < len(class_names):
            return class_names[idx]
        return None
    return [numeric_to_text(item, class_names) for item in inp]
