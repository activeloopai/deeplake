import numpy as np


def convert_poly_to_coco_format(masks):
    if isinstance(masks, np.ndarray):
        px = masks[..., 0]
        py = masks[..., 1]
        poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
        poly = [[float(p) for x in poly for p in x]]
        return poly
    poly = []
    for mask in masks:
        poly_i = convert_poly_to_coco_format(mask)
        poly.append([np.array(poly_i[0])])
    return poly