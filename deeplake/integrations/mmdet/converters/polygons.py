import numpy as np

from mmdet.core import BitmapMasks, PolygonMasks


def convert_to_coco_format(masks):
    if isinstance(masks, np.ndarray):
        px = masks[..., 0]
        py = masks[..., 1]
        poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
        poly = [[float(p) for x in poly for p in x]]
        return poly
    poly = []
    for mask in masks:
        poly_i = convert_to_coco_format(mask)
        poly.append([np.array(poly_i[0])])
    return poly


def process(polygons):
    """Convert polygons to list of ndarray and filter invalid polygons.

    Args:
        polygons (list[list]): Polygons of one instance.

    Returns:
        list[numpy.ndarray]: Processed polygons.
    """

    polygons = [np.array(p) for p in polygons]
    valid_polygons = []
    for polygon in polygons:
        if len(polygon) % 2 == 0 and len(polygon) >= 6:
            valid_polygons.append(polygon)
    return valid_polygons


def convert_polygons_to_mask(masks, poly2mask, shape):
    if poly2mask:
        masks = convert_to_coco_format(masks)
        masks = PolygonMasks(
            [process(polygons) for polygons in masks],
            shape[0],
            shape[1],
        )
    else:
        masks = BitmapMasks(
            masks.astype(np.uint8).transpose(2, 0, 1), *shape[:2]
        )