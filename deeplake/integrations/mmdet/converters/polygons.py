import numpy as np
from typing import Any, List, Union, Tuple

from mmdet.core import BitmapMasks, PolygonMasks


def convert_to_coco_format(
    masks: Union[np.ndarray, List[np.ndarray]]
) -> List[List[float]]:
    """
    Convert masks to COCO format.

    Args:
        masks (Union[np.ndarray, List[np.ndarray]]): Masks to be converted.

    Returns:
        List[List[float]]: Masks in COCO format.
    """
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


def process(polygons: List[List]) -> List[np.ndarray]:
    """
    Convert polygons to list of ndarray and filter invalid polygons.

    Args:
        polygons (List[List]): Polygons of one instance.

    Returns:
        List[np.ndarray]: Processed polygons.
    """
    polygons = [np.array(p) for p in polygons]
    valid_polygons = []
    for polygon in polygons:
        if len(polygon) % 2 == 0 and len(polygon) >= 6:
            valid_polygons.append(polygon)
    return valid_polygons


def convert_polygons_to_mask(
    masks: List[np.ndarray], poly2mask: bool, shape: Tuple[int, int]
) -> Union[BitmapMasks, PolygonMasks]:
    """
    Convert polygons to mask instances (BitmapMasks or PolygonMasks) based on the poly2mask flag.

    Args:
        masks (List[np.ndarray]): Masks to be converted.
        poly2mask (bool): Flag to determine whether to convert masks to PolygonMasks.
        shape (Tuple[int, int]): Shape of the mask.

    Returns:
        Union[BitmapMasks, PolygonMasks]: Converted masks as BitmapMasks or PolygonMasks instances.
    """
    if poly2mask:
        masks = convert_to_coco_format(masks)
        masks = PolygonMasks(
            [process(polygons) for polygons in masks],
            shape[0],
            shape[1],
        )
    else:
        masks = BitmapMasks(masks.astype(np.uint8).transpose(2, 0, 1), *shape[:2])
    return masks
