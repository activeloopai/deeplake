import numpy as np
from PIL import Image, ImageDraw  # type: ignore
from typing import Any, Dict, List, Tuple


def coco_pixel_2_pascal_pixel(boxes: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert bounding boxes from COCO format (top-left x, top-left y, width, height)
    to Pascal VOC format (top-left x, top-left y, bottom-right x, bottom-right y)
    in pixel coordinates.

    Args:
        boxes (np.ndarray): Array of bounding boxes in COCO format.
        shape (Tuple[int, int]): The shape of the image as (height, width).

    Returns:
        np.ndarray: Array of bounding boxes in Pascal VOC format.
    """
    pascal_boxes = np.empty((0, 4), dtype=boxes.dtype)
    if boxes.size != 0:
        pascal_boxes = np.stack(
            (
                boxes[:, 0],
                boxes[:, 1],
                boxes[:, 0] + boxes[:, 2],
                boxes[:, 1] + boxes[:, 3],
            ),
            axis=1,
        )
    return pascal_boxes


def poly_2_mask(
    polygons: List[List[Tuple[float, float]]], shape: Tuple[int, int]
) -> np.ndarray:
    """
    Convert polygons to binary masks.

    Args:
        polygons (List[List[Tuple[float, float]]]): List of polygons, where each polygon is a list of (x, y) coordinates.
        shape (Tuple[int, int]): The shape of the image as (height, width).

    Returns:
        np.ndarray: Array of binary masks with the same shape as the input image.
    """
    # TODO This doesnt fill the array inplace.    out = np.zeros(shape + (len(polygons),), dtype=np.uint8)
    out = np.zeros(shape + (len(polygons),), dtype=np.uint8)
    for i, polygon in enumerate(polygons):
        im = Image.fromarray(out[..., i])
        d = ImageDraw.Draw(im)
        d.polygon(polygon, fill=1)
        out[..., i] = np.asarray(im)
    return out


def coco_frac_2_pascal_pixel(boxes: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert bounding boxes from COCO format (top-left x, top-left y, width, height)
    in fractional coordinates to Pascal VOC format (top-left x, top-left y, bottom-right x, bottom-right y)
    in pixel coordinates.

    Args:
        boxes (np.ndarray): Array of bounding boxes in COCO format.
        shape (Tuple[int, int]): The shape of the image as (height, width).

    Returns:
        np.ndarray: Array of bounding boxes in Pascal VOC format.
    """
    bbox = np.empty((0, 4), dtype=boxes.dtype)

    if boxes.size != 0:
        x = boxes[:, 0] * shape[1]
        y = boxes[:, 1] * shape[0]
        w = boxes[:, 2] * shape[1]
        h = boxes[:, 3] * shape[0]
        bbox = np.stack((x, y, w, h), axis=1)
    return coco_pixel_2_pascal_pixel(bbox, shape)


def pascal_frac_2_pascal_pixel(boxes: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert bounding boxes from Pascal VOC format (top-left x, top-left y, bottom-right x, bottom-right y)
    in fractional coordinates to Pascal VOC format in pixel coordinates.

    Args:
        boxes (np.ndarray): Array of bounding boxes in Pascal VOC format.
        shape (Tuple[int, int]): The shape of the image as (height, width).

    Returns:
        np.ndarray: Array of bounding boxes in Pascal VOC format.
    """

    bbox = np.empty((0, 4), dtype=boxes.dtype)
    if boxes.size != 0:
        x_top = boxes[:, 0] * shape[1]
        y_top = boxes[:, 1] * shape[0]
        x_bottom = boxes[:, 2] * shape[1]
        y_bottom = boxes[:, 3] * shape[0]
        bbox = np.stack((x_top, y_top, x_bottom, y_bottom), axis=1)
    return bbox


def yolo_pixel_2_pascal_pixel(boxes: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert bounding boxes from YOLO format (center x, center y, width, height)
    in pixel coordinates to Pascal VOC format (top-left x, top-left y, bottom-right x, bottom-right y)
    in pixel coordinates.

    Args:
        boxes (np.ndarray): Array of bounding boxes in YOLO format.
        shape (Tuple[int, int]): The shape of the image as (height, width).

    Returns:
        np.ndarray: Array of bounding boxes in Pascal VOC format.
    """
    bbox = np.empty((0, 4), dtype=boxes.dtype)

    if boxes.size != 0:
        x_top = np.array(boxes[:, 0]) - np.floor(np.array(boxes[:, 2]) / 2)
        y_top = np.array(boxes[:, 1]) - np.floor(np.array(boxes[:, 3]) / 2)
        x_bottom = np.array(boxes[:, 0]) + np.floor(np.array(boxes[:, 2]) / 2)
        y_bottom = np.array(boxes[:, 1]) + np.floor(np.array(boxes[:, 3]) / 2)
        bbox = np.stack((x_top, y_top, x_bottom, y_bottom), axis=1)
    return bbox


def yolo_frac_2_pascal_pixel(boxes: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert bounding boxes from YOLO format (center x, center y, width, height)
    in fractional coordinates to Pascal VOC format (top-left x, top-left y, bottom-right x, bottom-right y)
    in pixel coordinates.

    Args:
        boxes (np.ndarray): Array of bounding boxes in YOLO format.
        shape (Tuple[int, int]): The shape of the image as (height, width).

    Returns:
        np.ndarray: Array of bounding boxes in Pascal VOC format.
    """
    bbox = np.empty((0, 4), dtype=boxes.dtype)

    if boxes.size != 0:
        x_center = boxes[:, 0] * shape[1]
        y_center = boxes[:, 1] * shape[0]
        width = boxes[:, 2] * shape[1]
        height = boxes[:, 3] * shape[0]
        bbox = np.stack((x_center, y_center, width, height), axis=1)
    return yolo_pixel_2_pascal_pixel(bbox, shape)


def get_bbox_format(bbox: List[float], bbox_info: Dict[str, Any]) -> Tuple[str, str]:
    """
    Determines the bounding box format based on the given bounding box and information.

    Args:
        bbox (List[float]): The bounding box coordinates.
        bbox_info (Dict[str, Any]): The bounding box information dictionary.

    Returns:
        Tuple[str, str]: A tuple containing the mode and type of the bounding box format.
    """
    bbox_info = bbox_info.get("coords", {})
    mode = bbox_info.get("mode", "LTWH")
    type = bbox_info.get("type", "pixel")

    if len(bbox_info) == 0 and np.mean(bbox) < 1:
        mode = "CCWH"
        type = "fractional"
    return (mode, type)
