import numpy as np
from typing import List


def pascal_pixel_2_coco_pixel(
    boxes: List[np.ndarray], images: List[np.ndarray]
) -> List[np.ndarray]:
    """
    Converts bounding box coordinates in Pascal VOC format from pixel coordinates to COCO format.

    Args:
        boxes (List[np.ndarray]): A list of Numpy arrays containing bounding box coordinates.
        images (List[np.ndarray]): A list of Numpy arrays containing images.

    Returns:
        List[np.ndarray]: A list of Numpy arrays containing bounding box coordinates in COCO format.
    """
    pascal_boxes = []
    for box in boxes:
        if box.size != 0:
            pascal_boxes.append(
                np.stack(
                    (
                        box[:, 0],
                        box[:, 1],
                        box[:, 2] - box[:, 0],
                        box[:, 3] - box[:, 1],
                    ),
                    axis=1,
                )
            )
        else:
            pascal_boxes.append(box)
    return pascal_boxes


def pascal_frac_2_coco_pixel(
    boxes: List[np.ndarray], images: List[np.ndarray]
) -> List[np.ndarray]:
    """
    Converts bounding box coordinates in Pascal VOC format from fraction coordinates to COCO format.

    Args:
        boxes (List[np.ndarray]): A list of Numpy arrays containing bounding box coordinates.
        images (List[np.ndarray]): A list of Numpy arrays containing images.

    Returns:
        List[np.ndarray]: A list of Numpy arrays containing bounding box coordinates in COCO format.
    """
    pascal_pixel_boxes = []
    for i, box in enumerate(boxes):
        if box.size != 0:
            shape = images[i].shape
            x_top = box[:, 0] * shape[1]
            y_top = box[:, 1] * shape[0]
            x_bottom = box[:, 2] * shape[1]
            y_bottom = box[:, 3] * shape[0]
            bbox = np.stack((x_top, y_top, x_bottom, y_bottom), axis=1)
        pascal_pixel_boxes.append(bbox)
    return pascal_pixel_2_coco_pixel(pascal_pixel_boxes, images)


def yolo_pixel_2_coco_pixel(
    boxes: List[np.ndarray], images: List[np.ndarray]
) -> List[np.ndarray]:
    """
    Converts bounding box coordinates in YOLO format from pixel coordinates to COCO format.

    Args:
        boxes (List[np.ndarray]): A list of Numpy arrays containing bounding box coordinates.
        images (List[np.ndarray]): A list of Numpy arrays containing images.

    Returns:
        List[np.ndarray]: A list of Numpy arrays containing bounding box coordinates in COCO format.
    """
    yolo_boxes = []
    for box in boxes:
        if box.size != 0:
            x_top = np.array(box[:, 0]) - np.floor(np.array(box[:, 2]) / 2)
            y_top = np.array(box[:, 1]) - np.floor(np.array(box[:, 3]) / 2)
            w = box[:, 2]
            h = box[:, 3]
            bbox = np.stack([x_top, y_top, w, h], axis=1)
        yolo_boxes.append(bbox)
    return yolo_boxes


def yolo_frac_2_coco_pixel(
    boxes: List[np.ndarray], images: List[np.ndarray]
) -> List[np.ndarray]:
    """
    Converts bounding box coordinates in YOLO format from fraction coordinates to COCO format.

    Args:
        boxes (List[np.ndarray]): A list of Numpy arrays containing bounding box coordinates.
        images (List[np.ndarray]): A list of Numpy arrays containing images.

    Returns:
        List[np.ndarray]: A list of Numpy arrays containing bounding box coordinates in COCO format.
    """
    yolo_boxes = []
    for i, box in enumerate(boxes):
        shape = images[i].shape
        x_center = box[:, 0] * shape[1]
        y_center = box[:, 1] * shape[0]
        width = box[:, 2] * shape[1]
        height = box[:, 3] * shape[0]
        bbox = np.stack((x_center, y_center, width, height), axis=1)
        yolo_boxes.append(bbox)
    return yolo_pixel_2_coco_pixel(yolo_boxes, images)


def coco_frac_2_coco_pixel(
    boxes: List[np.ndarray], images: List[np.ndarray]
) -> np.ndarray:
    """
    Converts bounding box coordinates in COCO format from fraction coordinates to pixel coordinates.

    Args:
        boxes (List[np.ndarray]): A list of Numpy arrays containing bounding box coordinates.
        images (List[np.ndarray]): A list of Numpy arrays containing images.

    Returns:
        np.ndarray: A Numpy array containing bounding box coordinates in pixel coordinates.
    """
    coco_pixel_boxes = []
    for i, box in enumerate(boxes):
        shape = images[i].shape
        x = box[:, 0] * shape[1]
        y = box[:, 1] * shape[0]
        w = box[:, 2] * shape[1]
        h = box[:, 3] * shape[0]
        bbox = np.stack((x, y, w, h), axis=1)
        coco_pixel_boxes.append(bbox)
    return np.array(coco_pixel_boxes)
