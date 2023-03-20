import numpy as np
from PIL import Image, ImageDraw  # type: ignore


def coco_pixel_2_pascal_pixel(boxes, shape):
    # Convert bounding boxes to Pascal VOC format and clip bounding boxes to make sure they have non-negative width and height
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


def poly_2_mask(polygons, shape):
    # TODO This doesnt fill the array inplace.    out = np.zeros(shape + (len(polygons),), dtype=np.uint8)
    out = np.zeros(shape + (len(polygons),), dtype=np.uint8)
    for i, polygon in enumerate(polygons):
        im = Image.fromarray(out[..., i])
        d = ImageDraw.Draw(im)
        d.polygon(polygon, fill=1)
        out[..., i] = np.asarray(im)
    return out


def coco_frac_2_pascal_pixel(boxes, shape):
    bbox = np.empty((0, 4), dtype=boxes.dtype)

    if boxes.size != 0:
        x = boxes[:, 0] * shape[1]
        y = boxes[:, 1] * shape[0]
        w = boxes[:, 2] * shape[1]
        h = boxes[:, 3] * shape[0]
        bbox = np.stack((x, y, w, h), axis=1)
    return coco_pixel_2_pascal_pixel(bbox, shape)


def pascal_frac_2_pascal_pixel(boxes, shape):
    bbox = np.empty((0, 4), dtype=boxes.dtype)
    if boxes.size != 0:
        x_top = boxes[:, 0] * shape[1]
        y_top = boxes[:, 1] * shape[0]
        x_bottom = boxes[:, 2] * shape[1]
        y_bottom = boxes[:, 3] * shape[0]
        bbox = np.stack((x_top, y_top, x_bottom, y_bottom), axis=1)
    return bbox


def yolo_pixel_2_pascal_pixel(boxes, shape):
    bbox = np.empty((0, 4), dtype=boxes.dtype)

    if boxes.size != 0:
        x_top = np.array(boxes[:, 0]) - np.floor(np.array(boxes[:, 2]) / 2)
        y_top = np.array(boxes[:, 1]) - np.floor(np.array(boxes[:, 3]) / 2)
        x_bottom = np.array(boxes[:, 0]) + np.floor(np.array(boxes[:, 2]) / 2)
        y_bottom = np.array(boxes[:, 1]) + np.floor(np.array(boxes[:, 3]) / 2)
        bbox = np.stack((x_top, y_top, x_bottom, y_bottom), axis=1)
    return bbox


def yolo_frac_2_pascal_pixel(boxes, shape):
    bbox = np.empty((0, 4), dtype=boxes.dtype)

    if boxes.size != 0:
        x_center = boxes[:, 0] * shape[1]
        y_center = boxes[:, 1] * shape[0]
        width = boxes[:, 2] * shape[1]
        height = boxes[:, 3] * shape[0]
        bbox = np.stack((x_center, y_center, width, height), axis=1)
    return yolo_pixel_2_pascal_pixel(bbox, shape)


def get_bbox_format(bbox, bbox_info):
    bbox_info = bbox_info.get("coords", {})
    mode = bbox_info.get("mode", "LTWH")
    type = bbox_info.get("type", "pixel")

    if len(bbox_info) == 0 and np.mean(bbox) < 1:
        mode = "CCWH"
        type = "fractional"
    return (mode, type)

