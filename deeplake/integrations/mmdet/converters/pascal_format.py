import coco_format_converters


BBOX_FORMAT_TO_COCO_CONVERTER = {
    ("LTWH", "pixel"): lambda x, y: x,
    ("LTWH", "fractional"): coco_format_converters.coco_frac_2_coco_pixel,
    ("LTRB", "pixel"): coco_format_converters.pascal_pixel_2_coco_pixel,
    ("LTRB", "fractional"): coco_format_converters.pascal_frac_2_coco_pixel,
    ("CCWH", "pixel"): coco_format_converters.yolo_pixel_2_coco_pixel,
    ("CCWH", "fractional"): coco_format_converters.yolo_frac_2_coco_pixel,
}


def convert(bbox, bbox_format, images):
    converter = BBOX_FORMAT_TO_COCO_CONVERTER[bbox_format]
    return converter(bbox, images)
