import os
import shutil

os.environ["BUGGER_OFF"] = "True"

import deeplake

token = "eyJhbGciOiJIUzUxMiIsImlhdCI6MTY1NTY2NTE0NiwiZXhwIjo0ODA5MjY1MTQ2fQ.eyJpZCI6InByb2dlcmRhdiJ9.QAaCeQumZzTTDodvm9L07eIzRSu1raKVeOjMnCniNHJujIsSJ5N5qeLglo8ZvucB7AuPn7YGK0x3_jaGumnYKw"

key_to_tensor = {"segmentation": "mask", "bbox": "bboxes"}

file_to_group = {"instances_val2017": "base_annotations"}

ignore_keys = [
    # "area",
    # "iscrowd",
    # "image_id",
    # "segmentation",
    "bbox",
    # "id",
    # "category_id",
]

# images_directory = "s3://activeloop-ds-creation-tests/coco_source/val2017_small/"
images_directory = "../ingestion_templates/datasets/coco/val2017"
# images_directory = "../ingestion_templates/datasets/coco/val2017_small"
annotation_files = [
    "../ingestion_templates/datasets/coco/annotations/instances_val2017.json"
    # "s3://activeloop-ds-creation-tests/coco_source/instances_val2017.json"
]


dest_path = "../ingested_coco"

try:
    shutil.rmtree(dest_path)
except:
    pass

deeplake.ingest_coco(
    images_directory=images_directory,
    annotation_files=annotation_files,
    dest=dest_path,
    ignore_keys=ignore_keys,
)
