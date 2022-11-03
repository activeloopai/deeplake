import deeplake

src = "../ingestion_templates/datasets/coco/val2017_small"
annotation_files = (
    "../ingestion_templates/datasets/coco/annotations/instances_val2017.json"
)
dest = "mem://coco_ingested"

key_to_tensor_mapping = {
    "segmentation": "masks",
    "bbox": "boxes",
    "category_id": "categories",
}
file_to_group_mapping = {"instances_val2017.json": "base_annotations"}
ignore_one_group = False

ds = deeplake.ingest_coco(
    src=src,
    annotation_files=annotation_files,
    dest=dest,
    key_to_tensor_mapping=key_to_tensor_mapping,
    file_to_group_mapping=file_to_group_mapping,
    ignore_one_group=ignore_one_group,
    ignore_keys=["iscrowd", "area"],
)

print(ds.tensors)
