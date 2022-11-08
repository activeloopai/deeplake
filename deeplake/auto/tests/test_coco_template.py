import deeplake

from ..unstructured.util import DatasetStructure, TensorStructure, GroupStructure


def test_dataset_structure(local_ds):
    dataset_structure = DatasetStructure(ignore_one_group=False)

    dataset_structure.add_first_level_tensor(
        TensorStructure("tensor1", params={"htype": "generic"}, primary=False)
    )
    dataset_structure.add_first_level_tensor(
        TensorStructure(
            "images",
            params={"htype": "image", "sample_compression": "jpeg"},
            primary=True,
        )
    )

    group = GroupStructure(
        "annotations", items=[TensorStructure("bboxes", params={"htype": "bbox"})]
    )
    group.add_item(TensorStructure("keypoints", params={"htype": "keypoints_coco"}))
    group.add_item(TensorStructure("masks", params={"htype": "binary_mask"}))

    sub_group = GroupStructure("sub_annotations")
    sub_group.add_item(TensorStructure("sub_tensor1", params={"htype": "generic"}))
    sub_group.add_item(TensorStructure("sub_tensor2", params={"htype": "generic"}))
    group.add_item(sub_group)

    dataset_structure.add_group(group)

    dataset_structure.create_structure(local_ds)

    tensors = local_ds.tensors
    assert len(tensors) == 7
    assert "tensor1" in tensors
    assert "annotations/sub_annotations/sub_tensor1" in tensors


def test_minimal_coco_ingestion(local_path):
    key_to_tensor = {"segmentation": "mask", "bbox": "bboxes"}
    file_to_group = {"instances_val2017": "base_annotations"}
    ignore_keys = ["area", "iscrowd"]

    ds = deeplake.ingest_coco(
        src="../ingestion_templates/datasets/coco/val2017_small",
        dest=local_path,
        annotation_files=[
            "../ingestion_templates/datasets/coco/annotations/instances_val2017.json"
        ],
        key_to_tensor_mapping=key_to_tensor,
        file_to_group_mapping=file_to_group,
        ignore_keys=ignore_keys,
        ignore_one_group=False,
    )

    assert ds.path == local_path
    assert len(ds.groups) == 1
    assert "images" in ds.tensors
    assert "base_annotations/mask" in ds.tensors
    assert "base_annotations/bboxes" in ds.tensors

    assert "base_annotations/iscrowd" not in ds.tensors
    assert "base_annotations/area" not in ds.tensors
