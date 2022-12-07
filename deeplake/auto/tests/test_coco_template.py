import os
import deeplake
import pytest

from deeplake.auto.unstructured.util import (
    DatasetStructure,
    TensorStructure,
    GroupStructure,
)
from deeplake.tests.common import get_dummy_data_path


@pytest.fixture
def coco_images_path():
    return get_dummy_data_path(os.path.join("tests_auto", "coco", "images"))


@pytest.fixture
def coco_annotation_path():
    return get_dummy_data_path(os.path.join("tests_auto", "coco", "annotations1.json"))


@pytest.fixture
def coco_annotation_path2():
    return get_dummy_data_path(os.path.join("tests_auto", "coco", "annotations2.json"))


def test_full_dataset_structure(local_ds):
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

    dataset_structure.create_full(local_ds)

    tensors = local_ds.tensors
    assert len(tensors) == 7
    assert "tensor1" in tensors
    assert "annotations/keypoints" in tensors
    assert "annotations/masks" in tensors
    assert "annotations/sub_annotations/sub_tensor1" in tensors


def test_missing_dataset_structure(local_ds):
    dataset_structure = DatasetStructure(ignore_one_group=False)

    local_ds.create_tensor("images", htype="image", sample_compression="jpeg")
    local_ds.create_tensor("annotations/masks", htype="binary_mask")

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

    dataset_structure.create_missing(local_ds)

    tensors = local_ds.tensors
    assert len(tensors) == 7
    assert "tensor1" in tensors
    assert "annotations/keypoints" in tensors
    assert "annotations/masks" in tensors
    assert "annotations/sub_annotations/sub_tensor1" in tensors


def test_minimal_coco_ingestion(
    local_path, coco_images_path, coco_annotation_path, coco_annotation_path2
):
    key_to_tensor = {"segmentation": "mask", "bbox": "bboxes"}
    file_to_group = {"annotations1": "group1", "annotations2": "group2"}
    ignore_keys = ["area", "iscrowd"]

    ds = deeplake.ingest_coco(
        images_directory=coco_images_path,
        dest=local_path,
        annotation_files=[coco_annotation_path, coco_annotation_path2],
        key_to_tensor_mapping=key_to_tensor,
        file_to_group_mapping=file_to_group,
        ignore_keys=ignore_keys,
        ignore_one_group=False,
    )

    assert ds.path == local_path
    assert "images" in ds.tensors
    assert "group1/mask" in ds.tensors
    assert "group2/mask" in ds.tensors
    assert "group1/bboxes" in ds.tensors
    assert "group2/bboxes" in ds.tensors
    assert "group1/iscrowd" not in ds.tensors
    assert "group2/iscrowd" not in ds.tensors


def test_ingestion_with_linked_images(
    local_path, coco_images_path, coco_annotation_path
):
    file_to_group = {"annotations1.json": "base_annotations"}
    ds = deeplake.ingest_coco(
        images_directory=coco_images_path,
        annotation_files=coco_annotation_path,
        file_to_group_mapping=file_to_group,
        dest=local_path,
        image_settings={"name": "linked_images", "linked": True},
    )

    assert ds.path == local_path
    assert "linked_images" in ds.tensors
    assert "base_annotations/bbox" in ds.tensors
    assert "base_annotations/segmentation" in ds.tensors
    assert ds.linked_images.htype == "link[image]"


def test_ingestion_with_existing_destination(
    local_ds, coco_images_path, coco_annotation_path
):
    local_ds.create_tensor(
        "linked_images", htype="link[image]", sample_compression="jpeg"
    )
    local_ds.create_tensor(
        "base_annotations/bbox", htype="bbox", sample_compression=None
    )

    file_to_group = {"annotations1.json": "base_annotations"}
    deeplake.ingest_coco(
        images_directory=coco_images_path,
        annotation_files=coco_annotation_path,
        file_to_group_mapping=file_to_group,
        dest=local_ds,
        image_settings={"name": "linked_images"},
    )

    assert "linked_images" in local_ds.tensors
    assert "base_annotations/bbox" in local_ds.tensors
    assert "base_annotations/segmentation" in local_ds.tensors
    assert local_ds.linked_images.htype == "link[image]"
