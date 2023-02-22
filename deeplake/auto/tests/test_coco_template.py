import pytest
import pathlib

import deeplake

from deeplake.auto.unstructured.util import (
    DatasetStructure,
    TensorStructure,
    GroupStructure,
)
from deeplake.util.exceptions import IngestionError


def test_full_dataset_structure(local_ds):
    dataset_structure = DatasetStructure(ignore_one_group=False)

    dataset_structure.add_first_level_tensor(
        TensorStructure("tensor1", params={"htype": "generic"})
    )
    dataset_structure.add_first_level_tensor(
        TensorStructure(
            "images",
            params={"htype": "image", "sample_compression": "jpeg"},
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

    assert dataset_structure.all_keys == {
        "images",
        "tensor1",
        "annotations/bboxes",
        "annotations/keypoints",
        "annotations/masks",
        "annotations/sub_annotations/sub_tensor1",
        "annotations/sub_annotations/sub_tensor2",
    }

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
        TensorStructure("tensor1", params={"htype": "generic"})
    )
    dataset_structure.add_first_level_tensor(
        TensorStructure(
            "images",
            params={"htype": "image", "sample_compression": "jpeg"},
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


@pytest.mark.parametrize("shuffle", [True, False])
def test_minimal_coco_ingestion(local_path, coco_ingestion_data, shuffle):
    key_to_tensor = {"segmentation": "mask", "bbox": "bboxes"}
    file_to_group = {"annotations1": "group1", "annotations2": "group2"}
    ignore_keys = ["area", "iscrowd"]

    ds = deeplake.ingest_coco(
        **coco_ingestion_data,
        dest=local_path,
        key_to_tensor_mapping=key_to_tensor,
        file_to_group_mapping=file_to_group,
        ignore_keys=ignore_keys,
        ignore_one_group=False,
        shuffle=shuffle,
    )

    assert ds.path == local_path
    assert "images" in ds.tensors
    assert "group1/category_id" in ds.tensors
    assert "group2/category_id" in ds.tensors
    assert "group1/mask" in ds.tensors
    assert "group2/mask" in ds.tensors
    assert "group1/bboxes" in ds.tensors
    assert "group2/bboxes" in ds.tensors
    assert "group1/iscrowd" not in ds.tensors
    assert "group2/iscrowd" not in ds.tensors


def test_minimal_coco_with_connect(
    s3_path,
    coco_ingestion_data,
    hub_cloud_path,
    hub_cloud_dev_token,
    hub_cloud_dev_managed_creds_key,
):
    params = {**coco_ingestion_data}

    ds = deeplake.ingest_coco(
        **params,
        dest=s3_path,
        connect_kwargs={
            "dest_path": hub_cloud_path,
            "creds_key": hub_cloud_dev_managed_creds_key,
            "token": hub_cloud_dev_token,
        },
    )

    assert ds.path == hub_cloud_path
    assert "images" in ds.tensors
    assert "annotations1/bbox" in ds.tensors


def test_coco_ingestion_with_linked_images(
    s3_path,
    coco_ingestion_data,
    hub_cloud_path,
    hub_cloud_dev_token,
    hub_cloud_dev_managed_creds_key,
):
    file_to_group = {"annotations1.json": "base_annotations"}
    ds = deeplake.ingest_coco(
        **coco_ingestion_data,
        file_to_group_mapping=file_to_group,
        dest=s3_path,
        image_params={"name": "linked_images", "htype": "link[image]"},
        image_creds_key=hub_cloud_dev_managed_creds_key,
        connect_kwargs={
            "dest_path": hub_cloud_path,
            "creds_key": hub_cloud_dev_managed_creds_key,
            "token": hub_cloud_dev_token,
        },
    )

    assert ds.path == hub_cloud_path
    assert "linked_images" in ds.tensors
    assert ds.linked_images.num_samples > 0
    assert ds.linked_images.htype == "link[image]"
    assert "base_annotations/bbox" in ds.tensors
    assert "base_annotations/segmentation" in ds.tensors


def test_flat_coco_ingestion(local_path, coco_ingestion_data):
    params = {**coco_ingestion_data}
    params["annotation_files"] = params["annotation_files"][0]
    ds = deeplake.ingest_coco(**params, dest=local_path, ignore_one_group=True)

    assert ds.path == local_path
    assert len(ds.groups) == 0
    assert "images" in ds.tensors
    assert "bbox" in ds.tensors
    assert "segmentation" in ds.tensors


def test_coco_ingestion_with_invalid_mapping(local_path, coco_ingestion_data):
    non_unique_file_to_group = {
        "annotations1.json": "annotations",
        "annotations2.json": "annotations",
    }

    non_unique_key_to_tensor = {
        "segmentation": "mask",
        "bbox": "mask",
    }

    with pytest.raises(IngestionError):
        deeplake.ingest_coco(
            **coco_ingestion_data,
            file_to_group_mapping=non_unique_file_to_group,
            dest=local_path,
        )

    with pytest.raises(IngestionError):
        deeplake.ingest_coco(
            **coco_ingestion_data,
            key_to_tensor_mapping=non_unique_key_to_tensor,
            dest=local_path,
        )


def test_coco_ingestion_with_incomplete_data(local_path, coco_ingestion_data):
    only_images = {
        "images_directory": coco_ingestion_data["images_directory"],
        "annotation_files": [],
    }

    no_images = {
        # There are no supported images in the annotations directory
        "images_directory": pathlib.Path(
            coco_ingestion_data["annotation_files"][0]
        ).parent,
        "annotation_files": coco_ingestion_data["annotation_files"],
    }

    invalid_annotation_file_path = {
        "images_directory": coco_ingestion_data["images_directory"],
        "annotation_files": ["invalid_path"],
    }

    with pytest.raises(IngestionError):
        deeplake.ingest_coco(
            **no_images,
            dest=local_path,
        )

    with pytest.raises(IngestionError):
        deeplake.ingest_coco(
            **invalid_annotation_file_path,
            dest=local_path,
        )

    ds = deeplake.ingest_coco(
        **only_images,
        dest=local_path,
    )

    assert ds.path == local_path
    assert "images" in ds.tensors
    assert len(ds.tensors) == 1
    assert ds.images.num_samples == 10
