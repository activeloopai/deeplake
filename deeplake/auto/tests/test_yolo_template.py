import deeplake
import pytest
from deeplake.util.exceptions import IngestionError
import numpy as np
from click.testing import CliRunner


def create_yolo_export_dataset_basic():
    ds = deeplake.empty("mem://dummy")

    image_shape = (100, 100, 3)

    with ds:
        ds.create_tensor("images", htype="image", sample_compression="png")
        ds.create_tensor("boxes_ltwh", htype="bbox")
        ds.create_tensor(
            "labels", htype="class_label", class_names=["class_1", "class_2", "class_3"]
        )

        # Create numpy image array with random data
        ds.extend(
            {
                "images": [
                    np.random.randint(0, 255, image_shape, dtype=np.uint8),
                    np.random.randint(0, 255, image_shape, dtype=np.uint8),
                    np.random.randint(0, 255, image_shape, dtype=np.uint8),
                ],
                "boxes_ltwh": [
                    np.array([[0, 0, 50, 50], [25, 25, 75, 75]], dtype=np.float32),
                    np.array([[10, 10, 20, 20]], dtype=np.float32),
                    None,
                ],
                "labels": [np.array([0, 1]), np.array([0]), None],
            }
        )

    return ds


def create_yolo_export_dataset_complex():

    ds = create_yolo_export_dataset_basic()

    with ds:
        ds.create_tensor(
            "boxes_ccwh", htype="bbox", coords={"type": "pixel", "mode": "ccwh"}
        )
        ds.boxes_ccwh.extend(
            [
                np.array([[25, 25, 50, 50], [50, 50, 75, 75]], dtype=np.float32),
                np.array([[50, 50, 20, 20]], dtype=np.float32),
                None,
            ],
        )
    return ds


@pytest.mark.parametrize("shuffle", [True, False])
def test_minimal_yolo_ingestion(local_path, yolo_ingestion_data, shuffle):
    params = {
        "data_directory": yolo_ingestion_data["data_directory"],
        "class_names_file": yolo_ingestion_data["class_names_file"],
    }

    ds = deeplake.ingest_yolo(**params, shuffle=shuffle, dest=local_path)

    assert ds.path == local_path
    assert "images" in ds.tensors
    assert "boxes" in ds.tensors
    assert "labels" in ds.tensors
    assert len(ds.labels.info["class_names"]) > 0
    assert ds.boxes.htype == "bbox"


def test_minimal_yolo_ingestion_no_class_names(local_path, yolo_ingestion_data):
    params = {
        "data_directory": yolo_ingestion_data["data_directory"],
        "class_names_file": None,
    }

    ds = deeplake.ingest_yolo(**params, dest=local_path)

    assert ds.path == local_path
    assert "images" in ds.tensors
    assert "boxes" in ds.tensors
    assert "labels" in ds.tensors
    assert ds.labels.info["class_names"] == []
    assert ds.boxes.htype == "bbox"


def test_minimal_yolo_ingestion_separate_annotations(local_path, yolo_ingestion_data):
    params = {
        "data_directory": yolo_ingestion_data["data_directory_no_annotations"],
        "class_names_file": yolo_ingestion_data["class_names_file"],
        "annotations_directory": yolo_ingestion_data["annotations_directory"],
    }

    ds = deeplake.ingest_yolo(**params, dest=local_path)

    assert ds.path == local_path
    assert "images" in ds.tensors
    assert "boxes" in ds.tensors
    assert "labels" in ds.tensors
    assert len(ds.labels.info["class_names"]) > 0
    assert ds.boxes.htype == "bbox"


def test_minimal_yolo_ingestion_missing_annotations(local_path, yolo_ingestion_data):
    params = {
        "data_directory": yolo_ingestion_data["data_directory_missing_annotations"],
        "class_names_file": yolo_ingestion_data["class_names_file"],
        "allow_no_annotation": True,
    }

    ds = deeplake.ingest_yolo(**params, dest=local_path)

    assert ds.path == local_path
    assert "images" in ds.tensors
    assert "boxes" in ds.tensors
    assert "labels" in ds.tensors
    assert len(ds.labels.info["class_names"]) > 0
    assert ds.boxes.htype == "bbox"


def test_minimal_yolo_ingestion_unsupported_annotations(
    local_path, yolo_ingestion_data
):
    params = {
        "data_directory": yolo_ingestion_data["data_directory_unsupported_annotations"],
        "class_names_file": yolo_ingestion_data["class_names_file"],
    }

    with pytest.raises(IngestionError):
        ds = deeplake.ingest_yolo(**params, dest=local_path)


def test_minimal_yolo_ingestion_bad_data_path(local_path, yolo_ingestion_data):
    params = {
        "data_directory": yolo_ingestion_data["data_directory"] + "corrupt_this_path",
        "class_names_file": yolo_ingestion_data["class_names_file"],
    }

    with pytest.raises(IngestionError):
        ds = deeplake.ingest_yolo(**params, dest=local_path)


def test_minimal_yolo_ingestion_poly(local_path, yolo_ingestion_data):
    params = {
        "data_directory": yolo_ingestion_data["data_directory"],
        "class_names_file": yolo_ingestion_data["class_names_file"],
    }

    ds = deeplake.ingest_yolo(
        **params,
        dest=local_path,
        coordinates_params={"name": "polygons", "htype": "polygon"},
    )

    assert ds.path == local_path
    assert "images" in ds.tensors
    assert "polygons" in ds.tensors
    assert "labels" in ds.tensors
    assert len(ds.labels.info["class_names"]) > 0
    assert ds.polygons.htype == "polygon"


def test_minimal_yolo_with_connect(
    s3_path,
    yolo_ingestion_data,
    hub_cloud_path,
    hub_cloud_dev_token,
    hub_cloud_dev_managed_creds_key,
):
    params = {
        "data_directory": yolo_ingestion_data["data_directory"],
        "class_names_file": yolo_ingestion_data["class_names_file"],
    }

    ds = deeplake.ingest_yolo(
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
    assert "boxes" in ds.tensors
    assert "labels" in ds.tensors
    assert len(ds.labels.info["class_names"]) > 0
    assert ds.boxes.htype == "bbox"


def test_minimal_yolo_ingestion_with_linked_images(
    s3_path,
    yolo_ingestion_data,
    hub_cloud_path,
    hub_cloud_dev_token,
    hub_cloud_dev_managed_creds_key,
):
    params = {
        "data_directory": yolo_ingestion_data["data_directory"],
        "class_names_file": yolo_ingestion_data["class_names_file"],
    }

    ds = deeplake.ingest_yolo(
        **params,
        dest=s3_path,
        image_params={
            "name": "linked_images",
            "htype": "link[image]",
            "sample_compression": "png",
        },
        image_creds_key=hub_cloud_dev_managed_creds_key,
        connect_kwargs={
            "dest_path": hub_cloud_path,
            "creds_key": hub_cloud_dev_managed_creds_key,
            "token": hub_cloud_dev_token,
        },
    )

    assert ds.path == hub_cloud_path
    assert "linked_images" in ds.tensors
    assert "boxes" in ds.tensors
    assert "labels" in ds.tensors
    assert len(ds.labels.info["class_names"]) > 0
    assert ds.linked_images.htype == "link[image]"


def text_export_yolo_basic():
    """Basic test for export_yolo function to see if it runs without errors"""

    ds = create_yolo_export_dataset_basic()

    with CliRunner().isolated_filesystem():
        deeplake.export_yolo(ds, "/basic")


def text_export_yolo_edge_cases():

    ds = create_yolo_export_dataset_complex()

    with CliRunner().isolated_filesystem():
        deeplake.export_yolo(
            ds,
            "/custom_boxes",
            box_tensor="boxes_ccwh",
            label_tensor="labels",
            image_tensor="images",
            limit=1,
        )

    # Check for error about correct tensors not being found
    ds_empty = deeplake.empty("mem://dummy")
    with pytest.raises(ValueError):
        with CliRunner().isolated_filesystem():
            deeplake.export_yolo(ds_empty, "/no_tensors")

    # Check for error about class names not being present
    ds_empty = deeplake.empty("mem://dummy")
    with pytest.raises(ValueError):
        with CliRunner().isolated_filesystem():
            deeplake.export_yolo(ds_empty, "/no_class_names")
