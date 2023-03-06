import deeplake
import pytest
from deeplake.util.exceptions import IngestionError


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
