import deeplake


def test_minimal_yolo_ingestion(local_path, yolo_ingestion_data):

    ds = deeplake.ingest_yolo(**yolo_ingestion_data, dest=local_path)

    assert ds.path == local_path
    assert "images" in ds.tensors
    assert "boxes" in ds.tensors
    assert "labels" in ds.tensors
    assert ds.boxes.htype == "bbox"


def test_minimal_yolo_ingestion_poly(local_path, yolo_ingestion_data):

    ds = deeplake.ingest_yolo(
        **yolo_ingestion_data,
        dest=local_path,
        coordinates_settings={"name": "polygons", "htype": "polygon"},
    )

    assert ds.path == local_path
    assert "images" in ds.tensors
    assert "polygons" in ds.tensors
    assert "labels" in ds.tensors
    assert ds.polygons.htype == "polygon"


def test_yolo_ingestion_with_linked_images(local_path, yolo_ingestion_data):

    ds = deeplake.ingest_yolo(
        **yolo_ingestion_data,
        dest=local_path,
        image_settings={"name": "linked_images", "linked": True},
    )

    assert ds.path == local_path
    assert "linked_images" in ds.tensors
    assert ds.linked_images.htype == "link[image]"
