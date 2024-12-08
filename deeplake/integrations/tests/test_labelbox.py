import labelbox as lb
import os
import numpy as np

from deeplake.integrations.labelbox import (
    create_dataset_from_video_annotation_project,
    converter_for_video_project_with_id,
)


def validate_ds(ds):
    assert set(ds.tensors) == set(
        {
            "bbox/bbox",
            "bbox/fully_visible",
            "checklist",
            "frame_idx",
            "frames",
            "line",
            "mask/mask",
            "mask/mask_label",
            "mask/mask_labels",
            "metadata/current_frame_name",
            "metadata/data_row_id",
            "metadata/dataset_id",
            "metadata/dataset_name",
            "metadata/frame_count",
            "metadata/frame_number",
            "metadata/frame_rate",
            "metadata/global_key",
            "metadata/height",
            "metadata/label_creator",
            "metadata/name",
            "metadata/ontology_id",
            "metadata/project_name",
            "metadata/width",
            "point",
            "radio_bttn",
            "radio_bttn_scale",
            "text",
            "video_idx",
        }
    )

    assert np.all(
        ds["bbox/bbox"][0:3].numpy()
        == [[[1096, 9, 362, 369]], [[1096, 8, 362, 368]], [[1097, 8, 362, 368]]]
    )
    assert np.all(ds["bbox/fully_visible"][0:3].numpy() == [[0], [0], [0]])

    assert np.all(ds["bbox/bbox"][499].numpy() == [[1455, 0, 305, 78]])
    assert len(ds["bbox/bbox"]) == 500

    assert np.all(ds["bbox/fully_visible"][499].numpy() == [[1]])
    assert len(ds["bbox/fully_visible"]) == 500

    assert np.all(ds["checklist"][498:501].numpy() == [[], [], []])
    assert np.all(ds["checklist"][634].numpy() == [[]])
    assert np.all(ds["checklist"][635].numpy() == [[]])
    assert np.all(ds["checklist"][636].numpy() == [[0]])

    assert np.all(ds["checklist"][668].numpy() == [[0]])
    assert np.all(ds["checklist"][669].numpy() == [[1, 0]])

    assert np.all(
        ds["frame_idx"][245:255].numpy()
        == [[245], [246], [247], [248], [249], [250], [251], [252], [253], [254]]
    )

    assert np.all(
        ds["frame_idx"][495:505].numpy()
        == [[495], [496], [497], [498], [499], [0], [1], [2], [3], [4]]
    )

    assert np.all(ds["line"][245:255].numpy() == [])

    assert np.all(ds["mask/mask_label"][500].numpy() == [0, 1])

    assert np.all(ds["mask/mask_labels"][500].numpy() == [0])

    assert np.all(
        ds["metadata/current_frame_name"][245:255].numpy()
        == [
            ["output005_000245"],
            ["output005_000246"],
            ["output005_000247"],
            ["output005_000248"],
            ["output005_000249"],
            ["output005_000250"],
            ["output005_000251"],
            ["output005_000252"],
            ["output005_000253"],
            ["output005_000254"],
        ]
    )

    assert np.all(
        ds["metadata/current_frame_name"][495:505].numpy()
        == [
            ["output005_000495"],
            ["output005_000496"],
            ["output005_000497"],
            ["output005_000498"],
            ["output005_000499"],
            ["output006_000000"],
            ["output006_000001"],
            ["output006_000002"],
            ["output006_000003"],
            ["output006_000004"],
        ]
    )

    assert np.all(
        ds["video_idx"][245:255].numpy()
        == [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
    )

    assert np.all(
        ds["video_idx"][495:505].numpy()
        == [[0], [0], [0], [0], [0], [1], [1], [1], [1], [1]]
    )

    assert len(ds["point"]) == 626
    assert np.all(ds["point"][0].numpy() == [[]])
    assert np.all(ds["point"][499].numpy() == [[]])
    assert np.all(ds["point"][500].numpy() == [[1612, 76]])
    assert np.all(ds["point"][501].numpy() == [[1613, 75]])
    assert np.all(ds["point"][625].numpy() == [[1662, 0]])

    print("dataset is valid!")


import pytest


@pytest.mark.skip(reason="need to setup the environment variables")
def test_connect_to_labelbox():
    # the path where we want to create the dataset
    ds_path = "mem://labelbox_connect_test"

    API_KEY = os.environ["LABELBOX_TOKEN"]
    client = lb.Client(api_key=API_KEY)

    project_id = "cm4d6k0g001kl080fgluka1eu"

    # we pass the url presigner in cases when the videos are in cloud storage (
    # for this case azure blob storage) and the videos were added to labelbox with their integrations functionality.
    # the default one tries to use labelbox api to get the non public remote urls.
    def url_presigner(url):
        sas_token = os.environ["AZURE_SAS_TOKEN"]
        # the second value is the headers that will be added to the request
        return url.partition("?")[0] + "?" + sas_token, {}

    ds = create_dataset_from_video_annotation_project(
        ds_path,
        project_id,
        client,
        API_KEY,
        deeplake_token=os.environ["MY_ACTIVELOOP_PROD_TOKEN"],
        overwrite=True,
        url_presigner=url_presigner,
    )

    def ds_provider(p):
        # we need to have a clean branch to apply the annotations
        try:
            ds.delete_branch("labelbox")
        except:
            pass
        ds.checkout("labelbox", create=True)
        return ds

    converter = converter_for_video_project_with_id(
        project_id,
        client,
        ds_provider,
        API_KEY,
        group_mapping={"raster-segmentation": "mask"},
    )
    print("generating annotations")
    ds = converter.dataset_with_applied_annotations()

    # commit the annotations to the dataset
    ds.commit("add labelbox annotations")

    validate_ds(ds)
