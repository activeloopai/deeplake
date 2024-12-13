import labelbox as lb  # type: ignore
import os
import numpy as np
import pytest

from deeplake.integrations.labelbox import (
    create_dataset_from_video_annotation_project,
    converter_for_video_project_with_id,
    load_blob_file_paths_from_azure,
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
            "lb_meta/current_frame_name",
            "lb_meta/id",
            "lb_meta/url",
            "lb_meta/dataset_id",
            "lb_meta/dataset_name",
            "lb_meta/frame_count",
            "lb_meta/frame_number",
            "lb_meta/frame_rate",
            "lb_meta/global_key",
            "lb_meta/height",
            "lb_meta/label_creator",
            "lb_meta/name",
            "lb_meta/ontology_id",
            "lb_meta/project_name",
            "lb_meta/width",
            "point",
            "radio_bttn",
            "radio_bttn_scale",
            "text",
            "video_idx",
        }
    )

    assert ds.max_len == 876

    assert len(ds["radio_bttn"]) == 474
    assert np.all(ds["radio_bttn"][0].numpy() == [[0]])
    assert np.all(ds["radio_bttn"][20].numpy() == [[0]])
    assert np.all(ds["radio_bttn"][23].numpy() == [[1]])

    assert np.all(
        ds["bbox/bbox"][0:3].numpy()
        == [[[1092, 9, 361, 361]], [[1092, 8, 360, 361]], [[1093, 8, 361, 360]]]
    )
    assert np.all(ds["bbox/fully_visible"][0:3].numpy() == [[0], [0], [0]])

    assert np.all(ds["bbox/bbox"][499].numpy() == [[1463, 0, 287, 79]])
    assert len(ds["bbox/bbox"]) == 500

    assert np.all(ds["bbox/fully_visible"][499].numpy() == [[1]])
    assert len(ds["bbox/fully_visible"]) == 500

    assert np.all(ds["radio_bttn"][0].numpy() == [[0]])
    assert np.all(ds["radio_bttn"][0].numpy() == [[0]])

    assert np.all(ds["checklist"][499].numpy() == [[]])
    assert np.all(ds["checklist"][500].numpy() == [[0, 1]])
    assert np.all(ds["checklist"][598].numpy() == [[1, 0]])
    assert np.all(ds["checklist"][599].numpy() == [[0]])
    assert np.all(ds["checklist"][698].numpy() == [[0]])
    assert np.all(ds["checklist"][699].numpy() == [[1]])
    assert len(ds["checklist"]) == 739

    assert np.all(
        ds["frame_idx"][245:255].numpy()
        == [[245], [246], [247], [248], [249], [250], [251], [252], [253], [254]]
    )

    assert np.all(
        ds["frame_idx"][495:505].numpy()
        == [[495], [496], [497], [498], [499], [0], [1], [2], [3], [4]]
    )

    assert np.all(ds["line"][245:255].numpy() == [])

    assert np.all(ds["mask/mask_label"][500].numpy() == [1])
    assert np.all(ds["mask/mask_label"][739].numpy() == [0])

    assert np.all(ds["mask/mask_labels"][500].numpy() == [0, 1])
    assert np.all(ds["mask/mask_labels"][739].numpy() == [0])

    assert np.all(
        ds["lb_meta/current_frame_name"][245:255].numpy()
        == [
            ["output005_000246"],
            ["output005_000247"],
            ["output005_000248"],
            ["output005_000249"],
            ["output005_000250"],
            ["output005_000251"],
            ["output005_000252"],
            ["output005_000253"],
            ["output005_000254"],
            ["output005_000255"],
        ]
    )

    assert np.all(
        ds["lb_meta/current_frame_name"][495:505].numpy()
        == [
            ["output005_000496"],
            ["output005_000497"],
            ["output005_000498"],
            ["output005_000499"],
            ["output005_000500"],
            ["output004_000001"],
            ["output004_000002"],
            ["output004_000003"],
            ["output004_000004"],
            ["output004_000005"],
        ]
    )

    assert np.all(
        ds["lb_meta/global_key"][495:505].numpy()
        == [
            ['42e8ee3b-92dd-4205-987d-257f961227b4'],
            ['42e8ee3b-92dd-4205-987d-257f961227b4'],
            ['42e8ee3b-92dd-4205-987d-257f961227b4'],
            ['42e8ee3b-92dd-4205-987d-257f961227b4'],
            ['42e8ee3b-92dd-4205-987d-257f961227b4'],
            ['0fb82d86-0130-4b4f-bba4-5c4c3f250c93'],
            ['0fb82d86-0130-4b4f-bba4-5c4c3f250c93'],
            ['0fb82d86-0130-4b4f-bba4-5c4c3f250c93'],
            ['0fb82d86-0130-4b4f-bba4-5c4c3f250c93'],
            ['0fb82d86-0130-4b4f-bba4-5c4c3f250c93']
        ]
    )

    assert np.all(
        ds["lb_meta/url"][495:505].numpy()
        == [
            [
                "https://activeloopgen2.blob.core.windows.net/deeplake-tests/video_chunks/output005.mp4"
            ],
            [
                "https://activeloopgen2.blob.core.windows.net/deeplake-tests/video_chunks/output005.mp4"
            ],
            [
                "https://activeloopgen2.blob.core.windows.net/deeplake-tests/video_chunks/output005.mp4"
            ],
            [
                "https://activeloopgen2.blob.core.windows.net/deeplake-tests/video_chunks/output005.mp4"
            ],
            [
                "https://activeloopgen2.blob.core.windows.net/deeplake-tests/video_chunks/output005.mp4"
            ],
            [
                "https://activeloopgen2.blob.core.windows.net/deeplake-tests/video_chunks/output004.mp4"
            ],
            [
                "https://activeloopgen2.blob.core.windows.net/deeplake-tests/video_chunks/output004.mp4"
            ],
            [
                "https://activeloopgen2.blob.core.windows.net/deeplake-tests/video_chunks/output004.mp4"
            ],
            [
                "https://activeloopgen2.blob.core.windows.net/deeplake-tests/video_chunks/output004.mp4"
            ],
            [
                "https://activeloopgen2.blob.core.windows.net/deeplake-tests/video_chunks/output004.mp4"
            ],
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

    assert len(ds["point"]) == 857
    assert np.all(ds["point"][0].numpy() == [[]])
    assert np.all(ds["point"][499].numpy() == [[]])
    assert np.all(ds["point"][800].numpy() == [[1630, 49]])

    print("dataset is valid!")


def get_azure_sas_token():
    import datetime

    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import (
        BlobServiceClient,
        ContainerSasPermissions,
        generate_container_sas,
    )

    # Construct the blob endpoint from the account name
    account_url = "https://activeloopgen2.blob.core.windows.net"

    # Create a BlobServiceClient object using DefaultAzureCredential
    blob_service_client = BlobServiceClient(
        account_url, credential=DefaultAzureCredential()
    )
    # Get a user delegation key that's valid for 1 day
    delegation_key_start_time = datetime.datetime.now(datetime.timezone.utc)
    delegation_key_expiry_time = delegation_key_start_time + datetime.timedelta(days=1)

    user_delegation_key = blob_service_client.get_user_delegation_key(
        key_start_time=delegation_key_start_time,
        key_expiry_time=delegation_key_expiry_time,
    )

    start_time = datetime.datetime.now(datetime.timezone.utc)
    expiry_time = start_time + datetime.timedelta(days=1)

    sas_token = generate_container_sas(
        account_name="activeloopgen2",
        container_name="deeplake-tests",
        user_delegation_key=user_delegation_key,
        permission=ContainerSasPermissions(read=True, list=True),
        expiry=expiry_time,
        start=start_time,
    )

    return sas_token


@pytest.mark.skip(reason="labelbox api sometimes freezes")
def test_connect_to_labelbox():
    # the path where we want to create the dataset
    ds_path = "mem://labelbox_connect_test"

    API_KEY = os.environ["LABELBOX_TOKEN"]
    client = lb.Client(api_key=API_KEY)

    project_id = "cm4hts5gf0109072nbpl390xc"

    sas_token = get_azure_sas_token()

    # we pass the url presigner in cases when the videos are in cloud storage (
    # for this case azure blob storage) and the videos were added to labelbox with their integrations functionality.
    # the default one tries to use labelbox api to get the non public remote urls.
    def url_presigner(url):
        # the second value is the headers that will be added to the request
        return url.partition("?")[0] + "?" + sas_token, {}

    ds = create_dataset_from_video_annotation_project(
        ds_path,
        project_id,
        client,
        API_KEY,
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


@pytest.mark.skip(reason="somemtimes fails with timeout")
def test_labelbox_azure_utils():
    files = load_blob_file_paths_from_azure(
        "activeloopgen2",
        "deeplake-tests",
        "video_chunks",
        get_azure_sas_token(),
        lambda x: x.endswith(".mp4"),
    )
    assert set([os.path.basename(f.partition("?")[0]) for f in files]) == {
        "output004.mp4",
        "output005.mp4",
        "output006.mp4",
    }
