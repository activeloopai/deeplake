import labelbox as lb
import os
import tempfile
import pytest

from deeplake.integrations.labelbox import (
    create_dataset_from_video_annotation_project,
    converter_for_video_project_with_id,
)


@pytest.mark.skip(reason="Sometimes fails due to Labelbox authentication issues")
def test_labelbox():
    with tempfile.TemporaryDirectory() as temp_dir:
        ds_path = os.path.join(temp_dir, "labelbox_ds")
        API_KEY = os.environ["LABELBOX_TOKEN"]
        client = lb.Client(api_key=API_KEY)

        project_id = "cm3x920j0002m07xy5ittaqj6"
        ds = create_dataset_from_video_annotation_project(
            ds_path, project_id, client, API_KEY, overwrite=True
        )

        def ds_provider(p):
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
        ds = converter.dataset_with_applied_annotations()

        ds.commit("add labelbox annotations")

        assert set(ds.tensors) == set(
            {
                "bbox/bbox",
                "bbox/fully_visible",
                "checklist",
                "frame_idx",
                "frames",
                "line",
                "mask",
                "mask_labels",
                "point",
                "radio_bttn",
                "radio_bttn_scale",
                "text",
                "video_idx",
            }
        )
