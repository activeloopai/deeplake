import numpy as np
from typing import Generator, Tuple, Optional, Any
import requests
from collections import Counter
from deeplake.integrations.labelbox.deeplake_utils import (
    generic_tensor_create_kwargs_,
    image_tensor_create_kwargs_,
)


def is_remote_resource_public_(url):
    try:
        response = requests.head(url, allow_redirects=True)
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        return False


def filter_video_paths_(video_paths, strategy):
    if strategy == "all":
        return video_paths
    unique_paths = set(video_paths)
    if strategy == "fail":
        if len(unique_paths) != len(video_paths):
            counter = Counter(video_paths)
            duplicates = [k for k, v in counter.items() if v > 1]
            raise ValueError("Duplicate video paths detected: " + ", ".join(duplicates))
        return video_paths

    if strategy == "skip":
        if len(unique_paths) != len(video_paths):
            counter = Counter(video_paths)
            duplicates = [k for k, v in counter.items() if v > 1]
            print(
                "Duplicate video paths detected, filtering out duplicates: ", duplicates
            )
        return list(unique_paths)

    raise ValueError(f"Invalid data upload strategy: {strategy}")


def frame_generator_(
    video_path: str, header: Optional[dict[str, Any]] = None, retries: int = 5
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Generate frames from a video file.

    Parameters:
    video_path (str): Path to the video file
    header (dict, optional): Optional request header for authorization

    Yields:
    tuple: (frame_number, frame_data)
        - frame_number (int): The sequential number of the frame
        - frame_data (numpy.ndarray): The frame image data
    """

    def get_video_container(current_retries):
        import av

        try:
            return av.open(video_path, options=header)
        except Exception as e:
            if current_retries > 0:
                print(f"Failed opening video: {e}. Retrying...")
                return get_video_container(current_retries - 1)
            else:
                raise e

    try:
        container = get_video_container(retries)
        print(f"Start generating frames from {video_path}")
        frame_num = 0
        for frame in container.decode(video=0):
            yield frame_num, frame.to_ndarray(format="rgb24")
            frame_num += 1
    except Exception as e:
        print(f"Failed generating frames: {e}")


def frames_batch_generator_(
    video_path: str,
    header: Optional[dict[str, Any]] = None,
    batch_size=100,
    retries: int = 5,
):
    frames, indexes = [], []
    for frame_num, frame in frame_generator_(video_path, header, retries):
        frames.append(frame)
        indexes.append(frame_num)
        if len(frames) < batch_size:
            continue
        yield indexes, frames
        frames, indexes = [], []

    if len(frames):
        yield indexes, frames


def external_url_from_video_project_(p):
    if "external_id" in p["data_row"]:
        return p["data_row"]["external_id"]
    return p["data_row"]["row_data"]


def validate_video_project_data_impl_(project_j, deeplake_dataset, project_id):
    if "labelbox_meta" not in deeplake_dataset.info:
        return False
    info = deeplake_dataset.info["labelbox_meta"]

    if info["type"] != "video":
        return False

    if project_id != info["project_id"]:
        return False

    if len(project_j) != len(info["sources"]):
        return False

    if len(project_j) == 0:
        return True

    ontology_ids = set()

    for p in project_j:
        url = external_url_from_video_project_(p)
        if url not in info["sources"]:
            return False

        ontology_ids.add(p["projects"][project_id]["project_details"]["ontology_id"])

    if len(ontology_ids) != 1:
        return False

    return True


PROJECT_DATA_VALIDATION_MAP_ = {"video": validate_video_project_data_impl_}


def validate_project_data_(proj, ds, project_id, type):
    if type not in PROJECT_DATA_VALIDATION_MAP_:
        raise ValueError(f"Invalid project data type: {type}")
    return PROJECT_DATA_VALIDATION_MAP_[type](proj, ds, project_id)


def validate_video_project_creation_data_impl_(project_j, project_id):
    if len(project_j) == 0:
        return True

    for p in project_j:
        for l in p["projects"][project_id]["labels"]:
            if l["label_kind"] != "Video":
                return False

        if p["media_attributes"]["asset_type"] != "video":
            return False

    return True


PROJECT_DATA_CREATION_VALIDATION_MAP_ = {
    "video": validate_video_project_creation_data_impl_
}


def validate_project_creation_data_(proj, project_id, type):
    if type not in PROJECT_DATA_CREATION_VALIDATION_MAP_:
        raise ValueError(f"Invalid project creation data type: {type}")
    return PROJECT_DATA_CREATION_VALIDATION_MAP_[type](proj, project_id)


def labelbox_get_project_json_with_id_(client, project_id, fail_on_error=False):
    # Set the export params to include/exclude certain fields.

    import labelbox as lb  # type: ignore

    export_params = {
        "attachments": True,
        "metadata_fields": True,
        "data_row_details": True,
        "project_details": True,
        "label_details": True,
        "performance_details": True,
        "interpolated_frames": False,
    }

    # Note: Filters follow AND logic, so typically using one filter is sufficient.
    filters = {
        "last_activity_at": ["2000-01-01 00:00:00", "2050-01-01 00:00:00"],
    }

    project = client.get_project(project_id)
    export_task = project.export(params=export_params, filters=filters)

    print(
        "requesting project info from labelbox with id",
        project_id,
        "export task id",
        export_task.uid,
    )
    export_task.wait_till_done()

    # Stream results and errors
    if export_task.has_errors():

        def error_handler(error):
            if fail_on_error:
                raise ValueError(f"Labelbox export task failed with errors: {error}")
            print("Labelbox export task failed with errors:", error)

        export_task.get_buffered_stream(stream_type=lb.StreamType.ERRORS).start(
            stream_handler=error_handler
        )

    if export_task.has_result():
        # Start export stream
        stream = export_task.get_buffered_stream()

        print("project info is ready for project with id", project_id)
        return [data_row.json for data_row in stream]

    raise ValueError("This should not happen")


def create_tensors_default_(ds):
    ds.add_column("frames", **image_tensor_create_kwargs_())
    ds.add_column("frame_idx", **generic_tensor_create_kwargs_("int32"))
    ds.add_column("video_idx", **generic_tensor_create_kwargs_("int32"))


def fill_data_default_(ds, group_ids, indexes, frames):
    ds.extend(["frames", "video_idx", "frame_idx"], [frames, group_ids, indexes])
