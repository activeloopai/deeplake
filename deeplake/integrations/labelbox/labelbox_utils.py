import numpy as np
from typing import Generator, Tuple
import labelbox as lb  # type: ignore
import av
import requests

def is_remote_resource_public_(url):
    try:
        response = requests.head(url, allow_redirects=True)
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        return False

def frame_generator_(
    video_path: str, header: dict, retries: int = 5
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

def frames_batch_generator_(video_path: str, header: dict=None, batch_size=100, retries: int = 5):
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
        return  p["data_row"]["external_id"]
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
    print('requesting project info from labelbox with id', project_id)
    # Set the export params to include/exclude certain fields.
    export_params = {
        "attachments": False,
        "metadata_fields": False,
        "data_row_details": False,
        "project_details": True,
        "label_details": False,
        "performance_details": False,
        # interpolated_frames does not work with the latest version of the API 6.2.0
        "interpolated_frames": False,
        "embeddings": False,
    }

    # Note: Filters follow AND logic, so typically using one filter is sufficient.
    filters = {
        "last_activity_at": ["2000-01-01 00:00:00", "2050-01-01 00:00:00"],
        "label_created_at": ["2000-01-01 00:00:00", "2050-01-01 00:00:00"],
    }

    project = client.get_project(project_id)
    export_task = project.export(params=export_params, filters=filters)

    export_task.wait_till_done()

    # Provide results with JSON converter
    # Returns streamed JSON output strings from export task results/errors, one by one

    projects = []

    # Callback used for JSON Converter
    def json_stream_handler(output: lb.BufferedJsonConverterOutput):
        projects.append(output.json)

    def error_stream_handler(error):
        if fail_on_error:
            raise Exception(f"Error during export: {error}")
        print(f"Error during export: {error}")

    if export_task.has_errors():
        export_task.get_buffered_stream(stream_type=lb.StreamType.ERRORS).start(
            stream_handler=error_stream_handler
        )

    if export_task.has_result():
        export_json = export_task.get_buffered_stream(
            stream_type=lb.StreamType.RESULT
        ).start(stream_handler=json_stream_handler)

    print('project info is ready for project with id', project_id)

    return projects


def create_tensors_default_(ds):
    ds.create_tensor("frames", htype="image", sample_compression="jpg")
    ds.create_tensor("frame_idx", htype="generic", dtype="int32")
    ds.create_tensor("video_idx", htype="generic", dtype="int32")


def fill_data_default_(ds, group_ids, indexes, frames):
    ds["frames"].extend(frames)
    ds["video_idx"].extend(group_ids)
    ds["frame_idx"].extend(indexes)
