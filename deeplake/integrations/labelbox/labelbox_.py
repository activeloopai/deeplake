import deeplake
import os
import labelbox as lb  # type: ignore
import tempfile

from deeplake.integrations.labelbox.labelbox_utils import *
from deeplake.integrations.labelbox.labelbox_converter import labelbox_video_converter
from deeplake.integrations.labelbox.v3_converters import *


def converter_for_video_project_with_id(
    project_id,
    client,
    deeplake_ds_loader,
    lb_api_key,
    group_mapping=None,
    fail_on_error=False,
    fail_on_labelbox_project_export_error=False,
):
    """
    Creates a converter for Labelbox video project to a Deeplake dataset format based on annotation types.

    Args:
        project_id (str): The unique identifier for the Labelbox project to convert.
        client (LabelboxClient): An authenticated Labelbox client instance for API access.
        deeplake_ds_loader (callable): A function that creates/loads a Deeplake dataset given a name.
        lb_api_key (str): Labelbox API key for authentication.
        group_mapping (dict, optional): A dictionary mapping annotation kinds (labelbox_kind) to the desired tensor group name (tensor_name). This mapping determines whether annotations of the same kind should be grouped into the same tensor or kept separate.
        fail_on_error (bool, optional): Whether to raise an exception if data validation fails. Defaults to False.
        fail_on_labelbox_project_export_error (bool, optional): Whether to raise an exception if Labelbox project export fails. Defaults to False.

    Returns:
        labelbox_type_converter or None: Returns a labelbox_type_converter if successful, None if no data is found.
        The returned converter can be used to apply Labelbox annotations to a Deeplake dataset.

    Raises:
        Exception: If project data validation fails.

    Example:
        >>> client = LabelboxClient(api_key='your_api_key')
        >>> converter = converter_for_video_project_with_id(
        ...     '<project_id>',
        ...     client,
        ...     lambda name: deeplake.load(name),
        ...     'your_api_key',
        ...     group_mapping={"raster-segmentation": "mask"}
        ... )
        >>> if converter:
        ...     # Use converter to apply annotations
        ...     ds = converter.dataset_with_applied_annotations()

    Notes:
        - Supports Video ontology from labelbox.
        - The function first validates the project data before setting up converters.
    """
    project_json = labelbox_get_project_json_with_id_(client, project_id, fail_on_labelbox_project_export_error)

    if len(project_json) == 0:
        print("no data")
        return None

    ds_name = project_json[0]["projects"][project_id]["name"]
    deeplake_dataset = deeplake_ds_loader(ds_name)

    if not validate_project_data_(project_json, deeplake_dataset, project_id, "video"):
        if fail_on_error:
            raise Exception("Data validation failed")

    ontology_id = project_json[0]["projects"][project_id]["project_details"][
        "ontology_id"
    ]
    ontology = client.get_ontology(ontology_id)

    converters = {
        "rectangle": bbox_converter_,
        "radio": radio_converter_,
        "checklist": checkbox_converter_,
        "point": point_converter_,
        "line": line_converter_,
        "raster-segmentation": raster_segmentation_converter_,
        "text": text_converter_,
    }
    return labelbox_video_converter(
        ontology,
        converters,
        project_json,
        project_id,
        deeplake_dataset,
        {"ds": deeplake_dataset, "lb_api_key": lb_api_key},
        group_mapping=group_mapping,
    )


def create_dataset_for_video_annotation_with_custom_data_filler(
    deeplake_ds_path,
    video_paths,
    lb_client,
    data_filler,
    deeplake_creds=None,
    deeplake_org_id=None,
    deeplake_token=None,
    overwrite=False,
    lb_ontology=None,
    lb_batch_priority=5,
    lb_dataset_name=None,
    fail_on_error=False,
    video_generator_batch_size=100
):
    """
    Creates a Deeplake dataset for video annotation and sets up corresponding Labelbox project.
    Processes videos frame-by-frame using a custom data filler function.

    Args:
       deeplake_ds_path (str): Path where the Deeplake dataset will be created/stored.
           Can be local path or remote path (e.g. 'hub://org/dataset')
       video_paths (List[str]): List of paths to video files to be processed can be either all local or all pre-signed remote. 
       lb_client (LabelboxClient): Authenticated Labelbox client instance
       data_filler (dict): Dictionary containing two functions:
           - 'create_tensors': callable(ds) -> None
               Creates necessary tensors in the dataset
           - 'fill_data': callable(ds, idx, frame_num, frame) -> None
               Fills dataset with processed frame data
       deeplake_creds (dict): Dictionary containing credentials for deeplake
       deeplake_org_id (str, optional): Organization ID for Deeplake cloud storage.
       deeplake_token (str, optional): Authentication token for Deeplake cloud storage.
       overwrite (bool, optional): Whether to overwrite existing dataset. Defaults to False
       lb_ontology (Ontology, optional): Labelbox ontology to connect to project. Defaults to None
       lb_batch_priority (int, optional): Priority for Labelbox batches. Defaults to 5
       lb_dataset_name (str, optional): Custom name for Labelbox dataset.
           Defaults to deeplake_ds_path basename + '_from_deeplake'
       fail_on_error (bool, optional): Whether to raise an exception if data validation fails. Defaults to False

    Returns:
       Dataset: Created Deeplake dataset containing processed video frames and metadata for Labelbox project
    """
    ds = deeplake.empty(
        deeplake_ds_path,
        creds=deeplake_creds,
        org_id=deeplake_org_id,
        token=deeplake_token,
        overwrite=overwrite,
    )

    data_filler["create_tensors"](ds)

    for idx, video_path in enumerate(video_paths):
        for frame_indexes, frames in frames_batch_generator_(video_path, batch_size=video_generator_batch_size):
            data_filler["fill_data"](ds, [idx] * len(frames), frame_indexes, frames)

    if lb_dataset_name is None:
        lb_dataset_name = os.path.basename(deeplake_ds_path) + "_from_deeplake"

    assets = video_paths

    # validate paths
    all_local = [os.path.exists(p) for p in video_paths]
    if any(all_local) and not all(all_local):
        raise Exception(f'video paths must be all local or all remote: {video_paths}')

    if len(all_local):
        if not all_local[0]:
            assets = [{
            "row_data": p,
            "media_type": "VIDEO",
            "metadata_fields": [],
            "attachments": []
        } for p in video_paths]

    print('uploading videos to labelbox')
    lb_ds = lb_client.create_dataset(name=lb_dataset_name)
    task = lb_ds.create_data_rows(assets)
    task.wait_till_done()

    if task.errors:
        raise Exception(f'failed to upload videos to labelbox: {task.errors}')

    print('successfuly uploaded videos to labelbox')

    # Create a new project
    project = lb_client.create_project(
        name=os.path.basename(deeplake_ds_path), media_type=lb.MediaType.Video
    )

    ds.info["labelbox_meta"] = {
        "project_id": project.uid,
        "type": "video",
        "sources": video_paths,
        "project_name": os.path.basename(deeplake_ds_path),
    }

    task = project.create_batches_from_dataset(
        name_prefix=lb_dataset_name, dataset_id=lb_ds.uid, priority=lb_batch_priority
    )

    if task.errors():
        if fail_on_error:
            raise Exception(f"Error creating batches: {task.errors()}")

    if lb_ontology:
        project.connect_ontology(lb_ontology)

    ds.commit()

    return ds


def create_dataset_for_video_annotation(
    deeplake_ds_path,
    video_paths,
    lb_client,
    deeplake_creds=None,
    deeplake_org_id=None,
    deeplake_token=None,
    overwrite=False,
    lb_ontology=None,
    lb_batch_priority=5,
    fail_on_error=False,
    video_generator_batch_size=100,
):
    """
    See create_dataset_for_video_annotation_with_custom_data_filler for complete documentation.

    The only difference is this function uses default tensor creation and data filling functions:
    - create_tensors_default_: Creates default tensor structure
    - fill_data_default_: Fills tensors with default processing
    """
    return create_dataset_for_video_annotation_with_custom_data_filler(
        deeplake_ds_path,
        video_paths,
        lb_client,
        data_filler={
            "create_tensors": create_tensors_default_,
            "fill_data": fill_data_default_,
        },
        deeplake_creds=deeplake_creds,
        deeplake_org_id=deeplake_org_id,
        deeplake_token=deeplake_token,
        lb_ontology=lb_ontology,
        lb_batch_priority=lb_batch_priority,
        overwrite=overwrite,
        fail_on_error=fail_on_error,
        video_generator_batch_size=video_generator_batch_size,
    )


def create_dataset_from_video_annotation_project_with_custom_data_filler(
    deeplake_ds_path,
    project_id,
    lb_client,
    lb_api_key,
    data_filler,
    deeplake_creds=None,
    deeplake_org_id=None,
    deeplake_token=None,
    overwrite=False,
    fail_on_error=False,
    url_presigner=None,
    video_generator_batch_size=100,
    fail_on_labelbox_project_export_error=False,
):
    """
    Creates a Deeplake dataset from an existing Labelbox video annotation project using custom data processing.
    Downloads video frames from Labelbox and processes them using provided data filler functions.

    Args:
       deeplake_ds_path (str): Path where the Deeplake dataset will be created/stored.
           Can be local path or cloud path (e.g. 'hub://org/dataset')
       project_id (str): Labelbox project ID to import data from
       lb_client (LabelboxClient): Authenticated Labelbox client instance
       lb_api_key (str): Labelbox API key for accessing video frames
       data_filler (dict): Dictionary containing two functions:
           - 'create_tensors': callable(ds) -> None
               Creates necessary tensors in the dataset
           - 'fill_data': callable(ds, group_ids, indexes, frames) -> None
               Fills dataset with processed frame batches
       deeplake_creds (dict): Dictionary containing credentials for deeplake
       deeplake_org_id (str, optional): Organization ID for Deeplake cloud storage.
       deeplake_token (str, optional): Authentication token for Deeplake cloud storage.
           Required if using hub:// path. Defaults to None
       overwrite (bool, optional): Whether to overwrite existing dataset. Defaults to False
       fail_on_error (bool, optional): Whether to raise an exception if data validation fails. Defaults to False
       url_presigner (callable, optional): Function that takes a URL and returns a pre-signed URL and headers (str, dict). Default will use labelbox access token to access the data. Is useful when used cloud storage integrations.
       video_generator_batch_size (int, optional): Number of frames to process in each batch. Defaults to 100
       fail_on_labelbox_project_export_error (bool, optional): Whether to raise an exception if Labelbox project export fails. Defaults to False

    Returns:
       Dataset: Created Deeplake dataset containing processed video frames and Labelbox metadata.
       Returns empty dataset if no data found in project.

    Notes:
        - The function does not fetch the annotations from Labelbox, only the video frames. After creating the dataset, use the converter to apply annotations.
    """
    ds = deeplake.empty(
        deeplake_ds_path,
        overwrite=overwrite,
        creds=deeplake_creds,
        org_id=deeplake_org_id,
        token=deeplake_token,
    )
    data_filler["create_tensors"](ds)

    proj = labelbox_get_project_json_with_id_(lb_client, project_id, fail_on_labelbox_project_export_error)
    if len(proj) == 0:
        print("no data")
        return ds

    if not validate_project_creation_data_(proj, project_id, "video"):
        if fail_on_error:
            raise Exception("Data validation failed")

    video_files = []

    if url_presigner is None:
        def default_presigner(url):
            if lb_api_key is None:
                return url, {}
            return url, {"headers": {"Authorization": f"Bearer {lb_api_key}"}}
        url_presigner = default_presigner

    for idx, p in enumerate(proj):
        video_url = p["data_row"]["row_data"]
        header = None
        if not os.path.exists(video_url):
            if not is_remote_resource_public_(video_url):
                video_url, header = url_presigner(video_url)
        for frame_indexes, frames in frames_batch_generator_(video_url, header=header, batch_size=video_generator_batch_size):
            data_filler["fill_data"](ds, [idx] * len(frames), frame_indexes, frames)
        video_files.append(external_url_from_video_project_(p))

    ds.info["labelbox_meta"] = {
        "project_id": project_id,
        "type": "video",
        "sources": video_files,
        "project_name": proj[0]["projects"][project_id]["name"],
    }

    ds.commit()

    return ds


def create_dataset_from_video_annotation_project(
    deeplake_ds_path,
    project_id,
    lb_client,
    lb_api_key,
    deeplake_creds=None,
    deeplake_org_id=None,
    deeplake_token=None,
    overwrite=False,
    fail_on_error=False,
    url_presigner=None,
    video_generator_batch_size=100,
    fail_on_labelbox_project_export_error=False,
):
    """
    See create_dataset_from_video_annotation_project_with_custom_data_filler for complete documentation.

    The only difference is this function uses default tensor creation and data filling functions:
    - create_tensors_default_: Creates default tensor structure
    - fill_data_default_: Fills tensors with default processing
    """
    return create_dataset_from_video_annotation_project_with_custom_data_filler(
        deeplake_ds_path,
        project_id,
        lb_client,
        lb_api_key,
        data_filler={
            "create_tensors": create_tensors_default_,
            "fill_data": fill_data_default_,
        },
        deeplake_creds=deeplake_creds,
        deeplake_org_id=deeplake_org_id,
        deeplake_token=deeplake_token,
        overwrite=overwrite,
        fail_on_error=fail_on_error,
        url_presigner=url_presigner,
        video_generator_batch_size=video_generator_batch_size,
        fail_on_labelbox_project_export_error=fail_on_labelbox_project_export_error,
    )
