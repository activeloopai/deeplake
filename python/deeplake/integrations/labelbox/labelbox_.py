import json
import deeplake
import os
import uuid

from deeplake.integrations.labelbox.labelbox_utils import *
from deeplake.integrations.labelbox.labelbox_converter import labelbox_image_converter, labelbox_video_converter
from deeplake.integrations.labelbox.converters import *
from deeplake.integrations.labelbox.labelbox_metadata_utils import *
from deeplake.integrations.labelbox.deeplake_utils import *

def _converter_for_media_project_with_id(
    project_id,
    deeplake_ds_loader,
    lb_api_key,
    media_type,
    group_mapping=None,
    fail_on_error=False,
    fail_on_labelbox_project_export_error=False,
    generate_metadata=True,
    metadata_prefix="lb_meta",
    project_json=None,
) -> Optional[labelbox_video_converter]:
    """
    Creates a converter for Labelbox media project to a Deeplake dataset format based on annotation types.

    Args:
        project_id (str): The unique identifier for the Labelbox project to convert.
        deeplake_ds_loader (callable): A function that creates/loads a Deeplake dataset given a name.
        lb_api_key (str): Labelbox API key for authentication.
        group_mapping (dict, optional): A dictionary mapping annotation kinds (labelbox_kind) to the desired tensor group name (tensor_name). This mapping determines whether annotations of the same kind should be grouped into the same tensor or kept separate.
        fail_on_error (bool, optional): Whether to raise an exception if data validation fails. Defaults to False.
        fail_on_labelbox_project_export_error (bool, optional): Whether to raise an exception if Labelbox project export fails. Defaults to False.
        generate_metadata (bool, optional): Whether to generate metadata tensors. Defaults to True.
        metadata_prefix (str, optional): Prefix for metadata tensors. Defaults to "lb_meta". Will be ignored if generate_metadata is False.
        project_json (Any, optional): Optional project JSON data to use for conversion. If not provided, the function will fetch the project data from Labelbox.

    Returns:
        Optional[labelbox_video_converter]: Returns a labelbox_type_converter if successful, None if no data is found.
        The returned converter can be used to apply Labelbox annotations to a Deeplake dataset.

    Raises:
        Exception: If project data validation fails.

    Example:
        >>> converter = converter_for_media_project_with_id(
        ...     '<project_id>',
        ...     lambda name: deeplake.load(name),
        ...     'your_api_key',
        ...     group_mapping={"raster-segmentation": "mask"}
        ... )
        >>> if converter:
        ...     # Use converter to apply annotations
        ...     ds = converter.dataset_with_applied_annotations()

    Notes:
        - Supports both Image and Video ontologies from labelbox.
        - The function first validates the project data before setting up converters.
    """
    assert media_type in ["image", "video"], "media_type must be either 'image' or 'video'"
    import labelbox as lb  # type: ignore

    client = lb.Client(api_key=lb_api_key)
    if project_json is None:
        project_json = labelbox_get_project_json_with_id_(
            client, project_id, fail_on_labelbox_project_export_error
        )

    if len(project_json) == 0:
        print("no data")
        return None

    ds_name = project_json[0]["projects"][project_id]["name"]
    wrapped_dataset = dataset_wrapper(deeplake_ds_loader(ds_name))

    if not validate_project_data_(project_json, wrapped_dataset, project_id, media_type):
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
        "polygon": polygon_converter_,
    }
    if generate_metadata:
        tensor_name_generator = lambda name: (
            f"{metadata_prefix}/{name}" if metadata_prefix else name
        )
        metadata_generators = {
            tensor_name_generator(f"{media_type}_name"): {
                "generator": get_media_name_from_media_project_,
                "create_tensor_kwargs": text_tensor_create_kwargs_(),
            },
            tensor_name_generator("id"): {
                "generator": get_data_row_id_from_media_project_,
                "create_tensor_kwargs": text_tensor_create_kwargs_(),
            },
            tensor_name_generator("row_data"): {
                "generator": lambda project, ctx: get_data_row_url_from_media_project_(
                    project, ctx
                ),
                "create_tensor_kwargs": text_tensor_create_kwargs_(),
            },
            tensor_name_generator("label_creator"): {
                "generator": get_label_creator_from_media_project_,
                "create_tensor_kwargs": text_tensor_create_kwargs_(),
            },
           
            tensor_name_generator("width"): {
                "generator": get_width_from_media_project_,
                "create_tensor_kwargs": generic_tensor_create_kwargs_("int32"),
            },
            tensor_name_generator("height"): {
                "generator": get_height_from_media_project_,
                "create_tensor_kwargs": generic_tensor_create_kwargs_("int32"),
            },
            tensor_name_generator("ontology_id"): {
                "generator": get_ontology_id_from_media_project_,
                "create_tensor_kwargs": text_tensor_create_kwargs_(),
            },
            tensor_name_generator("project_name"): {
                "generator": get_project_name_from_media_project_,
                "create_tensor_kwargs": text_tensor_create_kwargs_(),
            },
            tensor_name_generator("dataset_name"): {
                "generator": get_dataset_name_from_media_project_,
                "create_tensor_kwargs": text_tensor_create_kwargs_(),
            },
            tensor_name_generator("dataset_id"): {
                "generator": get_dataset_id_from_media_project_,
                "create_tensor_kwargs": text_tensor_create_kwargs_(),
            },
            tensor_name_generator("global_key"): {
                "generator": get_global_key_from_media_project_,
                "create_tensor_kwargs": text_tensor_create_kwargs_(),
            },
         
        }
        if media_type == "video":
            metadata_generators[tensor_name_generator("frame_rate")] = {
                "generator": get_frame_rate_from_video_project_,
                "create_tensor_kwargs": generic_tensor_create_kwargs_("int32"),
            }
            metadata_generators[tensor_name_generator("frame_count")]= {
                "generator": get_frame_count_from_video_project_,
                "create_tensor_kwargs": generic_tensor_create_kwargs_("int32"),
            }
            metadata_generators[tensor_name_generator("frame_number")] = {
                "generator": lambda project, ctx: ctx["frame_idx"]
                + 1,  # 1-indexed frame number
                "create_tensor_kwargs": generic_tensor_create_kwargs_("int32"),
            }
            metadata_generators[tensor_name_generator("current_frame_name")] = {
                "generator": lambda project, ctx: f"{get_media_name_from_media_project_(project, ctx)}_{(ctx['frame_idx'] + 1):06d}",  # 1-indexed frame number
                "create_tensor_kwargs": text_tensor_create_kwargs_(),
            }

    else:
        metadata_generators = None

    return labelbox_video_converter(
        ontology,
        converters,
        project_json,
        project_id,
        wrapped_dataset,
        {"ds": wrapped_dataset, "lb_api_key": lb_api_key},
        metadata_generators=metadata_generators,
        group_mapping=group_mapping,
    ) if media_type == "video" else labelbox_image_converter(
        ontology,
        converters,
        project_json,
        project_id,
        wrapped_dataset,
        {"ds": wrapped_dataset, "lb_api_key": lb_api_key},
        metadata_generators=metadata_generators,
        group_mapping=group_mapping,
    )

def converter_for_video_project_with_id(
    project_id,
    deeplake_ds_loader,
    lb_api_key,
    group_mapping=None,
    fail_on_error=False,
    fail_on_labelbox_project_export_error=False,
    generate_metadata=True,
    metadata_prefix="lb_meta",
    project_json=None,
) -> Optional[labelbox_video_converter]:
    """
    Creates a converter for Labelbox video project to a Deeplake dataset format based on annotation types.

    Args:
        project_id (str): The unique identifier for the Labelbox project to convert.
        deeplake_ds_loader (callable): A function that creates/loads a Deeplake dataset given a name.
        lb_api_key (str): Labelbox API key for authentication.
        group_mapping (dict, optional): A dictionary mapping annotation kinds (labelbox_kind) to the desired tensor group name (tensor_name). This mapping determines whether annotations of the same kind should be grouped into the same tensor or kept separate.
        fail_on_error (bool, optional): Whether to raise an exception if data validation fails. Defaults to False.
        fail_on_labelbox_project_export_error (bool, optional): Whether to raise an exception if Labelbox project export fails. Defaults to False.
        generate_metadata (bool, optional): Whether to generate metadata tensors. Defaults to True.
        metadata_prefix (str, optional): Prefix for metadata tensors. Defaults to "lb_meta". Will be ignored if generate_metadata is False.
        project_json (Any, optional): Optional project JSON data to use for conversion. If not provided, the function will fetch the project data from Labelbox.


    Returns:
        Optional[labelbox_video_converter]: Returns a labelbox_type_converter if successful, None if no data is found.
        The returned converter can be used to apply Labelbox annotations to a Deeplake dataset.

    Raises:
        Exception: If project data validation fails.

    Example:
        >>> converter = converter_for_video_project_with_id(
        ...     '<project_id>',
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
    return _converter_for_media_project_with_id(
        project_id,
        deeplake_ds_loader,
        lb_api_key,
        media_type="video",
        group_mapping=group_mapping,
        fail_on_error=fail_on_error,
        fail_on_labelbox_project_export_error=fail_on_labelbox_project_export_error,
        generate_metadata=generate_metadata,
        metadata_prefix=metadata_prefix,
        project_json=project_json,
    )

def converter_for_image_project_with_id(
    project_id,
    deeplake_ds_loader,
    lb_api_key,
    group_mapping=None,
    fail_on_error=False,
    fail_on_labelbox_project_export_error=False,
    generate_metadata=True,
    metadata_prefix="lb_meta",
    project_json=None,
):
    return _converter_for_media_project_with_id(
        project_id,
        deeplake_ds_loader,
        lb_api_key,
        media_type="image",
        group_mapping=group_mapping,
        fail_on_error=fail_on_error,
        fail_on_labelbox_project_export_error=fail_on_labelbox_project_export_error,
        generate_metadata=generate_metadata,
        metadata_prefix=metadata_prefix,
        project_json=project_json,
    )

def create_labelbox_annotation_project(
    video_paths,
    lb_dataset_name,
    lb_project_name,
    lb_api_key,
    lb_ontology=None,
    lb_batch_priority=5,
    data_upload_strategy="fail",
    lb_batches_name=None,
    lb_iam_integration_id="DEFAULT",
    lb_global_key_generator=lambda x: str(uuid.uuid4()),
    media_type="VIDEO"  # Default to VIDEO, can be changed to IMAGE if needed
):
    """
    Creates labelbox dataset for video annotation and sets up corresponding Labelbox project.

    Args:
       video_paths (List[str]): List of paths to video files to be processed can be either all local or all pre-signed remote.
       lb_dataset_name (str): Name for Labelbox dataset.
       lb_project_name (str): Name for Labelbox project.
       lb_api_key (str): Labelbox API key for authentication.
       lb_ontology (Ontology, optional): Labelbox ontology to connect to project. Defaults to None
       lb_batch_priority (int, optional): Priority for Labelbox batches. Defaults to 5
       data_upload_strategy (str, optional): Strategy for uploading data to Labelbox. Can be 'fail', 'skip', or 'all'. Defaults to 'fail'
       lb_batches_name (str, optional): Name for Labelbox batches. Defaults to None. If None, will use lb_dataset_name + '_batch-'
       lb_iam_integration_id (str, optional): IAM integration id for Labelbox. Defaults to 'DEFAULT'
       lb_global_key_generator (callable, optional): Function to generate global keys for data rows. Defaults to lambda x: str(uuid.uuid4())
    """
    assert media_type in ["VIDEO", "IMAGE"], "media_type must be either 'VIDEO' or 'IMAGE'"
    media_string = "video" if media_type == "VIDEO" else "image"
    import labelbox as lb  # type: ignore

    lb_client = lb.Client(api_key=lb_api_key)

    video_paths = filter_video_paths_(video_paths, data_upload_strategy)

    assets = video_paths

    # validate paths
    all_local = [os.path.exists(p) for p in video_paths]
    if any(all_local) and not all(all_local):
        raise Exception(f"{media_string} paths must be all local or all remote: {video_paths}")

    if len(all_local):
        if not all_local[0]:
            assets = [
                {
                    "row_data": p,
                    "global_key": lb_global_key_generator(p),
                    "media_type": media_string.upper(),
                    "metadata_fields": [],
                    "attachments": [],
                }
                for p in video_paths
            ]

    if lb_iam_integration_id and lb_iam_integration_id != "DEFAULT":
        lb_org = lb_client.get_organization()
        integrations = lb_org.get_iam_integrations()
        tmp_integration = None
        for integration in integrations:
            if integration.uid == lb_iam_integration_id:
                tmp_integration = integration
                break
        if tmp_integration is None:
            raise Exception(f"iam integration {lb_iam_integration_id} not found")
        lb_iam_integration = tmp_integration
    else:
        lb_iam_integration = lb_iam_integration_id

    print(
        f"uploading {media_string}s to labelbox",
        (
            f"using iam integration: {lb_iam_integration}"
            if lb_iam_integration != "DEFAULT"
            else ""
        ),
    )

    lb_ds = lb_client.create_dataset(
        iam_integration=lb_iam_integration, name=lb_dataset_name
    )
    task = lb_ds.create_data_rows(assets)
    task.wait_till_done()

    if task.errors:
        raise Exception(f"failed to upload {media_string}s to labelbox: {task.errors}")

    if len(all_local):
        if all_local[0]:
            print("assigning global keys to data rows")
            rows = [
                {
                    "data_row_id": lb_ds.data_row_for_external_id(p).uid,
                    "global_key": str(uuid.uuid4()),
                }
                for p in video_paths
            ]
            res = lb_client.assign_global_keys_to_data_rows(rows)
            if res["status"] != "SUCCESS":
                raise Exception(f"failed to assign global keys to data rows: {res}")

    print(f"successfuly uploaded {media_string}s to labelbox")

    # Create a new project
    project = lb_client.create_project(
        name=lb_project_name, media_type=(lb.MediaType.Video if media_type == "VIDEO" else lb.MediaType.Image)
    )

    if lb_batches_name is None:
        lb_batches_name = lb_dataset_name + "_batch-"

    task = project.create_batches_from_dataset(
        name_prefix=lb_batches_name, dataset_id=lb_ds.uid, priority=lb_batch_priority
    )

    if task.errors():
        raise Exception(f"Error creating batches: {task.errors()}")

    if lb_ontology:
        project.connect_ontology(lb_ontology)

def _create_dataset_from_annotation_project_with_custom_data_filler(
    deeplake_ds_path,
    project_id,
    lb_api_key,
    data_filler,
    media_type,
    data_processor,
    deeplake_creds=None,
    deeplake_org_id=None,
    deeplake_token=None,
    overwrite=False,
    fail_on_error=False,
    url_presigner=None,
    fail_on_labelbox_project_export_error=False,
    project_json=None,
    **kwargs
) -> Tuple[deeplake.Dataset, Any]:
    """
    Common helper function for creating datasets from Labelbox annotation projects.
    
    Args:
        media_type (str): Type of media ("image" or "video")
        data_processor (callable): Function that processes the media data and fills the dataset
        **kwargs: Additional arguments specific to media type (e.g., video_generator_batch_size)
    """
    import labelbox as lb  # type: ignore

    lb_client = lb.Client(api_key=lb_api_key)

    wrapped_dataset = dataset_wrapper.create(
        deeplake_ds_path,
        token=deeplake_token,
        org_id=deeplake_org_id,
        creds=deeplake_creds,
        overwrite=overwrite,
    )

    data_filler["create_tensors"](wrapped_dataset)

    if project_json is None:
        project_json = labelbox_get_project_json_with_id_(
            lb_client, project_id, fail_on_labelbox_project_export_error
        )
    if len(project_json) == 0:
        print("no data")
        return wrapped_dataset.ds, project_json

    if not validate_project_creation_data_(project_json, project_id, media_type):
        if fail_on_error:
            raise Exception("Data validation failed")

    if url_presigner is None:
        def default_presigner(url):
            if lb_api_key is None:
                return url, {}
            return url, {"headers": {"Authorization": f"Bearer {lb_api_key}"}}
        url_presigner = default_presigner

    media_files = data_processor(project_json, wrapped_dataset, data_filler, url_presigner, **kwargs)

    wrapped_dataset.metadata["labelbox_meta"] = {
        "project_id": project_id,
        "type": media_type,
        "sources": media_files,
        "project_name": project_json[0]["projects"][project_id]["name"],
    }

    return wrapped_dataset.ds, project_json

def _process_video_data(project_json, wrapped_dataset, data_filler, url_presigner, video_generator_batch_size=100):
    """Process video data and fill dataset."""
    video_files = []
    for idx, p in enumerate(project_json):
        video_url = p["data_row"]["row_data"]
        header = None
        if not os.path.exists(video_url):
            if not is_remote_resource_public_(video_url):
                video_url, header = url_presigner(video_url)
        for frame_indexes, frames in frames_batch_generator_(
            video_url, header=header, batch_size=video_generator_batch_size
        ):
            data_filler["fill_data"](
                wrapped_dataset, [idx] * len(frames), frame_indexes, frames
            )
        video_files.append(external_url_from_media_project_(p))
    return video_files

def _process_image_data(project_json, wrapped_dataset, data_filler, url_presigner):
    """Process image data and fill dataset."""
    image_files = []
    image_data = []
    
    indexic = []
    print("Processing image data...")
    for idx, p in enumerate(project_json):
        indexic.append(idx)
        image_url = p["data_row"]["row_data"]
        header = None
        if not os.path.exists(image_url):
            if not is_remote_resource_public_(image_url):
                image_url, headers_dict = url_presigner(image_url)
                header = headers_dict
        
        # Download the image
        image_array = download_image_from_url_(image_url, header)
        if image_array is not None:
            image_data.append(image_array)
            image_files.append(external_url_from_media_project_(p))
    
    # Fill the dataset with image data
    data_filler["fill_data"](wrapped_dataset, image_data, indexic)
    return image_files

def create_dataset_from_video_annotation_project(
    deeplake_ds_path,
    project_id,
    lb_api_key,
    deeplake_creds=None,
    deeplake_org_id=None,
    deeplake_token=None,
    overwrite=False,
    fail_on_error=False,
    url_presigner=None,
    video_generator_batch_size=100,
    fail_on_labelbox_project_export_error=False,
    project_json=None,
) -> Tuple[deeplake.Dataset, Any]:
    return create_dataset_from_video_annotation_project_with_custom_data_filler(
        deeplake_ds_path=deeplake_ds_path,
        project_id=project_id,
        lb_api_key=lb_api_key,
        data_filler={
            "create_tensors": create_video_tensors_default_,
            "fill_data": fill_video_data_default_,
        },
        deeplake_creds=deeplake_creds,
        deeplake_org_id=deeplake_org_id,
        deeplake_token=deeplake_token,
        overwrite=overwrite,
        fail_on_error=fail_on_error,
        url_presigner=url_presigner,
        video_generator_batch_size=video_generator_batch_size,
    )

def create_dataset_from_video_annotation_project_with_custom_data_filler(
    deeplake_ds_path,
    project_id,
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
    project_json=None,
) -> Tuple[deeplake.Dataset, Any]:
    """
    Creates a Deeplake dataset from an existing Labelbox video annotation project using custom data processing.
    Downloads video frames from Labelbox and processes them using provided data filler functions.

    Args:
       deeplake_ds_path (str): Path where the Deeplake dataset will be created/stored.
           Can be local path or cloud path (e.g. 'hub://org/dataset')
       project_id (str): Labelbox project ID to import data from
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
       project_json (Any, optional): Optional project JSON data to use for conversion. If not provided, the function will fetch the project data from Labelbox.

    Returns:
       Tuple: Created Deeplake dataset containing processed video frames and Labelbox metadata and the Labelbox project JSON.
       Returns empty dataset if no data found in project.

    Notes:
        - The function does not fetch the annotations from Labelbox, only the video frames. After creating the dataset, use the converter to apply annotations.
    """
    return _create_dataset_from_annotation_project_with_custom_data_filler(
        deeplake_ds_path=deeplake_ds_path,
        project_id=project_id,
        lb_api_key=lb_api_key,
        data_filler=data_filler,
        media_type="video",
        data_processor=_process_video_data,
        deeplake_creds=deeplake_creds,
        deeplake_org_id=deeplake_org_id,
        deeplake_token=deeplake_token,
        overwrite=overwrite,
        fail_on_error=fail_on_error,
        url_presigner=url_presigner,
        fail_on_labelbox_project_export_error=fail_on_labelbox_project_export_error,
        project_json=project_json,
        video_generator_batch_size=video_generator_batch_size,
    )

def create_dataset_from_image_annotation_project_with_custom_data_filler(
    deeplake_ds_path,
    project_id,
    lb_api_key,
    data_filler,
    deeplake_creds=None,
    deeplake_org_id=None,
    deeplake_token=None,
    overwrite=False,
    fail_on_error=False,
    url_presigner=None,
    fail_on_labelbox_project_export_error=False,
    project_json=None,
) -> Tuple[deeplake.Dataset, Any]:
    """
    Creates a Deeplake dataset from an existing Labelbox image annotation project using custom data processing.
    Downloads images from Labelbox and processes them using provided data filler functions.

    Args:
       deeplake_ds_path (str): Path where the Deeplake dataset will be created/stored.
           Can be local path or cloud path (e.g. 'hub://org/dataset')
       project_id (str): Labelbox project ID to import data from
       lb_api_key (str): Labelbox API key for accessing images
       data_filler (dict): Dictionary containing two functions:
           - 'create_tensors': callable(ds) -> None
               Creates necessary tensors in the dataset
           - 'fill_data': callable(ds, image_data) -> None
               Fills dataset with processed image data
       deeplake_creds (dict): Dictionary containing credentials for deeplake
       deeplake_org_id (str, optional): Organization ID for Deeplake cloud storage.
       deeplake_token (str, optional): Authentication token for Deeplake cloud storage.
           Required if using hub:// path. Defaults to None
       overwrite (bool, optional): Whether to overwrite existing dataset. Defaults to False
       fail_on_error (bool, optional): Whether to raise an exception if data validation fails. Defaults to False
       url_presigner (callable, optional): Function that takes a URL and returns a pre-signed URL and headers (str, dict). Default will use labelbox access token to access the data. Is useful when used cloud storage integrations.
       fail_on_labelbox_project_export_error (bool, optional): Whether to raise an exception if Labelbox project export fails. Defaults to False
       project_json (Any, optional): Optional project JSON data to use for conversion. If not provided, the function will fetch the project data from Labelbox.

    Returns:
       Tuple: Created Deeplake dataset containing processed images and Labelbox metadata and the Labelbox project JSON.
       Returns empty dataset if no data found in project.

    Notes:
        - The function does not fetch the annotations from Labelbox, only the images. After creating the dataset, use the converter to apply annotations.
    """
    return _create_dataset_from_annotation_project_with_custom_data_filler(
        deeplake_ds_path=deeplake_ds_path,
        project_id=project_id,
        lb_api_key=lb_api_key,
        data_filler=data_filler,
        media_type="image",
        data_processor=_process_image_data,
        deeplake_creds=deeplake_creds,
        deeplake_org_id=deeplake_org_id,
        deeplake_token=deeplake_token,
        overwrite=overwrite,
        fail_on_error=fail_on_error,
        url_presigner=url_presigner,
        fail_on_labelbox_project_export_error=fail_on_labelbox_project_export_error,
        project_json=project_json,
    )

def create_dataset_from_image_annotation_project(
    deeplake_ds_path,
    project_id,
    lb_api_key,
    deeplake_creds=None,
    deeplake_org_id=None,
    deeplake_token=None,
    overwrite=False,
    fail_on_error=False,
    url_presigner=None,
    fail_on_labelbox_project_export_error=False,
    project_json=None,
) -> Tuple[deeplake.Dataset, Any]:
    """
    See create_dataset_from_image_annotation_project_with_custom_data_filler for complete documentation.

    The only difference is this function uses default tensor creation and data filling functions:
    - create_image_tensors_default_: Creates default image tensor structure
    - fill_image_data_default_: Fills tensors with default processing
    """
    return create_dataset_from_image_annotation_project_with_custom_data_filler(
        deeplake_ds_path,
        project_id,
        lb_api_key,
        data_filler={
            "create_tensors": create_image_tensors_default_,
            "fill_data": fill_image_data_default_,
        },
        deeplake_creds=deeplake_creds,
        deeplake_org_id=deeplake_org_id,
        deeplake_token=deeplake_token,
        overwrite=overwrite,
        fail_on_error=fail_on_error,
        url_presigner=url_presigner,
        fail_on_labelbox_project_export_error=fail_on_labelbox_project_export_error,
        project_json=project_json,
    )