import deeplake
import os
import labelbox as lb

from deeplake.integrations.labelbox.labelbox_utils import *
from deeplake.integrations.labelbox.labelbox_converter import labelbox_video_converter
from deeplake.integrations.labelbox.v3_converters import *

def converter_for_video_project_with_id(project_id, client, deeplake_ds_loader, lb_api_key):
    """
    Creates a converter for Labelbox video project to a Deeplake dataset format based on annotation types.

    Args:
        project_id (str): The unique identifier for the Labelbox project to convert.
        client (LabelboxClient): An authenticated Labelbox client instance for API access.
        deeplake_ds_loader (callable): A function that creates/loads a Deeplake dataset given a name.
        lb_api_key (str): Labelbox API key for authentication.

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
        ...     'your_api_key'
        ... )
        >>> if converter:
        ...     # Use converter to apply annotations
        ...     ds = converter.dataset_with_applied_annotations()

    Notes:
        - Supports Video ontology from labelbox.
        - The function first validates the project data before setting up converters.
    """
    project_json = labelbox_get_project_json_with_id_(client, project_id)

    if len(project_json) == 0:
        print("no data")
        return None
    
    ds_name = project_json[0]["projects"][project_id]['name']
    deeplake_dataset = deeplake_ds_loader(ds_name)

    if not validate_project_data_(project_json, deeplake_dataset, project_id, 'video'):
        raise Exception("Data validation failed")

    ontology_id = project_json[0]["projects"][project_id]["project_details"]["ontology_id"]
    ontology = client.get_ontology(ontology_id)

    converters = {
        'rectangle': bbox_converter_,
        'radio': radio_converter_,
        'checklist': checkbox_converter_,
        'point': point_converter_,
        'line': line_converter_,
        'raster-segmentation': raster_segmentation_converter_,
        'text': text_converter_
    }
    return labelbox_video_converter(ontology, converters, project_json, project_id, deeplake_dataset, {'ds': deeplake_dataset, 'lb_api_key': lb_api_key})

def create_dataset_for_video_annotation_with_custom_data_filler(deeplake_ds_path, video_paths, lb_client, data_filler, deeplake_token=None, overwrite=False, lb_ontology=None, lb_batch_priority=5, lb_dataset_name=None):
    """
    Creates a Deeplake dataset for video annotation and sets up corresponding Labelbox project.
    Processes videos frame-by-frame using a custom data filler function.

    Args:
       deeplake_ds_path (str): Path where the Deeplake dataset will be created/stored. 
           Can be local path or remote path (e.g. 'hub://org/dataset')
       video_paths (List[str]): List of paths to video files to be processed can be local or pre-signed remote.
       lb_client (LabelboxClient): Authenticated Labelbox client instance
       data_filler (dict): Dictionary containing two functions:
           - 'create_tensors': callable(ds) -> None
               Creates necessary tensors in the dataset
           - 'fill_data': callable(ds, idx, frame_num, frame) -> None
               Fills dataset with processed frame data
       deeplake_token (str, optional): Authentication token for Deeplake cloud storage.
       overwrite (bool, optional): Whether to overwrite existing dataset. Defaults to False
       lb_ontology (Ontology, optional): Labelbox ontology to connect to project. Defaults to None
       lb_batch_priority (int, optional): Priority for Labelbox batches. Defaults to 5
       lb_dataset_name (str, optional): Custom name for Labelbox dataset.
           Defaults to deeplake_ds_path basename + '_from_deeplake'

    Returns:
       Dataset: Created Deeplake dataset containing processed video frames and metadata for Labelbox project
    """
    ds = deeplake.empty(deeplake_ds_path, token=deeplake_token, overwrite=overwrite)

    data_filler['create_tensors'](ds)

    for idx, video_path in enumerate(video_paths):
        for frame_num, frame in frame_generator_(video_path):
            data_filler['fill_data'](ds, idx, frame_num, frame)

    if lb_dataset_name is None:
        lb_dataset_name = os.path.basename(deeplake_ds_path) + '_from_deeplake'

    lb_ds = lb_client.create_dataset(name=lb_dataset_name)
    task = lb_ds.create_data_rows(video_paths)
    task.wait_till_done()

    # Create a new project
    project = lb_client.create_project(
        name=os.path.basename(deeplake_ds_path),
        media_type=lb.MediaType.Video           
    )

    ds.info['labelbox_meta'] = {
        'project_id': project.uid,
        'type': 'video',
        'sources': video_paths
    }

    task = project.create_batches_from_dataset(
        name_prefix=lb_dataset_name,
        dataset_id=lb_ds.uid,
        priority=lb_batch_priority
    )

    if task.errors():
        raise Exception(f"Error creating batches: {task.errors()}")
    
    if lb_ontology:
        project.connect_ontology(lb_ontology)

    ds.commit()
    
    print(ds.summary())

    return ds

def create_dataset_for_video_annotation(deeplake_ds_path, video_paths, lb_client, deeplake_token=None, overwrite=False, lb_ontology=None, lb_batch_priority=5):
    """
    See create_dataset_for_video_annotation_with_custom_data_filler for complete documentation.

    The only difference is this function uses default tensor creation and data filling functions:
    - create_tensors_default_: Creates default tensor structure
    - fill_data_default_: Fills tensors with default processing
    """
    return create_dataset_for_video_annotation_with_custom_data_filler(deeplake_ds_path, video_paths, lb_client, data_filler={'create_tensors': create_tensors_default_, 'fill_data': fill_data_default_}, deeplake_token=deeplake_token, lb_ontology=lb_ontology, lb_batch_priority=lb_batch_priority, overwrite=overwrite)

def create_dataset_from_video_annotation_project_with_custom_data_filler(deeplake_ds_path, project_id, lb_client, lb_api_key, data_filler, deeplake_token=None, overwrite=False):
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
           - 'fill_data': callable(ds, idx, frame_num, frame) -> None
               Fills dataset with processed frame data
       deeplake_token (str, optional): Authentication token for Deeplake cloud storage.
           Required if using hub:// path. Defaults to None
       overwrite (bool, optional): Whether to overwrite existing dataset. Defaults to False

    Returns:
       Dataset: Created Deeplake dataset containing processed video frames and Labelbox metadata.
       Returns empty dataset if no data found in project.

    Notes:
        - The function does not fetch the annotations from Labelbox, only the video frames. After creating the dataset, use the converter to apply annotations.
    """
    ds = deeplake.empty(deeplake_ds_path, overwrite=overwrite, token=deeplake_token)
    data_filler['create_tensors'](ds)

    proj = labelbox_get_project_json_with_id_(lb_client, project_id)
    if len(proj) == 0:
        print("no data")
        return ds
    
    if not validate_project_creation_data_(proj, project_id, 'video'):
        raise Exception("Data validation failed")

    video_files = []

    for idx, p in enumerate(proj):
        video_url = p["data_row"]["row_data"]
        for frame_num, frame in frame_generator_(video_url, f'Bearer {lb_api_key}'):
            data_filler['fill_data'](ds, idx, frame_num, frame)
        
        video_files.append(p['data_row']['external_id'])

    ds.info['labelbox_meta'] = {
        'project_id': project_id,
        'type': 'video',
        'sources': video_files
    }

    ds.commit()

    return ds

def create_dataset_from_video_annotation_project(deeplake_ds_path, project_id, lb_client, lb_api_key, deeplake_token=None, overwrite=False):
    """
    See create_dataset_from_video_annotation_project_with_custom_data_filler for complete documentation.

    The only difference is this function uses default tensor creation and data filling functions:
    - create_tensors_default_: Creates default tensor structure
    - fill_data_default_: Fills tensors with default processing
    """
    return create_dataset_from_video_annotation_project_with_custom_data_filler(deeplake_ds_path, project_id, lb_client, lb_api_key, data_filler={'create_tensors': create_tensors_default_, 'fill_data': fill_data_default_}, deeplake_token=deeplake_token, overwrite=overwrite)
