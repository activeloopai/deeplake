import deeplake
import os
import labelbox as lb
from deeplake.integrations.labelbox.labelbox_utils import *
from deeplake.integrations.labelbox.labelbox_converter import labelbox_type_converter
from deeplake.integrations.labelbox.v3_converters import *

def converter_for_video_project_with_id(project_id, client, deeplake_ds_loader, lb_api_key):
    project_json = labelbox_get_project_json_with_id_(client, project_id)

    if len(project_json) == 0:
        print("no data")
        return None
    
    ds_name = project_json[0]["projects"][project_id]['name']
    deeplake_dataset = deeplake_ds_loader(ds_name)

    print("validating project data with id", project_id)
    if not validate_project_data_(project_json, deeplake_dataset, project_id, 'video'):
        raise Exception("Data validation failed")
    
    print("project data is valid")

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
    return labelbox_type_converter(ontology, converters, project_json, project_id, deeplake_dataset, {'ds': deeplake_dataset, 'lb_api_key': lb_api_key})

def create_dataset_for_video_annotation_with_custom_data_filler(deeplake_ds_path, video_paths, lb_client, data_filler, overwrite=False, lb_ontology=None, lb_batch_priority=5, lb_dataset_name=None):
    ds = deeplake.empty(deeplake_ds_path, overwrite=overwrite)

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

    ds.info['labelbox_video_sources'] = video_paths
    ds.info['labelbox_project_id'] = project.uid
    ds.info['labelbox_dataset_id'] = lb_ds.uid

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

def create_dataset_for_video_annotation(deeplake_ds_path, video_paths, lb_client, overwrite=False, lb_ontology=None, lb_batch_priority=5):
    return create_dataset_for_video_annotation_with_custom_data_filler(deeplake_ds_path, video_paths, lb_client, data_filler={'create_tensors': create_tensors_default_, 'fill_data': fill_data_default_}, lb_ontology=lb_ontology, lb_batch_priority=lb_batch_priority, overwrite=overwrite)

def create_dataset_from_video_annotation_project_with_custom_data_filler(deeplake_ds_path, project_id, lb_client, lb_api_key, data_filler, overwrite=False):
    ds = deeplake.empty(deeplake_ds_path, overwrite=overwrite)
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

    ds.info['labelbox_video_sources'] = video_files
    ds.info['labelbox_project_id'] = project_id
    ds.info['labelbox_dataset_id'] = 'unknown'

    ds.commit()

    return ds

def create_dataset_from_video_annotation_project(deeplake_ds_path, project_id, lb_client, lb_api_key, overwrite=False):
    return create_dataset_from_video_annotation_project_with_custom_data_filler(deeplake_ds_path, project_id, lb_client, lb_api_key, data_filler={'create_tensors': create_tensors_default_, 'fill_data': fill_data_default_}, overwrite=overwrite)
