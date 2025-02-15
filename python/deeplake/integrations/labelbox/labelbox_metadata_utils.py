import os


def get_video_name_from_video_project_(project, ctx):
    if "data_row" not in project:
        return None
    if "external_id" in project["data_row"]:
        return os.path.splitext(os.path.basename(project["data_row"]["external_id"]))[0]
    if "row_data" in project["data_row"]:
        return os.path.splitext(os.path.basename(project["data_row"]["row_data"]))[0]
    return None


def get_data_row_id_from_video_project_(project, ctx):
    try:
        return project["data_row"]["id"]
    except:
        return None


def get_data_row_url_from_video_project_(project, ctx):
    try:
        return project["data_row"]["row_data"]
    except:
        return None


def get_label_creator_from_video_project_(project, ctx):
    try:
        return project["projects"][ctx["project_id"]]["labels"][0]["label_details"][
            "created_by"
        ]
    except:
        return None


def get_frame_rate_from_video_project_(project, ctx):
    try:
        return project["media_attributes"]["frame_rate"]
    except:
        return None


def get_frame_count_from_video_project_(project, ctx):
    try:
        return project["media_attributes"]["frame_count"]
    except:
        return None


def get_width_from_video_project_(project, ctx):
    try:
        return project["media_attributes"]["width"]
    except:
        return None


def get_height_from_video_project_(project, ctx):
    try:
        return project["media_attributes"]["height"]
    except:
        return None


def get_ontology_id_from_video_project_(project, ctx):
    try:
        return project["projects"][ctx["project_id"]]["project_details"]["ontology_id"]
    except:
        return None


def get_project_name_from_video_project_(project, ctx):
    try:
        return project["projects"][ctx["project_id"]]["name"]
    except:
        return None


def get_dataset_name_from_video_project_(project, ctx):
    try:
        return project["data_row"]["details"]["dataset_name"]
    except:
        return None


def get_dataset_id_from_video_project_(project, ctx):
    try:
        return project["data_row"]["details"]["dataset_id"]
    except:
        return None


def get_global_key_from_video_project_(project, ctx):
    try:
        return project["data_row"]["global_key"]
    except:
        return None
