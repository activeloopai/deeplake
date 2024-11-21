import labelbox as lb
import os
import tempfile

from deeplake.integrations.labelbox import create_dataset_from_video_annotation_project, converter_for_video_project_with_id

def test_labelbox():
    with tempfile.TemporaryDirectory() as temp_dir:
        ds_path = os.path.join(temp_dir, 'labelbox_ds')
        API_KEY = os.environ['LABELBOX_API_TOKEN']
        client = lb.Client(api_key=API_KEY)

        project_id = 'cm3rxazmh00nk07xx2xe2973u'
        ds = create_dataset_from_video_annotation_project(ds_path, project_id, client, API_KEY, overwrite=True)
        def ds_provider(p):
            try:
                ds.delete_branch('labelbox')
            except:
                pass
            ds.checkout('labelbox', create=True)
            return ds
        converter = converter_for_video_project_with_id(project_id, client, ds_provider, API_KEY)
        ds = converter.dataset_with_applied_annotations()

        ds.commit('add labelbox annotations')

        print(ds.summary())
