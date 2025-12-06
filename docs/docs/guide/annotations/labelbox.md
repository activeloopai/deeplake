---
seo_title: "Labelbox Integration | Annotations"
description: "Labelbox integration for annotations."
---
# Labelbox Integration
This document describes how to create Deep Lake datasets from [Labelbox](https://labelbox.com/) annotations. The API also allows you to update the dataset with new annotations.

## Prerequisites

```bash
python -m pip install labelbox
```

## Supported Labelbox Ontologies

- [Video Ontology](#video-ontology)
- [Image Ontology](#image-ontology)

### Video Ontology

For video ontolgy, python `av` library is used to extract frames from videos.

```bash
python -m pip install av
```

### Uploading videos for annotation to Labelbox

Deeplake supports uploading videos to Labelbox using the [Labelbox API](https://docs.labelbox.com/reference/getting-started).

<!-- test-context
```python
LABELBOX_API_KEY = 'LABELBOX_API_KEY'

class labelbox_mock:
    def __init__(self):
        pass
    def Client(self, *args, **kwargs):
        global create_labelbox_annotation_project
        create_labelbox_annotation_project = lambda *args, **kwargs: None
        return lb_client_mock()
        return self

labelbox = labelbox_mock()

class lb_client_mock:
    def get_ontology(self, ontology_id):
        return ontology_id
```
-->

```python
from deeplake.integrations import create_labelbox_annotation_project

client = labelbox.Client(api_key=LABELBOX_API_KEY)

files = [] # list of video urls, can be all local or all remote.

# connect the ontology to the project
ontology = client.get_ontology('ontology_id_from_labelbox')

# create annotation project in labelbox
create_labelbox_annotation_project(files, 'dataset-for-deeplake-tests', 'project-for-deeplake-tests', LABELBOX_API_KEY, lb_ontology=ontology)
```


### Creating a dataset from an annotated Labelbox project

To create a dataset from an annotated Labelbox project, you can use the following code:

<!-- test-context
```python
class dataset_mock:
    def __init__(self):
        pass
    def commit(self, *args, **kwargs):
        pass
    def tag(self, *args, **kwargs):
        pass

class converter_mock:
    def __init__(self):
        pass
    def dataset_with_applied_annotations(self):
        return dataset_mock()

def get_project_id():
    global create_dataset_from_video_annotation_project
    global converter_for_video_project_with_id
    create_dataset_from_video_annotation_project = lambda *args, **kwargs: (dataset_mock(), dict())
    converter_for_video_project_with_id = lambda *args, **kwargs: converter_mock()
    return "project_id_from_labelbox"

LABELBOX_API_KEY = 'LABELBOX_API_KEY'
```
-->

```python
from deeplake.integrations import (
    create_dataset_from_video_annotation_project,
    converter_for_video_project_with_id
)

# the path where we want to create the dataset
ds_path = "mem://labelbox_connect_test"

# the project id of the labelbox project that we want to create the dataset from
project_id = get_project_id()

# we pass the url presigner in cases when the videos are in cloud storage (
# for this case azure blob storage) and the videos were added to labelbox with their integrations functionality.
# the default one tries to use labelbox api to get the non public remote urls.
def url_presigner(url):
    sas_token = "<your azure token here>"
    # the second value is the headers that will be added to the request
    return url.partition("?")[0] + "?" + sas_token, {}

# create the dataset, this will extract the frames from the videos and create the dataset.
# the project_json is a json file that contains the project information from labelbox which we can reuse during the labels fetching.
ds, project_json = create_dataset_from_video_annotation_project(
    ds_path,
    project_id,
    LABELBOX_API_KEY,
    url_presigner=url_presigner,
)

# commit the dataset
ds.commit("create dataset")

# define the dataset provider
# the dataset provider can be used to update do some other operations on the dataset, before the annotations are applied.
def ds_provider(p):
    # we need to keep p (labelbox project name) with the ds path in case we need to refetch labeles.
    # this step is completely optional, we just need to be able to load the correct dataset for refetching labels.
    # our refetching example will be using the same mapping to retrieve the ds_path from the project name.
    with open(f'{project_id}_mapping.json', 'w') as f:
        import json
        json.dump({p: ds_path}, f)
    try:
        # create a new branch , where all the annotations will be stored.
        # the main branch will be intact.
        ds.branch("labelbox")
    except:
        pass
    return ds.branches["labelbox"].open()

# create the converter
converter = converter_for_video_project_with_id(
    project_id,
    ds_provider,
    LABELBOX_API_KEY,
    group_mapping={"raster-segmentation": "mask"},
    project_json=project_json,
)

# generate the annotations
ds = converter.dataset_with_applied_annotations()

# commit the annotations to the dataset
ds.commit("add labelbox annotations")
```

### Re-fetching the annotations from Labelbox to the existing dataset

At the moment, the for re-fetching the annotations from Labelbox to the existing dataset is not supported. However it will be supported in the future. In the meantime, you can keep the annotations in a separate dataset. There are only 2 requirements:

- The dataset should have the same length as the dataset that you have created from Labelbox.
- The dataset should have the same `labelbox_meta` metadata as the dataset that you have created from Labelbox.

Then you can UNION the two datasets.


### Image ontology

### Uploading images for annotation to Labelbox

Deeplake supports uploading images to Labelbox using the [Labelbox API](https://docs.labelbox.com/reference/getting-started).

<!-- test-context
```python
LABELBOX_API_KEY = 'LABELBOX_API_KEY'

class labelbox_mock:
    def __init__(self):
        pass
    def Client(self, *args, **kwargs):
        global create_labelbox_annotation_project
        create_labelbox_annotation_project = lambda *args, **kwargs: None
        return lb_client_mock()
        return self

labelbox = labelbox_mock()

class lb_client_mock:
    def get_ontology(self, ontology_id):
        return ontology_id
```
-->

```python
from deeplake.integrations import create_labelbox_annotation_project

client = labelbox.Client(api_key=LABELBOX_API_KEY)

files = [] # list of image urls, can be all local or all remote.

# connect the ontology to the project
ontology = client.get_ontology('ontology_id_from_labelbox')

# create annotation project in labelbox
create_labelbox_annotation_project(files, 'dataset-for-deeplake-tests', 'project-for-deeplake-tests', LABELBOX_API_KEY, lb_ontology=ontology, media_type = "IMAGE")
```


### Creating a dataset from an annotated Labelbox project
To create a dataset from an annotated Labelbox image project, the logic is similar to the video ontology, but you will use the `create_dataset_from_image_annotation_project` function instead. Here is an example:

<!-- test-context
```python
class dataset_mock:
    def __init__(self):
        pass
    def commit(self, *args, **kwargs):
        pass
    def tag(self, *args, **kwargs):
        pass

class converter_mock:
    def __init__(self):
        pass
    def dataset_with_applied_annotations(self):
        return dataset_mock()

def get_project_id():
    global create_dataset_from_image_annotation_project
    global converter_for_image_project_with_id
    create_dataset_from_image_annotation_project = lambda *args, **kwargs: (dataset_mock(), dict())
    converter_for_image_project_with_id = lambda *args, **kwargs: converter_mock()
    return "project_id_from_labelbox"

LABELBOX_API_KEY = 'LABELBOX_API_KEY'
```
-->

```python
import deeplake
from deeplake.integrations import (
    create_dataset_from_image_annotation_project,
    converter_for_image_project_with_id,
)
# the path where we want to create the dataset
ds_path = "mem://labelbox_connect_test"
# the project id of the labelbox project that we want to create the dataset from
project_id = get_project_id()

# create the dataset, this will extract the frames from the videos and create the dataset.
ds, project_json = create_dataset_from_image_annotation_project(
    ds_path,
    project_id,
    LABELBOX_API_KEY,
)

# commit the dataset
ds.commit("create dataset")
# define the dataset provider
# the dataset provider can be used to update do some other operations on the dataset, before the
# annotations are applied.
def ds_provider(p):
    # we need to keep p (labelbox project name) with the ds path in case
    # we need to refetch labeles.
    # this step is completely optional, we just need to be able to load the correct dataset
    # for refetching labels.
    # our refetching example will be using the same mapping to retrieve the ds_path from the project name.
    with open(f'{project_id}_mapping.json', 'w') as f:
        import json
        json.dump({p: ds_path}, f)
    try:            
        # create a new branch , where all the annotations will be stored.
        # the main branch will be intact.
        ds.branch("labelbox")
    except:
        pass
    return ds.branches["labelbox"].open()

# create the converter
converter = converter_for_image_project_with_id(
    project_id,
    ds_provider,
    LABELBOX_API_KEY,
    group_mapping={"raster-segmentation": "mask"},
    project_json=project_json,
)
# generate the annotations
ds = converter.dataset_with_applied_annotations()
# commit the annotations to the dataset
ds.commit("add labelbox annotations")
```