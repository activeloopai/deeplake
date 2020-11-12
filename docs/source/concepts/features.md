# Features

## Overview

Hub features:

- Define the structure, shapes, dtypes of the final Dataset
- Add additional meta information(image channels, class names, etc.)
- Use special serialization/deserialization methods



## Available Features

### Primitive 

Wrapper to the numpy primitive data types like int32, float64, etc...

```python
from hub.features import Primitive

schema = { "scalar": Primitive(dtype="float32") }
```

### Tensor

Np-array like structure that contains any type of elements (Primitive and non-Primitive).

```python
from hub.features import Tensor

schema = {"tensor_1": Tensor((100, 200), "int32"),
          "tensor_2": Tensor((100, 400), "int64", chunks=(6, 50, 200)) }
```

### Image

Array representation of image of arbitrary shape and primitive data type. 

Default encoding format - `png` (`jpeg` is also supported).

```python
from hub.features import Image

schema = {"image": Image(shape=(None, None),
                         dtype="int32",
                         max_shape=(100, 100)
          ) }
```

### ClassLabel

Integer representation of feature labels. Can be constructed from number of labels, label names or a text file with a single label name in each line.

```python
from hub.features import ClassLabel

schema = {"class_label_1": ClassLabel(num_classes=10),
          "class_label_2": ClassLabel(names=['class1', 'class2', 'class3', ...]),
          "class_label_3": ClassLabel(names_file='/path/to/file/with/names')
          ) }
```

### Mask 

Array representation of binary mask. The shape of mask should have format: (height, width, 1).

```python
from hub.features import Image

schema = {"mask": Mask(shape=(244, 244, 1))}
```

### Segmentation

Segmentation array. Also constructs ClassLabel feature connector to support segmentation classes. 

The shape of segmentation mask should have format: (height, width, 1).

```python
from hub import Segmentation

schema = {"segmentation": Segmentation(shape=(244, 244, 1), dtype='uint8', 
                                       names=['label_1', 'label_2', ...])}
```


### BBox

Bounding box coordinates with shape (4, ).

```python
from hub import Segmentation

schema = {"bbox": BBox()}
```

## Arguments

If a feature has a dynamic shape, `max_shape` argument should be provided representing the maximum possible number of elements in each axis of the feature.

Argument `chunks` describes how to split tensor dimensions into chunks (files) to store them efficiently. If not chosen, it will be automatically detected how to split the information into chunks.



## API
```eval_rst
.. autoclass:: hub.features.audio.Audio
   :members:
   :no-undoc-members:
   :private-members:
   :special-members:
.. autoclass:: hub.features.bbox.BBox
   :members:
   :no-undoc-members:
   :private-members:
   :special-members:   
.. autoclass:: hub.features.class_label.ClassLabel
   :members:
   :no-undoc-members:
   :private-members:
   :special-members: 
.. autoclass:: hub.features.image.Image
   :members:
   :no-undoc-members:
   :private-members:
   :special-members:
.. automodule:: hub.features.features
   :members:
   :private-members:
   :special-members:
.. autoclass:: hub.features.mask.Mask
   :members:
   :no-undoc-members:
   :private-members:
   :special-members:
.. autoclass:: hub.features.polygon.Polygon
   :members:
   :no-undoc-members:
   :private-members:
   :special-members:
.. autoclass:: hub.features.segmentation.Segmentation
   :members:
   :no-undoc-members:
   :private-members:
   :special-members:
.. autoclass:: hub.features.sequence.Sequence
   :members:
   :no-undoc-members:
   :private-members:
   :special-members:
.. autoclass:: hub.features.video.Video
   :members:
   :no-undoc-members:
   :private-members:
   :special-members:
```
