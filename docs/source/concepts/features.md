# Schema

## Overview

Hub Schema:

- Define the structure, shapes, dtypes of the final Dataset
- Add additional meta information(image channels, class names, etc.)
- Use special serialization/deserialization methods



## Available Schemas

### Primitive 

Wrapper to the numpy primitive data types like int32, float64, etc...

```python
from hub.schema import Primitive

schema = { "scalar": Primitive(dtype="float32") }
```

### Tensor

Np-array like structure that contains any type of elements (Primitive and non-Primitive). Hub Tensors can't be visualized at [app.activeloop.ai](https://app.activeloop.ai).

```python
from hub.schema import Tensor

schema = {"tensor_1": Tensor((None, None), max_shape=(200, 200), "int32"),
          "tensor_2": Tensor((100, 400), "int64", chunks=(6, 50, 200)) }
```

### Image

Array representation of image of arbitrary shape and primitive data type. 

Default encoding format - `png` (`jpeg` is also supported).

```python
from hub.schema import Image

schema = {"image": Image(shape=(None, None),
                         dtype="int32",
                         max_shape=(100, 100)
          ) }
```

### ClassLabel

Integer representation of feature labels. Can be constructed from number of labels, label names or a text file with a single label name in each line.

```python
from hub.schema import ClassLabel

schema = {"class_label_1": ClassLabel(num_classes=10),
          "class_label_2": ClassLabel(names=['class1', 'class2', 'class3', ...]),
          "class_label_3": ClassLabel(names_file='/path/to/file/with/names')
          ) }
```

### Mask 

Array representation of binary mask. The shape of mask should have format: (height, width, 1).

```python
from hub.schema import Image

schema = {"mask": Mask(shape=(244, 244, 1))}
```

### Segmentation

Segmentation array. Also constructs ClassLabel feature connector to support segmentation classes. 

The shape of segmentation mask should have format: (height, width, 1).

```python
from hub.schema import Segmentation

schema = {"segmentation": Segmentation(shape=(244, 244, 1), dtype='uint8', 
                                       names=['label_1', 'label_2', ...])}
```


### BBox

Bounding box coordinates with shape (4, ).

```python
from hub.schema import BBox

schema = {"bbox": BBox()}
```

### Audio

Hub schema for audio files. A file can have any format ffmpeg understands. If `file_format` parameter isn't provided 
will attempt to infer it from the file extension. Also, `sample_rate` parameter can be added as additional metadata. User can access through info.schema[‘audio’].sample_rate.

```python
from hub.schema import Audio

schema = {'audio': Audio(shape=(300,)}
```

### Video

Video format support. 
Accepts as input a 4 dimensional uint8 array representing a video.
The video is stored as a sequence of encoded images. `encoding_format` can be any format supported by Image.
```python
from hub.schema import Video

schema = {'video': Video(shape=(20, None, None, 3), max_shape=(20, 1200, 1200, 3))}
```

### Text

Autoconverts given string into its integer(int64) representation.
```python
from hub.schema import Text

schema = {'text': Text(shape=(None, ), max_shape=(20, ))}
```

### Sequence

Correspond to sequence of `schema.HubSchema`.
At generation time, a list for each of the sequence element is given. The output
of `Dataset` will batch all the elements of the sequence together.
If the length of the sequence is static and known in advance, it should be
specified in the constructor using the `length` param.

```python
from hub.schema import Sequence, BBox

schema = {'sequence': Sequence(shape=(10, ), dtype=BBox)}
```

## Arguments

If a schema has a dynamic shape, `max_shape` argument should be provided representing the maximum possible number of elements in each axis of the feature.

Argument `chunks` describes how to split tensor dimensions into chunks (files) to store them efficiently. If not chosen, it will be automatically detected how to split the information into chunks.



## API
```eval_rst
.. autoclass:: hub.schema.audio.Audio
   :members:
   :no-undoc-members:
   :private-members:
   :special-members:
.. autoclass:: hub.schema.bbox.BBox
   :members:
   :no-undoc-members:
   :private-members:
   :special-members:   
.. autoclass:: hub.schema.class_label.ClassLabel
   :members:
   :no-undoc-members:
   :private-members:
   :special-members: 
.. autoclass:: hub.schema.image.Image
   :members:
   :no-undoc-members:
   :private-members:
   :special-members:
.. automodule:: hub.schema.features
   :members:
   :private-members:
   :special-members:
.. autoclass:: hub.schema.mask.Mask
   :members:
   :no-undoc-members:
   :private-members:
   :special-members:
.. autoclass:: hub.schema.polygon.Polygon
   :members:
   :no-undoc-members:
   :private-members:
   :special-members:
.. autoclass:: hub.schema.segmentation.Segmentation
   :members:
   :no-undoc-members:
   :private-members:
   :special-members:
.. autoclass:: hub.schema.sequence.Sequence
   :members:
   :no-undoc-members:
   :private-members:
   :special-members:
.. autoclass:: hub.schema.text.Text
   :members:
   :no-undoc-members:
   :private-members:
   :special-members:
.. autoclass:: hub.schema.video.Video
   :members:
   :no-undoc-members:
   :private-members:
   :special-members:
```
