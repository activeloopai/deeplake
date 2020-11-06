# Features

## Intro

Hub features:

- Define the structure, shapes, dtypes of the final Dataset
- Add additional meta information(image channels, class names, etc.)
- Use special serialization/deserialization methods


## Available Features

- **Primitive**: wrapper to the numpy primitive data types like int32, float64, etc...
- **Tensor**: np-array like structure that contains any type of elements (Primitive and non-Primitive). 
- **Image**: array representation of image of arbitrary shape and primitive data type. Default encoding format - `png` (`jpeg` is also supported)
- **ClassLabel**: integer representation of feature labels. Can be constructed from number of labels, label names or a text file with asingle label name in each line
- **Mask**: array representation of binary mask
- **Segmentation**: segmentation array. Also constructs ClassLabel feature connector to support segmentation classes.
- **BBox**: bounding box coordinates array with shape (4, )

If a feature has a dynamic shape, `max_shape` argument should be provided representing the maximum possible number of elements in each axis of the feature.

Argument `chunks` describes how to split tensor dimensions into chunks (files) to store them efficiently. If not chosen, it will be automatically detected how to split the information into chunks.
## Usage

```python

from hub.features import ClassLabel, Image, Mask
from hub.features import Tensor, BBox, Segmentation

schema = {
    "image": Image(shape=(244, 244, 3), dtype='uint16'),
    "label": ClassLabel(num_classes=10),
    "mask": Mask(shape=(244, 244, 1)),
    "bbox": BBox()
    "segmentation": Segmentation(shape=(244, 244, 1), dtype='uint8', 
                                names=['label_1', 'label_2', ...])
    "other": Tensor(shape=(None, None),
            dtype="int32",
            max_shape=(100, 100),
            chunks=(100, 100, 100))
}
```

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
