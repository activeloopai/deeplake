Htypes
======

.. currentmodule:: hub.htype

.. role:: blue


Htype is the class of a tensor: image, bounding box, generic tensor, etc.

The htype of a tensor can be specified at its creation

>>> ds.create_tensor("my_tensor", htype="...")

If not specified, the tensor's htype defaults to "generic".

Specifying an htype allows for strict settings and error handling, and it is critical for increasing the performance of hub datasets containing rich data such as images and videos.

Supported htypes and their respective defaults are:

+----------------------------+-----------+---------------+
| HTYPE                      |  DTYPE    |  COMPRESSION  |
+============================+===========+===============+
| generic                    |  None     |  None         |
+----------------------------+-----------+---------------+
| image                      |  uint8    |  None         |
+----------------------------+-----------+---------------+
| image.rgb                  |  uint8    |  None         |
+----------------------------+-----------+---------------+
| image.gray                 |  uint8    |  None         |
+----------------------------+-----------+---------------+
| class_label                |  uint32   |  None         |
+----------------------------+-----------+---------------+
| bbox                       |  float32  |  None         |
+----------------------------+-----------+---------------+
| video                      |  uint8    |  None         |
+----------------------------+-----------+---------------+
| instance_label             | uint32    |  None         |
+----------------------------+-----------+---------------+
| point                      | int32     |  None         |
+----------------------------+-----------+---------------+
| binary_mask                |  bool     |  None         |
+----------------------------+-----------+---------------+
| segment_mask               |  uint32   |  None         |
+----------------------------+-----------+---------------+
| keypoints_coco             |  int32    |  None         |
+----------------------------+-----------+---------------+
| point                      |  int32    |  None         |
+----------------------------+-----------+---------------+
| audio                      |  float64  |  None         |
+----------------------------+-----------+---------------+
| text                       |  str      |  None         |
+----------------------------+-----------+---------------+
| json                       |  Any      |  None         |
+----------------------------+-----------+---------------+
| list                       |  List     |  None         |
+----------------------------+-----------+---------------+
| dicom                      |  None     |  dcm          |
+----------------------------+-----------+---------------+
| link                       |  str      |  None         |
+----------------------------+-----------+---------------+
| sequence                   |  None     |  None         |
+----------------------------+-----------+---------------+
| point_cloud                | None      | las           |
+----------------------------+-----------+---------------+

Image Htype
~~~~~~~~~~~

- Tensor dimensions: ``(# images, height, width, # channels)`` or ``(# images, height, width)``.

Images can be stored in hub as compressed bytes or as raw arrays. Due to the high compression ratio for most image 
formats, it is highly recommended to store compressed images using the ``sample_compression`` input to the create_tensor method.

:blue:`Creating an image tensor`
--------------------------------

An image tensor can be created using

>>> ds.create_tensor("images", htype="image", sample_compression="jpg")

OR

>>> ds.create_tensor("images", htype="image", chunk_compression="jpg")

- Optional args:
    - :dtype: Defaults to ``uint8``.
- Supported compressions:

>>> [None, "bmp", "dib", "gif", "ico", "jpeg", "jpeg2000", "pcx", "png", "ppm", "sgi", "tga", "tiff", 
... "webp", "wmf", "xbm", "eps", "fli", "im", "msp", "mpo"]

:blue:`Appending image samples`
-------------------------------

- Image samples can be of type ``np.ndarray`` or hub :class:`~hub.core.sample.Sample`. A hub image sample can be created using :meth:`hub.read`.
- Input sample shape: ``(height, width, # channels)`` or ``(height, width)``

:blue:`Examples`

Appending pixel data with array

>>> ds.images.append(np.zeros((5, 5, 3), dtype=np.uint8))

Appening hub image sample

>>> ds.images.append(hub.read("images/0001.jpg"))

You can append multiple samples at the same time using :meth:`~hub.core.tensor.Tensor.extend`.

>>> ds.images.extend([hub.read(f"images/000{i}.jpg") for i in range(10)])

.. note::
    If the compression format of the input sample does not match the ``sample_compression`` of the tensor,
    hub will decompress and recompress the image for storage, which may significantly slow down the upload
    process. The upload process is fastest when the image compression matches the ``sample_compression``.

:blue:`image.rgb and image.gray htypes`
---------------------------------------

``image.rgb`` and ``image.gray`` htypes can be used to force your samples to be of RGB or grayscale type.
i.e., if RGB images are appened to an  ``image.gray`` tensor, hub will convert them to grayscale and if grayscale images
are appended to an ``image.rgb`` tensor, hub will convert them to RGB format.

image.rgb and image.gray tensors can be created using

>>> ds.create_tensor("rgb_images", htype="image.rgb", sample_compression="...")

>>> ds.create_tensor("gray_images", htype="image.gray", sample_compression="...")

Video Htype
~~~~~~~~~~~

- Tensor dimensions: ``(# videos, # frames, height, width, # channels)`` or ``(# videos, # frames, height, width)``

:blue:`Creating a video tensor`
-------------------------------

A video tensor can be created using

>>> ds.create_tensor("videos", htype="video", sample_compression="mp4")

- Optional args:
    - :dtype: Defaults to ``uint8``.
- Supported compressions:

>>> [None, "mp4", "mkv", "avi"]

:blue:`Appending video samples`
-------------------------------

- Video samples can be of type ``np.ndarray`` or :class:`~hub.core.sample.Sample` which is returned by :meth:`hub.read`.
- Hub does not support compression of raw video frames. Therefore, array of raw frames can only be appended to tensors with
  ``None`` compression.
- Recompression of samples read with :meth:`hub.read <hub.read>` is also not supported.
- Input sample shape: ``(# frames, height, width, channels)`` or ``(# frames, height, width)``

:blue:`Examples`

Appending hub video sample

>>> ds.videos.append(hub.read("videos/0012.mp4"))

Extending with multiple videos

>>> ds.videos.extend([hub.read(f"videos/00{i}.mp4") for i in range(10)])

Audio Htype
~~~~~~~~~~~

- Tensor dimensions: ``(# audios, # frames, # channels)`` or ``(# audios, # frames)``

:blue:`Creating an audio tensor`
--------------------------------

An audio tensor can be created using

>>> ds.create_tensor("audios", htype="audio", sample_compression="mp3")

- Optional args:
    - :dtype: Defaults to ``float64``.
- Supported compressions:

>>> [None, "mp3", "wav", "flac"]

:blue:`Appending audio samples`
-------------------------------

- Audio samples can be of type ``np.ndarray`` or :class:`~hub.core.sample.Sample` which is returned by :meth:`hub.read`.
- Like videos, Hub does not support compression or recompression of input audio samples. Thus, samples of type ``np.ndarray``
  can only be appended to tensors with ``None`` compression.
- Input sample shape: ``(# frames, # channels)`` or ``(# frames)``

:blue:`Examples`

Appending hub audio sample

>>> ds.audios.append(hub.read("audios/001.mp3"))

Extending with hub audio samples

>>> ds.audios.extend([hub.read(f"videos/00{i}.mp3") for i in range(10)])

Class Label Htype
~~~~~~~~~~~~~~~~~

- Tensor dimensions: ``(# samples, # labels)``

Class labels are stored as numerical values in tensors, which are indices of the list ``tensor.info.class_names``.

:blue:`Creating a class label tensor`
-------------------------------------

A class label tensor can be created using

>>> classes = ["airplanes", "cars", "birds", "cats", "deer", "dogs", "frogs", "horses", "ships", "trucks"]
>>> ds.create_tensor("labels", htype="class_label", class_names=classes, chunk_compression="lz4")

- Optional args:
    - :class_names: This must be a **list of strings**. ``tensor.info.class_names`` will be set to this list.
    - :ref:`sample_compression <sample_compression>` OR :ref:`chunk_compression <chunk_compression>`.
    - :dtype: Defaults to ``uint32``.
- Supported compressions:

>>> ["lz4"]

You can also choose to set the class names after tensor creation.

>>> ds.labels.info.update(class_names = ["airplanes", "cars", "birds", "cats", "deer", "dogs", "frogs", "horses", "ships", "trucks"])

.. note::
    If specifying compression, since the number of labels in one sample will be too low, ``chunk_compression`` 
    would be the better option to use.

:blue:`Appending class labels`
------------------------------

- Class labels can be appended as ``int``, ``str``, ``np.ndarray`` or ``list`` of ``int`` or ``str``.
- In case of strings, ``tensor.info.class_names`` is updated automatically.

:blue:`Examples`

Appending index

>>> ds.labels.append(0)
>>> ds.labels.append(np.zeros((5,), dtype=np.uint32))

Extending with list of indices

>>> ds.labels.extend([[0, 1, 2], [1, 3]])

Appending text labels

>>> ds.labels.append(["cars", "airplanes"])

Segmentation Mask Htype
~~~~~~~~~~~~~~~~~~~~~~~

- Tensor dimensions: ``(# samples, height, width)``

Segmentation masks are 2D representations of class labels where the numerical label data is encoded in an array of
same shape as the image. The numerical values are indices of the list ``tensor.info.class_names``.

:blue:`Creating a segment_mask tensor`
--------------------------------------

A segment_mask tensor can be created using

>>> classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle"]
>>> ds.create_tensor("masks", htype="segment_mask", class_names=classes, sample_compression="lz4")

- Optional args:
    - :class_names: This must be a **list of strings**. ``tensor.info.class_names`` will be set to this list.
    - :ref:`sample_compression <sample_compression>` OR :ref:`chunk_compression <chunk_compression>`
    - :dtype: Defaults to ``uint32``.
- Supported compressions:

>>> ["lz4"]

You can also choose to set the class names after tensor creation.

>>> ds.labels.info.update(class_names = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle"])

.. note::
    Since segmentation masks often contain large amounts of data, it is recommended to compress them
    using ``lz4``.

:blue:`Appending segmentation masks`
------------------------------------

- Segmentation masks can be appended as ``np.ndarray`` s.
- Input sample shape: ``(height, width)``

:blue:`Examples`

>>> ds.masks.append(np.zeros((512, 512)))

.. note::
    Since each pixel can only be labeled once, segmentation masks are not appropriate for datasets where objects
    might overlap, or where multiple objects within the same class must be distinguished. For these use cases,
    please use `htype = "binary_mask" <Binary Mask Htype>`.

Binary Mask Htype
~~~~~~~~~~~~~~~~~

- Tensor dimensions: ``(# samples, height, width, # objects in a sample)``

Binary masks are similar to segmentation masks, except that each object is represented by a channel in the mask.
Each channel in the mask encodes values for a single object. A pixel in a mask channel should have a value of 1
if the pixel of the image belongs to this object and 0 otherwise. The labels corresponding to the channels should
be stored in an adjacent tensor of htype ``class_label``, in which the number of labels at a given index is equal
to the number of objects (number of channels) in the binary mask.

:blue:`Creating a binary_mask tensor`
-------------------------------------

A binary_mask tensor can be created using

>>> ds.create_tensor("masks", htype="binary_mask", sample_compression="lz4")

- Optional args:
    - :ref:`sample_compression <sample_compression>` OR :ref:`chunk_compression <chunk_compression>`
    - :dtype: Defaults to ``bool``.
- Supported compressions:

>>> ["lz4"]

.. note::
    Since segmentation masks often contain large amounts of data, it is recommended to compress them
    using ``lz4``.

:blue:`Appending binary masks`
------------------------------

- Binary masks can be appended as ``np.ndarray`` s.
- Input sample shape: ``(height, width, # objects in the sample)``

:blue:`Examples`

Appending a binary mask with 5 objects

>>> ds.masks.append(np.zeros((512, 512, 5), dtype="bool"))
>>> ds.labels.append(["aeroplane", "aeroplane", "bottle", "bottle", "bird"])

COCO Keypoints Htype
~~~~~~~~~~~~~~~~~~~~

- Tensor Dimensions: ``(# samples, 3 x # keypoints, # objects in a sample)``

:ref:`COCO keypoints <https://cocodataset.org/#format-data>` are a convention for storing points of interest
in an image. Each keypoint consists of 3 values: ``x - coordinate``, ``y - coordinate`` and ``v - visibility``.
A set of ``K`` keypoints of an object is represented as:

[x\ :sub:`1`, y\ :sub:`1`, v\ :sub:`1`, x\ :sub:`2`, y\ :sub:`2`, v\ :sub:`2`, ..., x\ :sub:`k`, y\ :sub:`k`, v\ :sub:`k`]

The visibility ``v`` can be one of three values:

:0: keypoint not in image.
:1: keypoint in image but not visible.
:2: keypoint in image and visible.

:blue:`Creating a keypoints_coco`
---------------------------------

A keypoints_coco tensor can be created using

>>> ds.create_tensor("keypoints", htype="keypoints_coco", keypoints=["knee", "elbow", "head"], connections=[[0, 1], [1, 2]])

- Optional args:
    - :keypoints: List of strings describing the ``i`` th keypoint. ``tensor.info.keypoints`` will be set to this list.
    - :connections: List of strings describing which points should be connected by lines in the visualizer


Sequence htype
~~~~~~~~~~~~~~

- A special meta htype for tensors where each sample is a sequence. The items in the sequence are samples of another htype.
- It is a wrapper htype that can wrap other htypes like ``sequence[image]``, ``sequence[video]``, ``sequence[text]``, etc.

:blue:`Examples`

>>> ds.create_tensor("seq", htype="sequence")
>>> ds.seq.append([1, 2, 3])
>>> ds.seq.append([4, 5, 6])
>>> ds.seq.numpy()
array([[[1],
        [2],
        [3]],
       [[4],
        [5],
        [6]]])

>>> ds.create_tensor("image_seq", htype="sequence[image]", sample_compression="jpg")
>>> ds.image_seq.append([hub.read("img01.jpg"), hub.read("img02.jpg")])

Link htype
~~~~~~~~~~

- Link htype is a special meta htype that allows linking of external data (files) to the dataset, without storing the data in the dataset itself.
- Moreover, there can be variations in this htype, such as ``link[image]``, ``link[video]``, ``link[audio]``, etc. that would enable the activeloop visualizer to correctly display the data.
- No data is actually loaded until you try to read the sample from a dataset.
- There are a few exceptions to this:-
    - If ``verify=True`` was specified during ``create_tensor`` of the tensor to which this is being added, some metadata is read to verify the integrity of the sample.
    - If ``create_shape_tensor=True`` was specified during ``create_tensor`` of the tensor to which this is being added, the shape of the sample is read.
    - If ``create_sample_info_tensor=True`` was specified during ``create_tensor`` of the tensor to which this is being added, the sample info is read.

.. _linked_sample_examples:

:blue:`Examples`

>>> ds = hub.dataset("......")

Add the names of the creds you want to use (not needed for http/local urls)

>>> ds.add_creds_key("MY_S3_KEY")
>>> ds.add_creds_key("GCS_KEY")

Populate the names added with creds dictionary
These creds are only present temporarily and will have to be repopulated on every reload

>>> ds.populate_creds("MY_S3_KEY", {})   # add creds here
>>> ds.populate_creds("GCS_KEY", {})    # add creds here

Create a tensor that can contain links

>>> ds.create_tensor("img", htype="link[image]", verify=True, create_shape_tensor=False, create_sample_info_tensor=False)

Populate the tensor with links

>>> ds.img.append(hub.link("s3://abc/def.jpeg", creds_key="MY_S3_KEY"))
>>> ds.img.append(hub.link("gcs://ghi/jkl.png", creds_key="GCS_KEY"))
>>> ds.img.append(hub.link("https://picsum.photos/200/300")) # http path doesn’t need creds
>>> ds.img.append(hub.link("./path/to/cat.jpeg")) # local path doesn’t need creds
>>> ds.img.append(hub.link("s3://abc/def.jpeg"))  # this will throw an exception as cloud paths always need creds_key
>>> ds.img.append(hub.link("s3://abc/def.jpeg", creds_key="ENV"))  # this will use creds from environment

Accessing the data

>>> for i in range(5):
...     ds.img[i].numpy()
...

Updating a sample

>>> ds.img[0] = hub.link("./data/cat.jpeg")
