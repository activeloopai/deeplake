Htypes
======

.. currentmodule:: deeplake.htype

.. role:: blue

.. role:: bluebold


Htype is the class of a tensor: image, bounding box, generic tensor, etc.

The htype of a tensor can be specified at its creation

>>> ds.create_tensor("my_tensor", htype="...")

If not specified, the tensor's htype defaults to "generic".

Specifying an htype allows for strict settings and error handling, and it is critical for increasing the performance of Deep Lake datasets containing rich data such as images and videos.

Supported htypes and their respective defaults are:

.. csv-table:: Htype configs
    :file: _static/csv/htypes.csv
    :header-rows: 1

.. _image-htype:

Image Htype
~~~~~~~~~~~

- :bluebold:`Sample dimensions:` ``(height, width, # channels)`` :bluebold:`or` ``(height, width)``.

Images can be stored in Deep Lake as compressed bytes or as raw arrays. Due to the high compression ratio for most image 
formats, it is highly recommended to store compressed images using the ``sample_compression`` input to the create_tensor method.

:blue:`Creating an image tensor`
--------------------------------

An image tensor can be created using

>>> ds.create_tensor("images", htype="image", sample_compression="jpg")

OR

>>> ds.create_tensor("images", htype="image", chunk_compression="jpg")

- Optional args:
    - dtype: Defaults to ``uint8``.
- Supported compressions:

>>> [None, "bmp", "dib", "gif", "ico", "jpeg", "jpeg2000", "pcx", "png", "ppm", "sgi", "tga", "tiff", 
... "webp", "wmf", "xbm", "eps", "fli", "im", "msp", "mpo"]

:blue:`Appending image samples`
-------------------------------

- Image samples can be of type ``np.ndarray`` or Deep Lake :class:`~deeplake.core.sample.Sample` which can be created using :meth:`deeplake.read`.

:bluebold:`Examples`

Appending pixel data with array

>>> ds.images.append(np.zeros((5, 5, 3), dtype=np.uint8))

Appening Deep Lake image sample

>>> ds.images.append(deeplake.read("images/0001.jpg"))

You can append multiple samples at the same time using :meth:`~deeplake.core.tensor.Tensor.extend`.

>>> ds.images.extend([deeplake.read(f"images/000{i}.jpg") for i in range(10)])

.. note::
    If the compression format of the input sample does not match the ``sample_compression`` of the tensor,
    Deep Lake will decompress and recompress the image for storage, which may significantly slow down the upload
    process. The upload process is fastest when the image compression matches the ``sample_compression``.

.. _image-rgb-and-gray:

:blue:`image.rgb and image.gray htypes`
---------------------------------------

``image.rgb`` and ``image.gray`` htypes can be used to force your samples to be of RGB or grayscale type.
i.e., if RGB images are appended to an  ``image.gray`` tensor, Deep Lake will convert them to grayscale and if grayscale images
are appended to an ``image.rgb`` tensor, Deep Lake will convert them to RGB format.

image.rgb and image.gray tensors can be created using

>>> ds.create_tensor("rgb_images", htype="image.rgb", sample_compression="...")

>>> ds.create_tensor("gray_images", htype="image.gray", sample_compression="...")


.. _video-htype:

Video Htype
~~~~~~~~~~~

- :bluebold:`Sample dimensions:` ``(# frames, height, width, # channels)`` :bluebold:`or` ``(# frames, height, width)``

:blue:`Creating a video tensor`
-------------------------------

A video tensor can be created using

>>> ds.create_tensor("videos", htype="video", sample_compression="mp4")

- Optional args:
    - dtype: Defaults to ``uint8``.
- Supported compressions:

>>> [None, "mp4", "mkv", "avi"]

:blue:`Appending video samples`
-------------------------------

- Video samples can be of type ``np.ndarray`` or :class:`~deeplake.core.sample.Sample` which is returned by :meth:`deeplake.read`.
- Deep Lake does not support compression of raw video frames. Therefore, array of raw frames can only be appended to tensors with
  ``None`` compression.
- Recompression of samples read with :meth:`deeplake.read <deeplake.read>` is also not supported.

:bluebold:`Examples`

Appending Deep Lake video sample

>>> ds.videos.append(deeplake.read("videos/0012.mp4"))

Extending with multiple videos

>>> ds.videos.extend([deeplake.read(f"videos/00{i}.mp4") for i in range(10)])

.. _audio-htype:

Audio Htype
~~~~~~~~~~~

- :bluebold:`Sample dimensions:` ``(# samples in audio, # channels)`` :bluebold:`or` ``(# samples in audio,)``

:blue:`Creating an audio tensor`
--------------------------------

An audio tensor can be created using

>>> ds.create_tensor("audios", htype="audio", sample_compression="mp3")

- Optional args:
    - dtype: Defaults to ``float64``.
- Supported compressions:

>>> [None, "mp3", "wav", "flac"]

:blue:`Appending audio samples`
-------------------------------

- Audio samples can be of type ``np.ndarray`` or :class:`~deeplake.core.sample.Sample` which is returned by :meth:`deeplake.read`.
- Like videos, Deep Lake does not support compression or recompression of input audio samples. Thus, samples of type ``np.ndarray``
  can only be appended to tensors with ``None`` compression.

:bluebold:`Examples`

Appending Deep Lake audio sample

>>> ds.audios.append(deeplake.read("audios/001.mp3"))

Extending with Deep Lake audio samples

>>> ds.audios.extend([deeplake.read(f"videos/00{i}.mp3") for i in range(10)])

.. _class-label-htype:

Class Label Htype
~~~~~~~~~~~~~~~~~

- :bluebold:`Sample dimensions:` ``(# labels,)``

Class labels are stored as numerical values in tensors, which are indices of the list ``tensor.info.class_names``.

:blue:`Creating a class label tensor`
-------------------------------------

A class label tensor can be created using

>>> classes = ["airplanes", "cars", "birds", "cats", "deer", "dogs", "frogs", "horses", "ships", "trucks"]
>>> ds.create_tensor("labels", htype="class_label", class_names=classes, chunk_compression="lz4")

- Optional args:
    - class_names: This must be a **list of strings**. ``tensor.info.class_names`` will be set to this list.
    - :ref:`sample_compression <sample_compression>` or :ref:`chunk_compression <chunk_compression>`.
    - dtype: Defaults to ``uint32``.
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

:bluebold:`Examples`

Appending index

>>> ds.labels.append(0)
>>> ds.labels.append(np.zeros((5,), dtype=np.uint32))

Extending with list of indices

>>> ds.labels.extend([[0, 1, 2], [1, 3]])

Appending text labels

>>> ds.labels.append(["cars", "airplanes"])

.. _tag-htype:

Tag Htype
~~~~~~~~~

- :bluebold:`Sample dimensions:` ``(# tags,)``

This htype can be used to tag samples with one or more string values.

:blue:`Creating a tag tensor`
-----------------------------

A tag tensor can be created using

>>> ds.create_tensor("tags", htype="tag", chunk_compression="lz4")

- Optional args:
    - :ref:`chunk_compression <chunk_compression>`.

- Supported compressions:

>>> ["lz4"]

:blue:`Appending tag samples`
-----------------------------

- Tag samples can be appended as ``str`` or ``list`` of ``str``.

:bluebold:`Examples`

Appending a tag

>>> ds.tags.append("verified")

Extending with list of tags

>>> ds.tags.extend(["verified", "unverified"])

.. _bbox-htype:

Bounding Box Htype
~~~~~~~~~~~~~~~~~~

- :bluebold:`Sample dimensions:` ``(# bounding boxes, 4)``

Bounding boxes have a variety of conventions such as those used in YOLO, COCO, Pascal-VOC and others.
In order for bounding boxes to be correctly displayed by the visualizer, the format of the bounding box must be
specified in the coords key in tensor meta information mentioned below.

:blue:`Creating a bbox tensor`
------------------------------

A bbox tensor can be created using

>>> ds.create_tensor("boxes", htype="bbox", coords={"type": "fractional", "mode": "CCWH"})

- Optional args:
    - **coords**: A dictionary with keys "type" and "mode".
        - **type**: Specifies the units of bounding box coordinates.
            - "pixel": is in unit of pixels.
            - "fractional": is in units relative to the width and height of the image, such as in YOLO format.
        - **mode**: Specifies the convention for the 4 coordinates
            - "LTRB": left_x, top_y, right_x, bottom_y
            - "LTWH": left_x, top_y, width, height
            - "CCWH": center_x, center_y, width, height
    - **dtype**: Defaults to ``float32``.
    - :ref:`sample_compression <sample_compression>` or :ref:`chunk_compression <chunk_compression>`.

- Supported compressions:

>>> ["lz4"]

You can also choose to set the class names after tensor creation.

>>> ds.boxes.info.update(coords = {"type": "pixel", "mode": "LTRB"})

.. note::
    If the bounding box format is not specified, the visualizer will assume a YOLO format (``fractional`` + ``CCWH``) 
    if the box coordinates are < 1 on average. Otherwise, it will assume the COCO format (``pixel`` + ``LTWH``).

:blue:`Appending bounding boxes`
--------------------------------

- Bounding boxes can be appended as ``np.ndarrays`` or ``list`` or ``lists of arrays``.

:bluebold:`Examples`

Appending one bounding box

>>> box
array([[462, 123, 238,  98]])
>>> ds.boxes.append(box)

Appending sample with 3 bounding boxes

>>> boxes
array([[965, 110, 262,  77],
       [462, 123, 238,  98],
       [688, 108, 279, 116]])
>>> boxes.shape
(3, 4)
>>> ds.boxes.append(boxes)


.. _bbox-3d-htype:

3D Bounding Box Htype
~~~~~~~~~~~~~~~~~~~~~

In order for 3D bounding boxes to be correctly displayed by the visualizer, the format of the bounding box must
be specified in the coords key in tensor meta information mentioned below.


:blue:`Creating a 3d bbox tensor`
---------------------------------

.. note::
    In order for 3D bounding boxes to be correctly displayed by the visualizer, the format of the bounding box 
    must be specified in the coords key in tensor meta information mentioned below. In addition, for projecting 
    3D bounding boxes onto 2D data (such as an image), the :ref:`intrinsics <intrinsics-htype>` tensor must exist 
    in the dataset, or the intrinsics matrix must be specified in the ``ds.img_tensor.info`` dictionary, where the key is 
    ``"intrinsics"`` and the value is the matrix.

A 3d bbox tensor can be created using

>>> ds.create_tensor("3d_boxes", htype="bbox.3d", coords={"mode": "center"})

- Optional args:
    - **coords**: A dictionary with key "mode".
        - **mode**: Specifies the convention for the bbox coordinates.
            - "center": [center_x, center_y, center_z, size_x, size_y, size_z, rot_x, rot_y, rot_z]
                - :bluebold:`Sample dimensions:` ``(# bounding boxes, 9)``
                - ``size_x`` - is the length of the bounding box along x direction
                - ``size_y``  - is the width of the bounding box along y direction
                - ``size_z``  - is the height of the bounding box along z direction
                - ``rot_x`` - rotation angle along x axis, given in degrees
                - ``rot_y`` - rotation angle along y axis, given in degrees
                - ``rot_z`` - rotation angle along z axis, given in degrees
            - "vertex": 8 3D vertices - [[x0, y0, z0], [x1, y1, z1], [x2, y2, z2], ....., [x7, y7, z7]]
                - :bluebold:`Sample dimensions:` ``(# bounding boxes, 8, 3)``

                The vertex order is of the following form::

                                 4_____________________ 5
                                /|                    /|
                               / |                   / |
                              /  |                  /  |
                             /___|_________________/   |
                           0|    |                 | 1 |
                            |    |                 |   |
                            |    |                 |   |
                            |    |                 |   |
                            |    |_________________|___|
                            |   /  7               |   / 6
                            |  /                   |  /
                            | /                    | /
                            |/_____________________|/
                             3                      2

    - **dtype**: Defaults to ``float32``.
    - :ref:`sample_compression <sample_compression>` or :ref:`chunk_compression <chunk_compression>`.

- Supported compressions:

>>> ["lz4"]

.. note::
    rotation angles are specified in degrees, not radians

:blue:`Appending 3d bounding boxes`
-----------------------------------

- Bounding boxes can be appended as ``np.ndarrays`` or ``list`` or ``lists of arrays``.

:bluebold:`Examples`

Appending one bounding box

>>> box
array([[462, 123, 238,  98, 22, 36, 44, 18, 0, 36, 0]])
>>> ds.3d_boxes.append(box)

Appending sample with 3 bounding boxes

>>> boxes
array([[965, 110, 262,  77, 22, 36, 44, 18, 0, 28, 0],
       [462, 123, 238,  98, 26, 34, 24, 19, 0, -50, 0],
       [688, 108, 279, 116, 12, 32, 14, 38, 0, 30, 0]])
>>> boxes.shape
(9, 4)
>>> ds.3d_boxes.append(boxes)

.. _intrinsics-htype:

Intrinsics Htype
~~~~~~~~~~~~~~~~

- :bluebold:`Sample dimensions`: ``(# intrinsics matrices, 3, 3)``

The intrinsic matrix represents a projective transformation from the 3-D camera's coordinates into the 2-D image coordinates.
The intrinsic parameters include the focal length, the optical center, also known as the principal point.
The camera intrinsic matrix, :math:`K`, is defined as:

.. math::

    \begin{bmatrix}
    f_x & 0 & c_x \\
    0 & f_y & c_y \\
    0 & 0 & 1 
    \end{bmatrix}

- :math:`[c_x, c_y]` - Optical center (the principal point), in pixels.
- :math:`[f_x, f_y]` - Focal length in pixels.
- :math:`f_x = F / p_x`
- :math:`f_y = F / p_y`
- :math:`F` - Focal length in world units, typically expressed in millimeters.
- :math:`(p_x, p_y)` - Size of the pixel in world units.

:blue:`Creating an intrinsics tensor`
-------------------------------------

An intrinsics tensor can be created using

>>> ds.create_tensor("intrinsics", htype="intrinsics")

- Optional args:
    - :ref:`sample_compression <sample_compression>` or :ref:`chunk_compression <chunk_compression>`.
    - dtype: Defaults to ``float32``.
- Supported compressions:

>>> ["lz4"]

:blue:`Appending intrinsics matrices`
-------------------------------------

>>> intrinsic_params = np.zeros((3, 3))
>>> ds.intrinsics.append(intrinsic_params)


.. _segment-mask-htype:

Segmentation Mask Htype
~~~~~~~~~~~~~~~~~~~~~~~

- :bluebold:`Sample dimensions:` ``(height, width)``

Segmentation masks are 2D representations of class labels where the numerical label data is encoded in an array of
same shape as the image. The numerical values are indices of the list ``tensor.info.class_names``.

:blue:`Creating a segment_mask tensor`
--------------------------------------

A segment_mask tensor can be created using

>>> classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle"]
>>> ds.create_tensor("masks", htype="segment_mask", class_names=classes, sample_compression="lz4")

- Optional args:
    - class_names: This must be a **list of strings**. ``tensor.info.class_names`` will be set to this list.
    - :ref:`sample_compression <sample_compression>` or :ref:`chunk_compression <chunk_compression>`
    - dtype: Defaults to ``uint32``.

- Supported compressions:

>>> ["lz4"]

You can also choose to set the class names after tensor creation.

>>> ds.labels.info.update(class_names = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle"])

.. note::
    Since segmentation masks often contain large amounts of data, it is recommended to compress them
    using ``lz4``.

:blue:`Appending segmentation masks`
------------------------------------

- Segmentation masks can be appended as ``np.ndarray``.

:bluebold:`Examples`

>>> ds.masks.append(np.zeros((512, 512)))

.. note::
    Since each pixel can only be labeled once, segmentation masks are not appropriate for datasets where objects
    might overlap, or where multiple objects within the same class must be distinguished. For these use cases,
    please use :ref:`htype = "binary_mask" <Binary Mask Htype>`.

.. _binary-mask-htype:

Binary Mask Htype
~~~~~~~~~~~~~~~~~

- :bluebold:`Sample dimensions:` ``(height, width, # objects in a sample)``

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
    - ref:`sample_compression <sample_compression>` or :ref:`chunk_compression <chunk_compression>`
    - dtype: Defaults to ``bool``.
- Supported compressions:

>>> ["lz4"]

.. note::
    Since segmentation masks often contain large amounts of data, it is recommended to compress them
    using ``lz4``.

:blue:`Appending binary masks`
------------------------------

- Binary masks can be appended as ``np.ndarray``.

:bluebold:`Examples`

Appending a binary mask with 5 objects

>>> ds.masks.append(np.zeros((512, 512, 5), dtype="bool"))
>>> ds.labels.append(["aeroplane", "aeroplane", "bottle", "bottle", "bird"])

.. _keypoints-coco-htype:

COCO Keypoints Htype
~~~~~~~~~~~~~~~~~~~~

- :bluebold:`Sample dimensions:` ``(3 x # keypoints, # objects in a sample)``

`COCO keypoints <https://cocodataset.org/#format-data>`_ are a convention for storing points of interest
in an image. Each keypoint consists of 3 values: ``x - coordinate``, ``y - coordinate`` and ``v - visibility``.
A set of ``K`` keypoints of an object is represented as:

[x\ :sub:`1`, y\ :sub:`1`, v\ :sub:`1`, x\ :sub:`2`, y\ :sub:`2`, v\ :sub:`2`, ..., x\ :sub:`k`, y\ :sub:`k`, v\ :sub:`k`]

The visibility ``v`` can be one of three values:

:0: keypoint not in image.
:1: keypoint in image but not visible.
:2: keypoint in image and visible.

:blue:`Creating a keypoints_coco tensor`
----------------------------------------

A keypoints_coco tensor can be created using

>>> ds.create_tensor("keypoints", htype="keypoints_coco", keypoints=["knee", "elbow", "head"], connections=[[0, 1], [1, 2]])

- Optional args:
    - keypoints: List of strings describing the ``i`` th keypoint. ``tensor.info.keypoints`` will be set to this list.
    - connections: List of strings describing which points should be connected by lines in the visualizer.
    - :ref:`sample_compression <sample_compression>` or :ref:`chunk_compression <chunk_compression>`
    - dtype: Defaults to ``int32``.

- Supported compressions:

>>> ["lz4"]

You can also choose to set ``keypoints`` and / or ``connections`` after tensor creation.

>>> ds.keypoints.info.update(keypoints = ['knee', 'elbow',...])
>>> ds.keypoints.info.update(connections = [[0,1], [2,3], ...])

:blue:`Appending keypoints`
---------------------------

- Keypoints can be appended as ``np.ndarray`` or ``list``.

:bluebold:`Examples`

Appending keypoints sample with 3 keypoints and 4 objects

>>> ds.keypoints.update(keypoints = ["left ear", "right ear", "nose"])
>>> ds.keypoints.update(connections = [[0, 2], [1, 2]])
>>> kp_arr
array([[465, 398, 684, 469],
       [178, 363, 177, 177],
       [  2,   2,   2,   1],
       [454, 387, 646, 478],
       [177, 322, 137, 161],
       [  2,   2,   2,   2],
       [407, 379, 536, 492],
       [271, 335, 150, 143],
       [  2,   1,   2,   2]])
>>> kp_arr.shape
(9, 4)
>>> ds.keypoints.append(kp_arr)

.. warning::
    In order to correctly use the keypoints and connections metadata, it is critical that all objects
    in every sample have the same number of K keypoints in the same order. For keypoints that are not
    present in an image, they can be stored with dummy coordinates of x = 0, y = 0, and v = 0, and the
    visibility will prevent them from being drawn in the visualizer. 

.. _point-htype:

Point Htype
~~~~~~~~~~~

- :bluebold:`Sample dimensions:` ``(# points, 2)`` in case of 2-D (X, Y) co-ordinates or ``(# points, 3)`` in case of 3-D (X, Y, Z) co-ordinates of the point.

Points does not contain a fixed mapping across samples between the point order and real-world objects (i.e., point 0 
is an elbow, point 1 is a knee, etc.). If you require such a mapping, use `COCO Keypoints Htype`_.

:blue:`Creating a point tensor`
-------------------------------

A point tensor can be created using

>>> ds.create_tensor("points", htype="point", sample_compression=None)

- Optional args:
    - :ref:`sample_compression <sample_compression>` or :ref:`chunk_compression <chunk_compression>`
    - dtype: Defaults to ``int32``. 
- Supported compressions:

>>> ["lz4"]

:blue:`Appending point samples`
-------------------------------

- Points can be appended as ``np.ndarray`` or ``list``.

:bluebold:`Examples`

Appending 2 2-D points

>>> ds.points.append([[0, 1], [1, 3]])

Appending 2 3-D points

>>> ds.points.append(np.zeros((2, 3)))

.. _polygon-htype:

Polygon Htype
~~~~~~~~~~~~~

- :bluebold:`Sample dimensions:` ``(# polygons, # points per polygon, # co-ordinates per point)``

- Each sample in a tensor of ``polygon`` htype is a list of polygons.
- Each polygon is a list / array of points.
- All points in a sample should have the same number of co-ordinates (eg., cannot mix 2-D points with 3-D points).
- Different samples can have different number of polygons.
- Different polygons can have different number of points.

:blue:`Creating a polygon tensor`
---------------------------------

A polygon tensor can be created using

>>> ds.create_tensor("polygons", htype="polygon", sample_compression=None)

- Optional args:
    - :ref:`sample_compression <sample_compression>` or :ref:`chunk_compression <chunk_compression>`
    - dtype: Defaults to ``float32``. 
- Supported compressions:

>>> ["lz4"]

:blue:`Appending polygons`
--------------------------

- Polygons can be appended as a ``list`` of ``list of tuples`` or ``np.ndarray``.

:bluebold:`Examples`

Appending polygons with 2-D points

>>> poly1 = [(1, 2), (2, 3), (3, 4)]
>>> poly2 = [(10, 12), (14, 19)]
>>> poly3 = [(33, 32), (54, 67), (67, 43), (56, 98)]
>>> sample = [poly1, poly2, poly3]
>>> ds.polygons.append(sample)

Appending polygons with 3-D points

>>> poly1 = [(10, 2, 9), (12, 3, 8), (12, 10, 4)]
>>> poly2 = [(10, 1, 8), (5, 17, 11)]
>>> poly3 = [(33, 33, 31), (45, 76, 13), (60, 24, 17), (67, 87, 83)]
>>> sample = [poly1, poly2, poly3]
>>> ds.polygons.append(sample)

Appending polygons with numpy arrays

>>> import numpy as np
>>> sample = np.random.randint(0, 10, (5, 7, 2))  # 5 polygons with 7 points
>>> ds.polygons.append(sample)

>>> import numpy as np
>>> poly1 = np.random.randint(0, 10, (5, 2))
>>> poly2 = np.random.randint(0, 10, (8, 2))
>>> poly3 = np.random.randint(0, 10, (3, 2))
>>> sample = [poly1, poly2, poly3]
>>> ds.polygons.append(sample)

.. _nifti-htype:

Nifti Htype
~~~~~~~~~~~

- :bluebold:`Sample dimensions:` ``(# height, # width, # slices)`` or ``(# height, # width, # slices, # time unit)`` in case of time-series data.

:blue:`Creating a nifti tensor`
-------------------------------

A nifti tensor can be created using 

>>> ds.create_tensor("patients", htype="nifti", sample_compression="nii.gz")

- Supported compressions:

>>> ["nii.gz", "nii", None]

:blue:`Appending nifti data`
----------------------------

- Nifti samples can be of type ``np.ndarray`` or :class:`~deeplake.core.sample.Sample` which is returned by :meth:`deeplake.read`.
- Deep Lake does not support compression of raw nifti data. Therefore, array of raw frames can only be appended to tensors with
  ``None`` compression.

:bluebold:`Examples`

>>> ds.patients.append(deeplake.read("data/patient0.nii.gz"))

>>> ds.patients.extend([deeplake.read(f"data/patient{i}.nii.gz") for i in range(10)])

.. _point_cloud-htype:

Point Cloud Htype
~~~~~~~~~~~~~~~~~

- :bluebold:`Sample dimensions:` ``(# num_points, 3)``
- Point cloud samples can be of type ``np.ndarray`` or :class:`~deeplake.core.sample.Sample` which is returned by :meth:`deeplake.read`.
- Each point cloud is a list / array of points.
- All points in a sample should have the same number of co-ordinates.
- Different point clouds can have different number of points.

:blue:`Creating a point cloud tensor`
-------------------------------------

A point cloud tensor can be created using

>>> ds.create_tensor("point_clouds", htype="point_cloud", sample_compression="las")

- Optional args:
    - :ref:`sample_compression <sample_compression>`
- Supported compressions:

>>> [None, "las"]

:blue:`Appending point clouds`
------------------------------

- Point clouds can be appended as a ``np.ndarray``.

:bluebold:`Examples`

Appending point clouds with numpy arrays

>>> import numpy as np
>>> point_cloud1 = np.random.randint(0, 10, (5, 3))
>>> ds.point_clouds.append(point_cloud1)
>>> point_cloud2 = np.random.randint(0, 10, (15, 3))
>>> ds.point_clouds.append(point_cloud2)
>>> ds.point_clouds.shape
>>> (2, None, 3)

Or we can use :meth:`deeplake.read` method to add samples

>>> import deeplake as dp
>>> sample = dp.read("example.las") # point cloud with 100 points
>>> ds.point_cloud.append(sample)
>>> ds.point_cloud.shape
>>> (1, 100, 3)

.. _mesh-htype:

Mesh Htype
~~~~~~~~~~

- :bluebold:`Sample dimensions:` ``(# num_points, 3)``
- Mesh samples can be of type ``np.ndarray`` or :class:`~deeplake.core.sample.Sample` which is returned by :meth:`deeplake.read`.
- Each sample in a tensor of ``mesh`` htype is a mesh array (3-D object data).
- Each mesh is a list / array of points.
- Different meshes can have different number of points.

:blue:`Creating a mesh tensor`
------------------------------

A mesh tensor can be created using

>>> ds.create_tensor("mesh", htype="mesh", sample_compression="ply")

- Optional args:
    - :ref:`sample_compression <sample_compression>`
- Supported compressions:

>>> ["ply"]

:blue:`Appending meshes`
------------------------

:bluebold:`Examples`

Appending a ply file containing a mesh data to tensor

>>> import deeplake as dp
>>> sample = dp.read("example.ply")  # mesh with 100 points and 200 faces
>>> ds.mesh.append(sample)

>>> ds.mesh.shape
>>> (1, 100, 3)


.. _embedding-htype:

Embedding Htype
~~~~~~~~~~~~~~~

- :bluebold:`Sample dimensions:` ``(# elements in the embedding,)``

:blue:`Creating an embedding tensor`
------------------------------------

An embedding tensor can be created using

>>> ds.create_tensor("embedding", htype="embedding")

- Supported compressions:

>>> ["lz4", None]

:blue:`Appending embedding samples`
-----------------------------------

- Embedding samples can be of type ``np.ndarray``.

:bluebold:`Examples`

Appending Deep Lake embedding sample

>>> ds.embedding.append(np.random.uniform(low=-1, high=1, size=(1024)))

Extending with Deep Lake embeddding samples

>>> ds.embedding.extend([np.random.uniform(low=-1, high=1, size=(1024)) for i in range(10)])


.. _sequence-htype:

Sequence htype
~~~~~~~~~~~~~~

- A special meta htype for tensors where each sample is a sequence. The items in the sequence are samples of another htype.
- It is a wrapper htype that can wrap other htypes like ``sequence[image]``, ``sequence[video]``, ``sequence[text]``, etc.

:bluebold:`Examples`

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
>>> ds.image_seq.append([deeplake.read("img01.jpg"), deeplake.read("img02.jpg")])

.. _link-htype:

Link htype
~~~~~~~~~~

- Link htype is a special meta htype that allows linking of external data (files) to the dataset, without storing the data in the dataset itself.
- Moreover, there can be variations in this htype, such as ``link[image]``, ``link[video]``, ``link[audio]``, etc. that would enable the activeloop visualizer to correctly display the data.
- No data is actually loaded until you try to read the sample from a dataset.
- There are a few exceptions to this:-
    - If ``create_shape_tensor=True`` was specified during ``create_tensor`` of the tensor to which this is being added, the shape of the sample is read. This is ``True`` by default.
    - If ``create_sample_info_tensor=True`` was specified during ``create_tensor`` of the tensor to which this is being added, the sample info is read. This is ``True`` by default.
    - If ``verify=True`` was specified during ``create_tensor`` of the tensor to which this is being added, some metadata is read from them to verify the integrity of the link samples. This is ``True`` by default.
    - If you do not want to verify your links, all three of ``verify``, ``create_shape_tensor`` and ``create_sample_info_tensor`` have to be set to ``False``.

.. _linked_sample_examples:

:bluebold:`Examples`

>>> ds = deeplake.dataset("......")

:bluebold:`Adding credentials to the dataset`

You can add the names of the credentials you want to use (not needed for http/local urls)

>>> ds.add_creds_key("MY_S3_KEY")
>>> ds.add_creds_key("GCS_KEY")

and populate the added names with credentials dictionaries

>>> ds.populate_creds("MY_S3_KEY", {})   # add creds here
>>> ds.populate_creds("GCS_KEY", {})    # add creds here

These creds are only present temporarily and will have to be repopulated on every reload.

For datasets connected to Activeloop Platform,
`you can store your credentials on the platform <https://docs.activeloop.ai/storage-and-credentials/managed-credentials#managed-credentials-ui>`_ as Managed Credentials and 
use them just by adding the keys to your dataset. For example if you have managed credentials with names ``"my_s3_creds"``, ``"my_gcs_creds"``, you can add them to your dataset using
:meth:`Dataset.add_creds_key <deeplake.core.dataset.Dataset.add_creds_key>` without having to populate them.

>>> ds.add_creds_key("my_s3_creds", managed=True)
>>> ds.add_creds_key("my_gcs_creds", managed=True)


:bluebold:`Create a link tensor`

>>> ds.create_tensor("img", htype="link[image]", sample_compression="jpg")


:bluebold:`Populate the tensor with links`

>>> ds.img.append(deeplake.link("s3://abc/def.jpeg", creds_key="my_s3_key"))
>>> ds.img.append(deeplake.link("gcs://ghi/jkl.png", creds_key="GCS_KEY"))
>>> ds.img.append(deeplake.link("https://picsum.photos/200/300")) # http path doesn’t need creds
>>> ds.img.append(deeplake.link("./path/to/cat.jpeg")) # local path doesn’t need creds
>>> ds.img.append(deeplake.link("s3://abc/def.jpeg"))  # this will throw an exception as cloud paths always need creds_key
:bluebold:`Accessing the data`

>>> for i in range(5):
...     ds.img[i].numpy()
...

:bluebold:`Updating a sample`

>>> ds.img[0] = deeplake.link("./data/cat.jpeg")
