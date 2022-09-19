Sample Compressions
===================

.. currentmodule:: hub

File formats supported by Hub:

+----------------+-----------+----------------------------+
| File Type      |  Sample Compressions                   |
+================+========================================+
| Image          | | ``bmp``, ``dib``, ``gif``, ``ico``,  |
|                | | ``jpeg``, ``jpeg2000``, ``pcx``,     |
|                | | ``png``, ``ppm``, ``sgi``, ``tga``,  |
|                | | ``tiff``, ``webp``, ``wmf``, ``xbm`` |
+----------------+----------------------------------------+
| Video          | ``mp4``, ``mkv``, ``avi``              |
+----------------+----------------------------------------+
| Audio          | ``flac``, ``mp3``, ``wav``             |
+----------------+----------------------------------------+
| Dicom          | ``dcm``                                |
+----------------+----------------------------------------+

Sample compressions may have to be specified when :meth:`creating tensors <hub.core.dataset.Dataset.create_tensor>`
or in :meth:`hub.read`.
