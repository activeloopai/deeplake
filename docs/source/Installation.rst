Installation
============

Deep Lake can be installed with pip ::

    pip install deeplake

Deep Lake has the following extras that you can choose to install according to your needs.

+--------------------------------------+---------------------------------------+---------------------------------------------+
| Install command                      | Description                           | Dependencies installed                      |
+======================================+=======================================+=============================================+
| ``pip install deeplake[av]``         | Audio and video support via PyAV      | av                                          |
+--------------------------------------+---------------------------------------+---------------------------------------------+
| ``pip install deeplake[visualizer]`` | Visualize Deep Lake datasets within   | IPython, flask                              |
|                                      | notebooks                             |                                             |
+--------------------------------------+---------------------------------------+---------------------------------------------+
| ``pip install deeplake[gcp]``        | GCS support                           | | google-cloud-storage, google-auth,        |
|                                      |                                       | | google-auth-oauthlib                      |
+--------------------------------------+---------------------------------------+---------------------------------------------+
| ``pip install deeplake[dicom]``      | DICOM data support                    | pydicom                                     |
+--------------------------------------+---------------------------------------+---------------------------------------------+
| ``pip install deeplake[gdrive]``     | Google Drive support                  | | google-api-python-client, oauth2client,   |
|                                      |                                       | | google-auth, google-auth-oauthlib         |
+--------------------------------------+---------------------------------------+---------------------------------------------+
| ``pip install deeplake[point_cloud]``| Support for LiDAR point cloud data    | laspy                                       |
+--------------------------------------+---------------------------------------+---------------------------------------------+
| ``pip install deeplake[all]``        | Installs all of the above             |                                             |
+--------------------------------------+---------------------------------------+---------------------------------------------+
