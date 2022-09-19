Installation
============

Hub can be installed with pip ::

    pip install hub

Hub has the following extras that you can choose to install according to your needs.

+--------------------------------------+---------------------------------------+---------------------------------------------+
| Install command                      | Description                           | Dependencies installed                      |
+======================================+=======================================+=============================================+
| ``pip install hub[av]``              | Audio and video support via PyAV      | av                                          |
+--------------------------------------+---------------------------------------+---------------------------------------------+
| ``pip install hub[visualizer]``      | Visualize datasets within notebooks   | IPython, flask                              |
+--------------------------------------+---------------------------------------+---------------------------------------------+
| ``pip install hub[gcp]``             | GCS support                           | | google-cloud-storage, google-auth,        |
|                                      |                                       | | google-auth-oauthlib                      |
+--------------------------------------+---------------------------------------+---------------------------------------------+
| ``pip install hub[dicom]``           | DICOM data support                    | pydicom                                     |
+--------------------------------------+---------------------------------------+---------------------------------------------+
| ``pip install hub[gdrive]``          | Google Drive support                  | | google-api-python-client, oauth2client,   |
|                                      |                                       | | google-auth, google-auth-oauthlib         |
+--------------------------------------+---------------------------------------+---------------------------------------------+
| ``pip install hub[point_cloud]``     | Support for LiDAR point cloud data    | laspy                                       |
+--------------------------------------+---------------------------------------+---------------------------------------------+
| ``pip install hub[deeplake]``        | Installs                              | deeplake                                    |
|                                      | :ref:`experimental<Experimental>`     |                                             |
|                                      | features                              |                                             |
+--------------------------------------+---------------------------------------+---------------------------------------------+
| ``pip install hub[all]``             | Installs all of the above             |                                             |
+--------------------------------------+---------------------------------------+---------------------------------------------+
