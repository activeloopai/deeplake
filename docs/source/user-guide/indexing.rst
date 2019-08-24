Indexing
#################################

Indexing of Hub Array follows the same syntax as numpy array.
Data fetching happens when indexing of the Array was executed.
The package fetches the data from local cache or from cloud object storage (S3/GCS/Blob) when indexing of an array was executed.
When we write data to the indexed array, the package synchronizes the change to the local cache, and then to the S3/GCS/Blob backend.

.. ipython:: python

    >>> import hub
    >>> bigarray = hub.array((10000000000, 512, 512, 3), 
                    name="test/bigarray:v0")

    >>> import numpy as np
    >>> bigarray[0,:,:,:] = np.ones((512, 512, 3)) 
    >>> bigarray[0,0:5,0,0]
    array([1., 1., 1., 1., 1.])

    >>> bigarray[0:3,:,:,:].shape
    (3, 512, 512, 3)

    >>> bigarray[0:10,:,:,0].shape
    (10, 512, 512)

    >>> bigarray[0,0,0,0]
    1.0

    >>> bigarray[0,0,0,0] = 2.5
    >>> bigarray[0,0,0,0]
    2.5