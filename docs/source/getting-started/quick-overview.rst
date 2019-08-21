Getting Started
####################################

**Step 1.** Install

.. code-block:: shell

    pip3 install hub


**Step 2.** Lazy-load a public dataset, and fetch a single image with up to 50MB/s speed and plot

.. code-block:: python

    > import hub
    > imagenet = hub.load(name='imagenet')
    > imagenet.shape
    (1034908, 469, 387, 3)

    > import matplotlib.pyplot as plt
    > plt.imshow(imagenet[0])


**Step 3.** Compute the mean and standard deviation of any chunk of the full dataset. 
The package will download the chunk to the local environment and compute locally as a numpy array.

.. code-block:: python

    > imagenet[0:10,100:200,100:200].mean()
    0.132
    > imagenet[0:10,100:200,100:200].std()
    0.005

**Step 4.** Create your own array and access it from another machine

.. code-block:: python

    # Create on one machine
    > import numpy as np
    > mnist = hub.array((50000,28,28,1), name="name/random_name:v1")
    > mnist[0,:,:,:] = np.random.random((1,28,28,1))

    # Access it from another machine
    > mnist = hub.load(name='name/random_name:v1')
    > print(mnist[0])
