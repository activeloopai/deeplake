deeplake.random.seed
====================

Train and operate with the deeplake features and keep reproducibility between all the engines
See the ``deeplake.random.seed`` method on hot to operate with the random engines:

.. currentmodule:: deeplake.core.seed

.. autosummary::
    :toctree:
    :nosignatures:

    deeplake.random

Deeplake random number generator
--------------------------------
Deeplake does not provide random in-house random number generator but to control the reproducibility
and keep track on the stages of 
You can use :meth:`deeplake.random.seed` to seed the RNG for all the engines (both
enterprise and non enterprise)::

    import deeplake
    deeplake.random.seed(0)


Random number generators in other libraries
-------------------------------------------
If you or any of the libraries you are using rely on NumPy, you can seed the global
NumPy RNG with::

    import numpy as np
    np.random.seed(0)

    import random
    random.seed(0)

those will impact to all the places where deeplake uses those libraries but with use of use :meth:`deeplake.random.seed` we are nor changing any library
global state instead just initialising singlton instances in the random engines

DeeplakeRandom
~~~~~~~~~~~~~~~~~~

.. currentmodule:: deeplake.core.seed

.. autoclass:: DeeplakeRandom
    :members:
