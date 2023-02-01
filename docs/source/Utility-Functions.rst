Utility Functions
=================
.. currentmodule:: deeplake


General Functions
~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree:
    :nosignatures:

    list
    exists

Making Deep Lake Samples
~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::
    :toctree:
    :nosignatures:

    read
    link
    link_tiled

Parallelism
~~~~~~~~~~~
.. autosummary::
    :toctree:
    :nosignatures:

    compute
    compose

Transform pipelines returned by :func:`compute` and :func:`compose` are evaluated using ``eval``:

.. currentmodule:: deeplake.core.transform

.. autosummary::
    :toctree:
    :nosignatures:

    ~Pipeline.eval
