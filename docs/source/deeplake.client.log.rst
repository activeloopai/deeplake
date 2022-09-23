deeplake.client.log
===================
Deep Lake does logging using the "deeplake" logger. Logging level is ``logging.INFO`` by default. See example on how to change this.

>>> import deeplake
>>> import logging
>>> logger = logging.getLogger("deeplake")
>>> logger.setLevel(logging.WARNING)