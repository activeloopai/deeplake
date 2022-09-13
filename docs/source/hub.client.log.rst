hub.client.log
==============
Hub does logging using the "hub" logger. Logging level is ``logging.INFO`` by default. See example on how to change this.

>>> import hub
>>> import logging
>>> logger = logging.getLogger("hub")
>>> logger.setLevel(logging.WARNING)