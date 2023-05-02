import deeplake
from deeplake.constants import MB
from deeplake.enterprise.util import raise_indra_installation_error

try:
    from indra import api

    _INDRA_INSTALLED = True
except Exception:
    _INDRA_INSTALLED = False

import numpy as np

import uuid
from functools import partial
from typing import Optional, Any, Iterable, List, Dict, Callable, Union


def check_indra_installation(exec_option, indra_installed):
    if exec_option == "indra" and not indra_installed:
        raise raise_indra_installation_error(indra_import_error=False)
