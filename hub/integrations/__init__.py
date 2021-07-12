from hub.util.exceptions import WindowsSharedMemoryError


try:
    import os

    if os.name == "nt":
        raise WindowsSharedMemoryError
    from multiprocessing import shared_memory
    from .pytorch import dataset_to_pytorch
except (ImportError, WindowsSharedMemoryError):
    from .pytorch_old import dataset_to_pytorch

from .tensorflow import dataset_to_tensorflow
