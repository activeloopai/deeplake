try:
    from multiprocessing import shared_memory
    from .pytorch import dataset_to_pytorch
except ImportError:
    from .pytorch_old import dataset_to_pytorch

from .tensorflow import dataset_to_tensorflow
