def pytorch_installed():
    try:
        import torch

        torch.__version__
    except ImportError:
        return False
    return True


def tfds_installed():
    try:
        import tensorflow_datasets  # type: ignore
        import tensorflow  # type: ignore

        tensorflow_datasets.__version__
    except ImportError:
        return False
    return True
