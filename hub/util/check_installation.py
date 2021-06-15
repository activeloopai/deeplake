def pytorch_installed():
    try:
        import torch

        torch.__version__
    except ImportError:
        return False
    return True


def tensorflow_installed():
    try:
        import tensorflow  # type: ignore

        tensorflow.__version__
    except ImportError:
        return False
    return True
