def pytorch_installed():
    try:
        import torch

        torch.__version__
    except ImportError:
        return False
    return True
