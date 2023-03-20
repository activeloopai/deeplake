import torch


def empty_cuda():
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    return