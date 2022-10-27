try:
    from .mmdet_ import train_detector
except ImportError:
    train_detector = None
