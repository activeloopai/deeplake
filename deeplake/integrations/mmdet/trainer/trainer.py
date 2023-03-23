from train_detector_class import TrainDectector


def _train_detector(*args, **kwargs):
    trainer = TrainDectector(*args, **kwargs)
    trainer.run()