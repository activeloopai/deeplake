def mmdet_subiterable_dataset_eval(
    self,
    *args,
    **kwargs,
):
    return self.mmdet_dataset.evaluate(*args, **kwargs)
