def get_collect_keys(cfg):
    pipeline = cfg.train_pipeline
    for transform in pipeline:
        if transform["type"] == "Collect":
            return transform["keys"]
    raise ValueError("collection keys were not specified")