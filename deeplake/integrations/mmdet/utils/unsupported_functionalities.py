from deeplake.util.warnings import always_warn


def check_unused_dataset_fields(cfg):
    if cfg.get("dataset_type"):
        always_warn(
            "The deeplake mmdet integration does not use dataset_type to work with the data and compute metrics. All deeplake datasets are in the same deeplake format. To specify a metrics format, you should deeplake_metrics_format "
        )

    if cfg.get("data_root"):
        always_warn(
            "The deeplake mmdet integration does not use data_root, this input will be ignored"
        )


def check_unsupported_train_pipeline_fields(cfg, mode="train"):
    transforms = cfg.data[mode].pipeline

    for transform in transforms:
        transform_type = transform.get("type")

        if transform_type == "LoadImageFromFile":
            always_warn(
                "LoadImageFromFile is going to be skipped because deeplake mmdet integration does not use it"
            )

        if transform_type == "LoadAnnotations":
            always_warn(
                "LoadAnnotations is going to be skipped because deeplake mmdet integration does not use it"
            )

        if transform_type == "Corrupt":
            raise Exception("Corrupt augmentation is not supported yet.")

        elif transform_type == "CopyPaste":  # TO DO: @adolkhan resolve this
            raise Exception("CopyPaste augmentation is not supported yet")

        elif transform_type == "CutOut":  # TO DO: @adolkhan resolve this
            raise Exception("CutOut augmentation is not supported yet")

        elif transform_type == "Mosaic":  # TO DO: @adolkhan resolve this
            raise Exception("Mosaic augmentation is not supported yet")


def check_dataset_augmentation_formats(cfg):
    if cfg.get("train_dataset"):
        always_warn(
            "train_dataset is going to be unused. Datset types like: ConcatDataset, RepeatDataset, ClassBalancedDataset, MultiImageMixDataset are not supported."
        )


def check_unsupported_functionalities(cfg):
    check_unused_dataset_fields(cfg)
    check_unsupported_train_pipeline_fields(cfg, mode="train")
    check_unsupported_train_pipeline_fields(cfg, mode="val")
    check_dataset_augmentation_formats(cfg)
