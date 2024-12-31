class CocoAnnotationMissingError(Exception):
    def __init__(self, keys):
        super().__init__(
            (
                "COCO dataset ingestion expects to have `instances`, `keypoints` and `stuff`. "
                "{} {} missing."
            ).format(
                f"Key {keys[0]}" if len(keys) == 1 else f"Keys {', '.join(keys)}",
                "is" if len(keys) == 1 else "are",
            )
        )
