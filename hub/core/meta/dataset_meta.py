from hub.core.meta.meta import Meta


class DatasetMeta(Meta):
    def __init__(self):
        self.tensors = []

        super().__init__()

    @property
    def nbytes(self):
        # TODO: can optimize this
        return len(self.tobytes())

    def as_dict(self) -> dict:
        d = super().as_dict()
        d["tensors"] = self.tensors
        return d
