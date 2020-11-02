from hub.features.features import Tensor


class Sequence(Tensor):
    """`Sequence` correspond to sequence of `features.FeatureConnector`.
    At generation time, a list for each of the sequence element is given. The output
    of `Dataset` will batch all the elements of the sequence together.
    If the length of the sequence is static and known in advance, it should be
    specified in the constructor using the `length` param.

    Example:
    ```
    sequence = Sequence(Image(), length=NB_FRAME)
    ```
    """

    def __init__(self, feature: Tensor, length: int = None, chunks=True):
        """Construct a sequence of Tensors.
        Args:
        feature: the features to wrap
        length: `int`, length of the sequence if static and known in advance
        """
        super(Sequence, self).__init__(shape=(length,), dtype=feature, chunks=chunks)

    def get_attr_dict(self):
        """Return class attributes"""
        return self.__dict__
