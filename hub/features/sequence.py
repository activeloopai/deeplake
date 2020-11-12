from hub.features.features import Tensor


class Sequence(Tensor):
    """`Sequence` correspond to sequence of `features.HubFeature`.
    At generation time, a list for each of the sequence element is given. The output
    of `Dataset` will batch all the elements of the sequence together.
    If the length of the sequence is static and known in advance, it should be
    specified in the constructor using the `length` param.

    | Usage:
    ----------

    >>> sequence = Sequence(Image(), length=NB_FRAME)
    """

    def __init__(
        self,
        shape=(None,),
        dtype=None,
        chunks=None,
        compressor="lz4",
    ):
        """Construct a sequence of Tensors.
        Parameters
        ----------
        shape : Tuple[int] | int
            Single integer element tuple representing length of sequence
            If None then dynamic
        dtype : str | HubFeature
            Datatype of each element in sequence
        chunks : Tuple[int] | int
            Number of elements in chunk
            Works only for top level sequence
            You can also include number of samples in a single chunk
        """
        super().__init__(
            shape=shape,
            dtype=dtype,
            chunks=chunks,
            compressor=compressor,
        )

    def get_attr_dict(self):
        """Return class attributes"""
        return self.__dict__
