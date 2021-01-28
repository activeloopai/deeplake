"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

from hub.schema.features import Tensor


class Sequence(Tensor):
    """`Sequence` correspond to sequence of `features.HubSchema`.
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
        shape=(),
        max_shape=None,
        dtype=None,
        chunks=None,
        compressor="lz4",
    ):
        """| Construct a sequence of Tensors.

        Parameters
        ----------
        shape : Tuple[int] | int
            Single integer element tuple representing length of sequence
            If None then dynamic
        dtype : str | HubSchema
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
            max_shape=max_shape,
        )

    def __str__(self):
        out = super().__str__()
        out = "Sequence" + out[6:]
        return out

    def __repr__(self):
        return self.__str__()
