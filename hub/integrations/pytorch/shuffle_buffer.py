from typing import List, Any
from random import randrange
from functools import reduce
from operator import mul
import warnings


class ShuffleBuffer:
    """Shuffling buffer used to shuffle samples by the rule:

    Given new sample if buffer is not full, add sample to the buffer else pick other sample
    randomly and swap it with given sample.

    Args:
        size(int):  size of the buffer in bytes
    Raises:
        ValueError if buffer size is not set
    """

    def __init__(self, size: int) -> None:
        if size <= 0:
            raise ValueError("Buffer size should be positive value more than zero")

        self.size = size
        self.buffer: List[Any] = list()
        self.buffer_used = 0

    def exchange(self, sample):
        """Shuffle with existing elements in a buffer and return value if buffer is full or if `None` is provided as argument.

        Args:
            sample: new sample to add or None

        Returns:
            random sample or None,
            same sample if buffer is empty and sample doesn't fit
        """
        buffer_len = len(self.buffer)

        if not sample is None:
            sample_size = self._sample_size(sample)

            # fill buffer of not reach limit
            if self.buffer_used + sample_size <= self.size:
                self.buffer_used += sample_size
                self.buffer.append(sample)
                return None

            if buffer_len == 0:
                warnings.warn(
                    f"Buffer size is too small. Sample with size {sample_size} does not fit in buffer of size {self.size}"
                )
                return sample

            # exchange samples with shuffle buffer
            selected = randrange(buffer_len)
            val = self.buffer[selected]
            self.buffer[selected] = sample

            self.buffer_used += sample_size
            self.buffer_used -= self._sample_size(val)
            return val
        else:
            if buffer_len > 0:

                # return random selection
                selected = randrange(buffer_len)
                val = self.buffer.pop(selected)
                self.buffer_used -= self._sample_size(val)

                return val
            else:
                return None

    def emtpy(self) -> bool:
        return len(self.buffer) == 0

    def _sample_size(self, sample):
        return sum(
            [
                tensor.storage().element_size() * reduce(mul, tensor.shape, 1)
                for _, tensor in sample.items()
            ]
        )

    def __len__(self):
        return len(self.buffer)

    def __str__(self) -> str:
        return f"ShuffleBuffer(size = {self.size}, buffer_used = {self.buffer_used}, samples = {len(self.buffer)})"
