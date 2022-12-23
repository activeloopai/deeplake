from typing import List, Any, Sequence
from random import randrange
from functools import reduce
from operator import mul
import warnings
import numpy as np
import torch
from PIL import Image  # type: ignore
from io import BytesIO
from tqdm import tqdm  # type: ignore


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
        self.pbar = tqdm(
            total=self.size,
            desc="Please wait, filling up the shuffle buffer with samples.",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        )
        self.pbar_closed = False

    def exchange(self, sample):
        """Shuffle with existing elements in a buffer and return value if buffer is full or if `None` is provided as argument.

        Args:
            sample: new sample to add or None

        Returns:
            random sample or None,
            same sample if buffer is empty and sample doesn't fit
        """
        buffer_len = len(self.buffer)

        if sample is not None:
            sample_size = self._sample_size(sample)

            # fill buffer of not reach limit
            if self.buffer_used + sample_size <= self.size:
                self.buffer_used += sample_size
                self.pbar.update(sample_size)
                self.buffer.append(sample)
                return None
            elif not self.pbar_closed:
                self.close_buffer_pbar()

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
            if not self.pbar_closed:
                self.close_buffer_pbar()
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
        if isinstance(sample, (int, float)):
            return 8
        elif isinstance(sample, bool):
            return 1
        elif isinstance(sample, (str, bytes)):
            return len(sample)
        elif isinstance(sample, dict):
            return sum(self._sample_size(tensor) for tensor in sample.values())
        elif isinstance(sample, Sequence):
            return sum(self._sample_size(tensor) for tensor in sample)
        elif isinstance(sample, torch.Tensor):
            return sample.element_size() * reduce(mul, sample.shape, 1)
        elif isinstance(sample, np.ndarray):
            return sample.nbytes
        elif isinstance(sample, Image.Image):
            img = BytesIO()
            sample.save(img, sample.format)
            return len(img.getvalue())
        raise ValueError(
            f"Expected input of type bytes, dict, Sequence, torch.Tensor, np.ndarray or PIL image, got: {type(sample)}"
        )

    def __len__(self):
        return len(self.buffer)

    def __str__(self) -> str:
        return f"ShuffleBuffer(size = {self.size}, buffer_used = {self.buffer_used}, samples = {len(self.buffer)})"

    def close_buffer_pbar(self):
        if not self.pbar_closed:
            self.pbar.close()
            self.pbar_closed = True
            print("Shuffle buffer filling is complete.")
