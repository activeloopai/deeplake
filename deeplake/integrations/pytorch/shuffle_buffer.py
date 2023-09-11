from typing import List, Any, Sequence
from random import Random
from functools import reduce
from operator import mul
import numpy as np

import warnings
import sys

from PIL import Image  # type: ignore
from io import BytesIO
from tqdm import tqdm
from deeplake.util.warnings import always_warn
from deeplake.constants import MB
import deeplake


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
        from deeplake.core.seed import DeeplakeRandom

        self.random = Random(DeeplakeRandom().get_seed())
        self.size = size
        self.buffer: List[Any] = list()
        self.buffer_used = 0
        self.num_torch_tensors = 0
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
            num_torch_tensors = self._num_torch_tensors(sample)
            max_tensors = deeplake.constants.MAX_TENSORS_IN_SHUFFLE_BUFFER
            max_tensors_reached = (
                self.num_torch_tensors + num_torch_tensors >= max_tensors
            )
            # fill buffer of not reach limit
            if self.buffer_used + sample_size <= self.size and not max_tensors_reached:
                self.buffer_used += sample_size
                self.num_torch_tensors += num_torch_tensors
                self.pbar.update(sample_size)
                self.buffer.append(sample)
                return None
            elif not self.pbar_closed:
                if max_tensors_reached:
                    always_warn(
                        f"`MAX_TENSORS_IN_SHUFFLE_BUFFER` of {max_tensors} reached. Shuffle buffer will not be filled up to the `buffer_size` limit of {(self.size / MB):.3} MB."
                    )
                self.close_buffer_pbar()

            if buffer_len == 0:
                warnings.warn(
                    f"Buffer size is too small. Sample with size {sample_size} does not fit in buffer of size {self.size}"
                )
                return sample

            # exchange samples with shuffle buffer
            selected = self.random.randrange(buffer_len)
            val = self.buffer[selected]
            self.buffer[selected] = sample

            self.buffer_used += sample_size
            self.buffer_used -= self._sample_size(val)
            self.num_torch_tensors += num_torch_tensors
            self.num_torch_tensors -= self._num_torch_tensors(val)
            return val
        else:
            if not self.pbar_closed:
                self.close_buffer_pbar()
            if buffer_len > 0:
                # return random selection
                selected = self.random.randrange(buffer_len)
                val = self.buffer.pop(selected)
                self.buffer_used -= self._sample_size(val)

                return val
            else:
                return None

    def emtpy(self) -> bool:
        return len(self.buffer) == 0

    def _num_torch_tensors(self, sample):
        try:
            if sys.modules.get("torch"):
                from torch import Tensor as TorchTensor
            else:
                return 0
        except ImportError:
            return 0
        if isinstance(sample, TorchTensor):
            return 1
        elif isinstance(sample, bytes):
            return 0
        elif isinstance(sample, str):
            return 0
        elif isinstance(sample, dict):
            return sum(self._num_torch_tensors(tensor) for tensor in sample.values())
        elif isinstance(sample, Sequence):
            return sum(self._num_torch_tensors(tensor) for tensor in sample)
        else:
            return 0

    def _sample_size(self, sample):
        try:
            if sys.modules.get("torch"):
                from torch import Tensor as TorchTensor
            else:
                TorchTensor = None
        except ImportError:
            TorchTensor = None  # type: ignore

        try:
            if sys.modules.get("tensorflow"):
                from tensorflow import Tensor as TensorflowTensor
            else:
                TensorflowTensor = None
        except ImportError:
            TensorflowTensor = None  # type: ignore

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
        elif TorchTensor is not None and isinstance(sample, TorchTensor):
            return sample.element_size() * reduce(mul, sample.shape, 1)
        elif TensorflowTensor is not None and isinstance(sample, TensorflowTensor):
            return sample.dtype.size * reduce(mul, sample.shape.as_list(), 1)
        elif isinstance(sample, np.ndarray):
            return sample.nbytes
        elif isinstance(sample, Image.Image):
            size = sample.size
            num_pixels = size[0] * size[1]
            if sample.mode == "RGB":
                num_pixels = num_pixels * 3
            elif sample.mode == "RGBA":
                num_pixels = num_pixels * 4
            elif sample.mode == "L":
                num_pixels = num_pixels * 1
            num_bytes = num_pixels * 1  # change according to dtype of tensor later
            return num_bytes
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
