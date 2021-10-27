from typing import List, Any
from random import randrange


class ShuffleBuffer:
    def __init__(self, size) -> None:
        self.size = size
        self.buffer: List[Any] = list()

    def exchange(self, sample):
        buffer_len = len(self.buffer)

        if not sample is None:
            # fill buffer of not reach limit
            if buffer_len < self.size:
                self.buffer.append(sample)
                return None

            # exchange samples with shuffle buffer
            selected = randrange(buffer_len)
            val = self.buffer[selected]
            self.buffer[selected] = sample
            return val
        else:
            if buffer_len > 0:

                # return random selection
                selected = randrange(buffer_len)
                return self.buffer.pop(selected)
            else:
                return None

    def __len__(self):
        return len(self.buffer)
