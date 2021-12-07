from typing import Set
from hub.core.storage.cachable import Cachable


class CommitDiff(Cachable):
    """Stores set of diffs stored for a particular tensor in a commit."""

    def __init__(self, created=False) -> None:
        self.created = created
        self.data_added: Set[int] = set()
        self.data_updated: Set[int] = set()

    def tobytes(self) -> bytes:
        """Returns bytes representation of the commit diff

        The format stores the following information in order:
        1. The first byte is a boolean value indicating whether the tensor was created in the commit or not.
        2. The next 8 bytes are the number of elements in the data_added set, let's call this n.
        3. The next 8 * n bytes are the elements of the data_added set.
        4. The next 8 bytes are the number of elements in the data_updated set, let's call this m.
        5. The next 8 * m bytes are the elements of the data_updated set.
        """
        return b"".join(
            [
                self.created.to_bytes(1, "big"),
                len(self.data_added).to_bytes(8, "big"),
                *[idx.to_bytes(8, "big") for idx in self.data_added],
                len(self.data_updated).to_bytes(8, "big"),
                *[idx.to_bytes(8, "big") for idx in self.data_updated],
            ]
        )

    @classmethod
    def frombuffer(cls, data: bytes) -> "CommitDiff":
        """Creates a CommitDiff object from bytes"""
        commit_diff = cls()

        commit_diff.created = bool(int.from_bytes(data[0:1], "big"))

        added_ct = int.from_bytes(data[1:9], "big")
        commit_diff.data_added = {
            int.from_bytes(data[9 + i * 8 : 9 + (i + 1) * 8], "big")
            for i in range(added_ct)
        }

        updated_ct = int.from_bytes(data[9 + added_ct * 8 : 17 + added_ct * 8], "big")
        offset = 17 + added_ct * 8
        commit_diff.data_updated = {
            int.from_bytes(data[offset + i * 8 : offset + (i + 1) * 8], "big")
            for i in range(updated_ct)
        }

        return commit_diff

    @property
    def nbytes(self):
        """Returns number of bytes required to store the commit diff"""
        return 17 + (len(self.data_added) + len(self.data_updated)) * 8

    def add_data(self, global_indexes: Set[int]) -> None:
        """Adds new indexes to data added"""
        self.data_added.update(global_indexes)

    def update_data(self, global_index: int) -> None:
        """Adds new indexes to data updated"""
        if global_index not in self.data_added:
            self.data_updated.add(global_index)


def get_sample_indexes_added(initial_num_samples: int, samples) -> Set[int]:
    """Returns a set of indexes added to the tensor"""
    return set(range(initial_num_samples, initial_num_samples + len(samples)))
