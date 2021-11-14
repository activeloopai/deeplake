from typing import Set
from hub.core.storage.cachable import Cachable


class CommitDiff(Cachable):
    """Stores set of diffs stored for a particular tensor in a commit."""

    def __init__(self, created=False) -> None:
        self.created = created
        self.data_added: Set[int] = set()
        self.data_updated: Set[int] = set()

    def tobytes(self) -> bytes:
        """Returns bytes representation of the commit diff"""
        return b"".join(
            [
                self.created.to_bytes(1, "big"),
                len(self.data_added).to_bytes(4, "big"),
                *[idx.to_bytes(8, "big") for idx in self.data_added],
                len(self.data_updated).to_bytes(4, "big"),
                *[idx.to_bytes(8, "big") for idx in self.data_updated],
            ]
        )

    @classmethod
    def frombuffer(cls, data: bytes) -> "CommitDiff":
        """Creates a CommitDiff object from bytes"""
        commit_diff = cls()
        commit_diff.created = bool(int.from_bytes(data[0:1], "big"))
        data_added_ct = int.from_bytes(data[1:5], "big")
        data_added_indexes = [
            int.from_bytes(data[5 + i * 8 : 5 + (i + 1) * 8], "big")
            for i in range(data_added_ct)
        ]
        commit_diff.data_added = set(data_added_indexes)
        data_updated_ct = int.from_bytes(
            data[5 + data_added_ct * 8 : 9 + data_added_ct * 8], "big"
        )
        data_updated_indexes = [
            int.from_bytes(
                data[
                    9 + data_added_ct * 8 + i * 8 : 9 + data_added_ct * 8 + (i + 1) * 8
                ],
                "big",
            )
            for i in range(data_updated_ct)
        ]
        commit_diff.data_updated = set(data_updated_indexes)
        return commit_diff

    @property
    def nbytes(self):
        """Returns number of bytes required to store the commit diff"""
        return 1 + 4 + len(self.data_added) * 8 + 4 + len(self.data_updated) * 8

    def create_tensor(self) -> None:
        """If the tensor was"""
        self.created = True

    def add_data(self, global_indexes: Set[int]) -> None:
        """Adds new indexes to data added"""
        self.data_added.union(global_indexes)

    def update_data(self, global_index: int) -> None:
        """Adds new indexes to data updated"""
        if global_index not in self.data_added:
            self.data_updated.add(global_index)


def get_sample_indexes_added(initial_num_samples: int, samples) -> Set[int]:
    """Returns a set of indexes added to the tensor"""
    if initial_num_samples == 0:
        return set(range(len(samples)))
    else:
        return set(range(initial_num_samples, initial_num_samples + len(samples)))
