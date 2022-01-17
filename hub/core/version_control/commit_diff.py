from typing import Set, List
from hub.core.storage.cachable import Cachable


class CommitDiff(Cachable):
    """Stores set of diffs stored for a particular tensor in a commit."""

    def __init__(
        self, first_index=0, created=False, renamed=False, name_history=[]
    ) -> None:
        self.created = created
        self.renamed = renamed
        self.name_history = name_history
        self.data_added: List[int] = [first_index, first_index]
        self.data_updated: Set[int] = set()
        self.info_updated = False

        # this is stored for in place transforms in which we no longer need to considered older diffs about added/updated data
        self.data_transformed = False

    def tobytes(self) -> bytes:
        """Returns bytes representation of the commit diff

        The format stores the following information in order:
        1. The first byte is a boolean value indicating whether the tensor was created in the commit or not.
        2. The second byte is a boolean value indicating whether the info has been updated or not.
        3. The third byte is a boolean value indicating whether the data has been transformed using an inplace transform or not.
        4. The next 8 + 8 bytes are the two elements of the data_added list.
        5. The next 8 bytes are the number of elements in the data_updated set, let's call this m.
        6. The next 8 * m bytes are the elements of the data_updated set.
        """
        return b"".join(
            [
                self.created.to_bytes(1, "big"),
                self.renamed.to_bytes(1, "big"),
                self.info_updated.to_bytes(1, "big"),
                self.data_transformed.to_bytes(1, "big"),
                len(self.name_history).to_bytes(1, "big"),
                self.data_added[0].to_bytes(8, "big"),
                self.data_added[1].to_bytes(8, "big"),
                len(self.data_updated).to_bytes(8, "big"),
                *(idx.to_bytes(8, "big") for idx in self.data_updated),
                *(len(name).to_bytes(8, "big") for name in self.name_history),
                *(name.encode("utf-8") for name in self.name_history),
            ]
        )

    @classmethod
    def frombuffer(cls, data: bytes) -> "CommitDiff":
        """Creates a CommitDiff object from bytes"""
        commit_diff = cls()

        commit_diff.created = bool(int.from_bytes(data[:1], "big"))
        commit_diff.renamed = bool(int.from_bytes(data[1:2], "big"))
        commit_diff.info_updated = bool(int.from_bytes(data[2:3], "big"))
        commit_diff.data_transformed = bool(int.from_bytes(data[3:4], "big"))
        num_names = int.from_bytes(data[4:5])
        commit_diff.data_added = [
            int.from_bytes(data[5:13], "big"),
            int.from_bytes(data[13:21], "big"),
        ]
        num_updates = int.from_bytes(data[21:29], "big")
        commit_diff.data_updated = {
            int.from_bytes(data[29 + i * 8 : 37 + i * 8], "big")
            for i in range(num_updates)
        }
        pos = 37 + num_updates * 8
        len_names = [
            int.from_bytes(data[pos + i * 8 : pos + (i + 1) * 8])
            for i in range(num_names)
        ]
        pos += num_names * 8
        commit_diff.name_history.extend(
            [
                data[pos + sum(len_names[:i]) : pos + sum(len_names[: i + 1])].decode(
                    "utf-8"
                )
                for i in range(num_names)
            ]
        )

        return commit_diff

    @property
    def nbytes(self):
        """Returns number of bytes required to store the commit diff"""
        return (
            29
            + 8 * len(self.data_updated)
            + 8 * len(self.name_history)
            + sum(len(name) for name in self.name_history)
        )

    @property
    def num_samples_added(self) -> int:
        """Returns number of samples added"""
        return self.data_added[1] - self.data_added[0]

    def modify_info(self) -> None:
        """Stores information that the info has changed"""
        self.info_updated = True

    def add_data(self, count: int) -> None:
        """Adds new indexes to data added"""
        self.data_added[1] += count

    def update_data(self, global_index: int) -> None:
        """Adds new indexes to data updated"""
        if global_index not in self.data_added:
            self.data_updated.add(global_index)

    def transform_data(self) -> None:
        """Stores information that the data has been transformed using an inplace transform."""
        self.data_transformed = True

    def _pop(self) -> None:
        """Remove index for the last data added. Used by ChunkEngine._pop()"""
        if self.data_added[1] == self.data_added[0]:
            raise NotImplementedError(
                "Cannot pop sample which was added in a previous commit."
            )
        self.data_added[1] -= 1
