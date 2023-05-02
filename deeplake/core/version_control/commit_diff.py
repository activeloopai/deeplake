from typing import Set, List
from deeplake.core.storage.deeplake_memory_object import DeepLakeMemoryObject


class CommitDiff(DeepLakeMemoryObject):
    """Stores set of diffs stored for a particular tensor in a commit."""

    def __init__(self, first_index=0, created=False) -> None:
        self.is_dirty = created  # only put as dirty during init if created
        self.created = created
        self.data_added: List[int] = [first_index, first_index]
        self.data_updated: Set[int] = set()
        self.data_deleted: Set[int] = set()
        self.data_deleted_ids: Set[int] = set()
        self.info_updated = False
        self.cleared = False

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
        7. The next byte is a boolean value indicating whether the tensor was cleared in the commit or not.
        8. The next 8 bytes are the number of elements in the data_deleted set, let's call this n.
        9. The next 8 * n bytes are the elements of the data_deleted set.
        9. The next 8 * n bytes are the elements of the data_deleted_ids set.
        """
        return b"".join(
            [
                self.created.to_bytes(1, "big"),
                self.info_updated.to_bytes(1, "big"),
                self.data_transformed.to_bytes(1, "big"),
                self.data_added[0].to_bytes(8, "big"),
                self.data_added[1].to_bytes(8, "big"),
                len(self.data_updated).to_bytes(8, "big"),
                *(idx.to_bytes(8, "big") for idx in self.data_updated),
                self.cleared.to_bytes(1, "big"),
                len(self.data_deleted).to_bytes(8, "big"),
                *(idx.to_bytes(8, "big") for idx in self.data_deleted),
                *(idx.to_bytes(8, "big") for idx in self.data_deleted_ids),
            ]
        )

    @classmethod
    def frombuffer(cls, data: bytes) -> "CommitDiff":
        """Creates a CommitDiff object from bytes"""
        commit_diff = cls()

        commit_diff.created = bool(int.from_bytes(data[:1], "big"))
        commit_diff.info_updated = bool(int.from_bytes(data[1:2], "big"))
        commit_diff.data_transformed = bool(int.from_bytes(data[2:3], "big"))
        commit_diff.data_added = [
            int.from_bytes(data[3:11], "big"),
            int.from_bytes(data[11:19], "big"),
        ]
        num_updates = int.from_bytes(data[19:27], "big")
        commit_diff.data_updated = {
            int.from_bytes(data[27 + i * 8 : 35 + i * 8], "big")
            for i in range(num_updates)
        }
        pos = 35 + (num_updates - 1) * 8
        commit_diff.cleared = bool(int.from_bytes(data[pos : pos + 1], "big"))
        commit_diff.is_dirty = False
        pos += 1
        commit_diff.data_deleted = set()
        commit_diff.data_deleted_ids = set()
        if len(data) > pos:
            num_deletes = int.from_bytes(data[pos : pos + 8], "big")
            pos += 8
            commit_diff.data_deleted = {
                int.from_bytes(data[pos + i * 8 : pos + i * 8 + 8], "big")
                for i in range(num_deletes)
            }
            pos += num_deletes * 8
            if len(data) > pos:
                commit_diff.data_deleted_ids = {
                    int.from_bytes(data[pos + i * 8 : pos + i * 8 + 8], "big")
                    for i in range(num_deletes)
                }
        return commit_diff

    @property
    def nbytes(self):
        """Returns number of bytes required to store the commit diff"""
        return 36 + 8 * (len(self.data_updated) + len(self.data_deleted))

    @property
    def num_samples_added(self) -> int:
        """Returns number of samples added"""
        return self.data_added[1] - self.data_added[0]

    def modify_info(self) -> None:
        """Stores information that the info has changed"""
        self.info_updated = True
        self.is_dirty = True

    def add_data(self, count: int) -> None:
        """Adds new indexes to data added"""
        self.data_added[1] += count
        self.is_dirty = True

    def update_data(self, global_index: int) -> None:
        """Adds new indexes to data updated"""
        global_index = self.translate_index(global_index)
        if global_index not in range(*self.data_added):
            self.data_updated.add(global_index)
            self.is_dirty = True

    def clear_data(self):
        """Clears data"""
        self.data_added = [0, 0]
        self.data_updated = set()
        self.data_deleted = set()
        self.info_updated = False
        self.cleared = True
        self.is_dirty = True

    def transform_data(self) -> None:
        """Stores information that the data has been transformed using an inplace transform."""
        self.data_transformed = True
        self.is_dirty = True

    def pop(self, index, id) -> None:
        index = self.translate_index(index)
        if index not in range(*self.data_added):
            self.data_deleted.add(index)
            self.data_added[0] -= 1
        self.data_added[1] -= 1

        if index in self.data_updated:
            self.data_updated.remove(index)

        self.data_updated = {
            idx - 1 if idx > index else idx for idx in self.data_updated
        }
        if id is not None:
            self.data_deleted_ids.add(id)
        self.is_dirty = True

    def translate_index(self, index):
        if not self.data_deleted:
            return index
        offset = sum(i < index for i in self.data_deleted)
        return index + offset
