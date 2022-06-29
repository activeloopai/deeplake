import hub
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from hub.core.storage.hub_memory_object import HubMemoryObject
from hub.core.tiling.sample_tiles import SampleTiles


class TileEncoder(HubMemoryObject):
    def __init__(self, entries=None, version=None):
        self.is_dirty = False
        self.entries: Dict[int, Tuple[Tuple[int, ...], Tuple[int, ...]]] = entries or {}
        self.version = version or hub.__version__

    def register_sample(self, sample: SampleTiles, idx: int):
        """Registers a new tiled sample into the encoder.

        Args:
            sample: The sample to be registered.
            idx: The global sample index.
        """
        if sample.registered:
            return
        ss: Tuple[int, ...] = sample.sample_shape  # type: ignore
        ts: Tuple[int, ...] = sample.tile_shape
        self.entries[idx] = (ss, ts)
        sample.registered = True
        self.is_dirty = True

    def __delitem__(self, global_sample_index: int):
        del self.entries[global_sample_index]
        keys = sorted(self.entries.keys())
        for k in keys:
            if k > global_sample_index:
                self.entries[k - 1] = self.entries[k]
                del self.entries[k]
        self.is_dirty = True

    def __getitem__(self, global_sample_index: int):
        return self.entries[global_sample_index]

    def __contains__(self, global_sample_index: int):
        """Returns whether the index is present in the tile encoder. Useful for checking if a given sample is tiled."""
        return global_sample_index in self.entries

    def get_tile_shape(self, global_sample_index: int):
        return tuple(self[global_sample_index][1])

    def get_sample_shape(self, global_sample_index: int):
        return tuple(self[global_sample_index][0])

    def get_tile_layout_shape(self, global_sample_index: int) -> Tuple[int, ...]:
        """If you were to lay the tiles out in a grid, the tile layout shape would be the shape
        of the grid.

        Example:
            Sample shape:               (1000, 500)
            Tile shape:                 (10, 10)
            Output tile layout shape:   (100, 50)
        """

        tile_meta = self[global_sample_index]
        tile_shape = tile_meta[1]
        sample_shape = tile_meta[0]

        if len(tile_shape) != len(sample_shape):
            raise ValueError(
                "Tile shape and sample shape must have the same number of dimensions."
            )

        layout = [
            np.ceil(sample_shape_dim / tile_shape_dim)
            for tile_shape_dim, sample_shape_dim in zip(tile_shape, sample_shape)
        ]
        return tuple(int(x) for x in layout)

    @property
    def nbytes(self):
        len_version = 1 + len(self.version)
        entries = self.entries
        num_entries = len(entries)
        if num_entries == 0:
            return len_version + 8
        value = next(iter(entries.values()))
        num_dimensions = len(value[0])
        return len_version + 16 + (num_entries * (8 * (1 + num_dimensions * 2)))

    def tobytes(self) -> memoryview:
        """Serialize entries dict into bytes

        Corresponding to every key, there is a tuple of 2 shapes which are tuples.
        Example of entries is:-
        {
            0: (
                (1000, 500, 100, 3),
                (10, 10, 10, 3)
            ),
            7: (
                (3000, 600, 100, 4),
                (50, 50, 50, 4)
            )
        }
        All the tuples inside will have the same number of dimensions.
        The format stores the following information in order:
        1. First byte tells the length of the version string, call this v.
        2. The next v bytes are the version string.
        3. The next 8 bytes are the number of entries call this n.
        4. The next 8 bytes are the number of dimensions in the shapes, call this d.
        5. The next n * (8 * (1 + d + d)) bytes are the entries in the format:
            a. The first 8 bytes are the key.
            b. The next d * 8 bytes are the first shape in value.
            c. The next d * 8 bytes are the second shape in value.
        """
        entries = self.entries
        num_entries = len(entries)

        data = bytearray(self.nbytes)
        ofs = 0

        # store version length
        data[ofs] = len(self.version)
        ofs += 1

        # store version string
        data[ofs : ofs + len(self.version)] = self.version.encode("ascii")
        ofs += len(self.version)

        # store the number of entries
        data[ofs : ofs + 8] = num_entries.to_bytes(8, byteorder="little")
        ofs += 8

        if num_entries == 0:
            return memoryview(data)

        value = next(iter(entries.values()))
        num_dimensions = len(value[0])

        # store the number of dimensions
        data[ofs : ofs + 8] = num_dimensions.to_bytes(8, byteorder="little")
        ofs += 8

        # store the entries
        for key, value in entries.items():
            # store the key
            data[ofs : ofs + 8] = key.to_bytes(8, byteorder="little")
            ofs += 8

            # store the first shape
            first_shape = value[0]
            for dimension in first_shape:
                data[ofs : ofs + 8] = dimension.to_bytes(8, byteorder="little")
                ofs += 8

            # store the second shape
            second_shape = value[1]
            for dimension in second_shape:
                data[ofs : ofs + 8] = dimension.to_bytes(8, byteorder="little")
                ofs += 8

        return memoryview(data)

    @classmethod
    def frombuffer(cls, data: bytes):
        """Deserialize bytes into entries dict"""
        try:
            ofs = 0
            # Get version length
            version_length = data[ofs]
            ofs += 1

            # Get version string
            version = str(data[ofs : ofs + version_length], "ascii")
            ofs += version_length
            check_version(version)
            entries = parse_tile_encoder_entries(data, ofs, "little")
            enc = cls(entries, version=version)
        except Exception:
            # backwards compatibility
            entries = parse_tile_encoder_entries(data, 0, "big")
            enc = cls(entries)
        enc.is_dirty = False
        return enc

    def __getstate__(self) -> Dict[str, Any]:
        return {"entries": self.entries, "version": self.version}

    def __setstate__(self, state: Dict[str, Any]):
        self.entries = state["entries"]
        self.version = state["version"]


def parse_tile_encoder_entries(data, ofs: int, byteorder: str) -> Dict:
    # Get the number of entries
    num_entries = int.from_bytes(data[ofs : ofs + 8], byteorder=byteorder)  # type: ignore
    ofs += 8
    if num_entries == 0:
        return {}

    # Get the number of dimensions of the tuples
    num_dim = int.from_bytes(data[ofs : ofs + 8], byteorder=byteorder)  # type: ignore
    ofs += 8

    # Get the entries
    entries = {}
    for _ in range(num_entries):
        # Get the key
        key = int.from_bytes(data[ofs : ofs + 8], byteorder=byteorder)  # type: ignore
        ofs += 8

        first_shape: List[int] = []
        second_shape: List[int] = []

        for shp in [first_shape, second_shape]:
            for _ in range(num_dim):
                shp.append(int.from_bytes(data[ofs : ofs + 8], byteorder=byteorder))  # type: ignore
                ofs += 8

        # Add the entry to the dict
        entries[key] = (tuple(first_shape), tuple(second_shape))
    return entries


def check_version(version):
    if len(version) < 5 or len(version) > 7:
        raise ValueError("Invalid version length")
    if version.count(".") < 2 or version[0] != "2":
        raise ValueError("Invalid version format")
