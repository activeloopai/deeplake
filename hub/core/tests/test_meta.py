import pytest
from hub.core.meta.index_map import IndexMap


@pytest.mark.parametrize(
    "raw_entries",
    [
        [
            {
                "chunk_names": [
                    "e1a3f688-c2c3-11eb-8355-a45e60ec2e70",
                    "afsdafadsfasdfsfsdfsdds",
                ],
                "start_byte": 10,
                "end_byte": 102,
                "shape": None,
            },
            {
                "chunk_names": [],
                "start_byte": 0,
                "end_byte": 0,
                "shape": None,
            },
        ],
        [
            {
                "chunk_names": [
                    "e1a3f688-c2c3-11eb-8355-a45e60ec2e70",
                    "afsdafadsfasdfsfsdfsdds",
                ],
                "start_byte": 10,
                "end_byte": 102,
                "shape": (),
            },
            {
                "chunk_names": [0],
                "start_byte": 1,
                "end_byte": 102,
                "shape": None,
            },
            {
                "chunk_names": ["1010101010"],
                "start_byte": 1,
                "end_byte": 102,
                "shape": (0,),
            },
        ],
        [
            {
                "chunk_names": [],
                "start_byte": 0,
                "end_byte": 102,
                "shape": (1,),
            },
            {
                "chunk_names": [""],
                "start_byte": 1,
                "end_byte": 102,
                "shape": None,
            },
            {
                "chunk_names": ["1010101010"],
                "start_byte": 1,
                "end_byte": 102,
                "shape": (0,),
            },
        ],
    ],
)
def test_index_map(
    memory_storage,
    raw_entries,
):
    key = "index_map"
    index_map = IndexMap(key, memory_storage)
    for raw_entry in raw_entries:
        index_map.create_entry(**raw_entry)

    index_map2 = IndexMap(key, memory_storage)
    for i, raw_entry in enumerate(raw_entries):
        assert index_map2[i].asdict() == raw_entry