from deeplake.core.version_control.dataset_diff import DatasetDiff


def test_tobytes():
    diff = DatasetDiff()
    diff.tensor_renamed("old1", "newer1")
    diff.tensor_renamed("old2", "\u604f\u7D59")
    diff.tensor_deleted("deleted1")
    diff.tensor_deleted("deleted2")
    diff.tensor_deleted("deleted3")

    assert diff.tobytes() == b"".join(
        [
            False.to_bytes(1, "big"),
            int(2).to_bytes(8, "big"),
            len("old1".encode("utf-8")).to_bytes(8, "big"),
            len("newer1".encode("utf-8")).to_bytes(8, "big"),
            "old1newer1".encode("utf-8"),
            len("old2".encode("utf-8")).to_bytes(8, "big"),
            len("\u604f\u7D59".encode("utf-8")).to_bytes(8, "big"),
            "old2\u604f\u7D59".encode("utf-8"),
            int(3).to_bytes(8, "big"),
            len("deleted1".encode("utf-8")).to_bytes(8, "big"),
            "deleted1".encode("utf-8"),
            len("deleted2".encode("utf-8")).to_bytes(8, "big"),
            "deleted2".encode("utf-8"),
            len("deleted3".encode("utf-8")).to_bytes(8, "big"),
            "deleted3".encode("utf-8"),
        ]
    )
