import ast
from pathlib import Path

import pytest


def _extract_len_method(source_path, class_name):
    parsed = ast.parse(source_path.read_text(encoding="utf-8"))
    class_node = next(
        node
        for node in parsed.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "__len__"
    )
    isolated = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(isolated)
    namespace = {"super": super}
    exec(compile(isolated, str(source_path), "exec"), namespace)
    return namespace["__len__"]


@pytest.mark.parametrize(
    "relative_path,class_name",
    [
        ("mmdet/mmdet_dataset_.py", "MMDetDataset"),
        ("mmseg/mmseg_dataset_.py", "MMSegDataset"),
    ],
)
def test_validation_length_counts_all_samples_for_multi_sample_batches(
    relative_path, class_name
):
    source = (
        Path(__file__).resolve().parents[1]
        / "deeplake"
        / "integrations"
        / relative_path
    )
    get_length = _extract_len_method(source, class_name)

    class Dataset:
        mode = "val"
        batch_size = 2
        num_gpus = 1

        def __init__(self, length):
            self.dataset = range(length)

        def __len__(self):
            return len(self.dataset)

    dataset = Dataset(8)

    assert get_length(dataset) == 8
