import ast
from pathlib import Path

import numpy as np
import pytest


class InvalidSegmentError(Exception):
    def __init__(self, column_name, error):
        super().__init__(f"Error on {column_name} data getting: {error}")


def _load_get_gt_seg_maps():
    source = (
        Path(__file__).resolve().parents[1]
        / "deeplake"
        / "integrations"
        / "mmseg"
        / "mmseg_dataset_.py"
    )
    parsed = ast.parse(source.read_text(encoding="utf-8"))
    dataset_node = next(
        node
        for node in parsed.body
        if isinstance(node, ast.ClassDef) and node.name == "MMSegDataset"
    )
    method = next(
        node
        for node in dataset_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_gt_seg_maps"
    )
    isolated = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(isolated)
    namespace = {
        "InvalidSegmentError": InvalidSegmentError,
        "upcast_array": lambda value: value,
    }
    exec(compile(isolated, str(source), "exec"), namespace)
    return namespace["get_gt_seg_maps"]


@pytest.mark.parametrize("failing_index", [0, 1])
def test_invalid_ground_truth_mask_fails_evaluation_instead_of_misalignment(
    failing_index,
):
    get_gt_seg_maps = _load_get_gt_seg_maps()

    class Masks:
        def __getitem__(self, index):
            if index == failing_index:
                raise OSError("mask read failed")
            return np.array([index])

    class Dataset:
        masks_tensor_name = "masks"

        def __len__(self):
            return 3

        def _get_masks(self, name):
            assert name == "masks"
            return Masks()

    with pytest.raises(InvalidSegmentError, match="masks data getting"):
        list(get_gt_seg_maps(Dataset()))
