"""
License:
This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
"""

import numpy as np
import hub
from hub.schema import Tensor, Image, Text, Sequence, SchemaDict, BBox
import pytest


def test_objectview():
    schema = SchemaDict(
        {
            "a": Tensor((20, 20), dtype=int, max_shape=(20, 20)),
            "b": Sequence(dtype=BBox(dtype=float)),
            "c": Sequence(
                dtype=SchemaDict({"d": Sequence((), dtype=Tensor((5, 5), dtype=float))})
            ),
            "e": Sequence(
                dtype={"f": {"g": Tensor(5, dtype=int), "h": Tensor((), dtype=int)}}
            ),
        }
    )
    ds = hub.Dataset("./nested_seq", shape=(5,), mode="w", schema=schema)

    # dataset view to objectview
    dv = ds[3:5]
    dv["c", 0] = {"d": 5 * np.ones((2, 2, 5, 5))}
    assert (dv[0, "c", 0, "d", 0].compute() == 5 * np.ones((5, 5))).all()

    # dataset view unsqueezed
    with pytest.raises(IndexError):
        dv["c", "d"].compute()

    # dataset unsqueezed
    with pytest.raises(IndexError):
        ds["c", "d"].compute()

    # tensorview to object view
    # sequence of tensor
    ds["b", 0] = 0.5 * np.ones((5, 4))
    ds["b", 0] = 0.3 * np.ones((4,))
    assert (ds["b", 0].compute() == 0.3 * np.ones((4,))).all()

    # ds to object view
    assert (ds[3, "c", "d"].compute() == 5 * np.ones((2, 2, 5, 5))).all()

    # Sequence of schemadicts
    ds[0, "e"] = {"f": {"g": np.ones((3, 5)), "h": np.array([42, 25, 15])}}
    with pytest.raises(KeyError):
        ds[0, "e", 1].compute()
    assert (ds[0, "e", "f", "h"].compute() == np.array([42, 25, 15])).all()

    # With dataset view
    dv[0, "e"] = {"f": {"g": np.ones((3, 5)), "h": np.array([1, 25, 1])}}
    # dv[0, "e", 1]["f", "h"] = 25
    assert (dv[0, "e", "f", "h"].compute() == np.array([1, 25, 1])).all()

    # If not lazy mode all slices should be stable
    ds.lazy = False
    assert ds[0, "e", 0, "f", "h"] == 42
    with pytest.raises(KeyError):
        ds[0, "e", 1]["f", "h"] == 25
    ds.lazy = True

    # make an objectview
    ov = ds["c", "d"]
    with pytest.raises(IndexError):
        ov.compute()
    assert (ov[3].compute() == 5 * np.ones((2, 2, 5, 5))).all()
    # ov[3, 1] = 2 * np.ones((2, 5, 5))
    assert (ov[3][0, 0].compute() == 5 * np.ones((5, 5))).all()
    assert (ov[3][1].compute() == 5 * np.ones((2, 5, 5))).all()


def test_errors():
    schema = SchemaDict(
        {
            "a": Tensor((None, None), dtype=int, max_shape=(20, 20)),
            "b": Sequence(
                dtype=SchemaDict(
                    {"e": Tensor((None,), max_shape=(10,), dtype=BBox(dtype=float))}
                )
            ),
            "c": Sequence(
                dtype=SchemaDict({"d": Sequence((), dtype=Tensor((5, 5), dtype=float))})
            ),
        }
    )
    ds = hub.Dataset("./nested_seq", shape=(5,), mode="w", schema=schema)

    # Invalid schema
    with pytest.raises(ValueError):
        ds["b", 0, "e", 1]

    # Too many indices
    with pytest.raises(IndexError):
        ds["c", 0, "d", 1, 1, 0, 0, 0]
    with pytest.raises(IndexError):
        ds["c", :2, "d"][0, 1, 1, 0, 0, 0]
    ob = ds["c", :2, "d"][0, 2:5, 1, 0, 0]
    assert str(ob[1]) == "ObjectView(subpath='/c/d', indexes=0, slice=[3, 1, 0, 0])"
    with pytest.raises(IndexError):
        ob[1, 0]

    # Key Errors
    # wrong key
    with pytest.raises(KeyError):
        ds["b", "c"]
    # too many keys
    with pytest.raises(KeyError):
        ds["c", "d", "e"]
    with pytest.raises(KeyError):
        ds["c", "d"]["e"]


if __name__ == "__main__":
    test_objectview()
    test_errors()
