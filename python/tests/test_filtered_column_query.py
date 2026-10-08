"""Exercise sparse filters through real columns and nested dataset queries.

Run with the native extension built from this checkout. A published wheel can
validate the fixture/API, but cannot validate changes to the C++ header here.
"""

import deeplake
import pytest


@pytest.fixture
def reordered_view(tmp_path):
    """Create an indexed column and a view with source mapping [9, 2, 7]."""
    dataset = deeplake.create(str(tmp_path / "filtered-column"))
    dataset.add_column("id", deeplake.types.Int32())
    dataset.add_column("rank", deeplake.types.Int32())
    dataset.add_column("embedding", deeplake.types.Embedding(size=2))
    dataset.append(
        [
            {
                "id": index,
                "rank": {9: 0, 2: 1, 7: 2}.get(index, 10),
                "embedding": [1.0, float(index)],
            }
            for index in range(10)
        ]
    )
    dataset.commit()
    view = dataset.query("SELECT * WHERE rank < 3 ORDER BY rank")
    assert [int(row["id"]) for row in view] == [9, 2, 7]
    return view


@pytest.mark.parametrize("source_id", [2, 7])
def test_sparse_filter_after_first_view_position(reordered_view, source_id):
    """One selected row may occupy view position one or two, not position zero."""
    result = reordered_view.query(
        f"SELECT * WHERE id = {source_id} "
        f"ORDER BY COSINE_SIMILARITY(embedding, ARRAY[1.0, {source_id}.0]) "
        "DESC LIMIT 1"
    )
    assert [int(row["id"]) for row in result] == [source_id]


def test_top_k_without_additional_filter(reordered_view):
    """Mapping a source hit back to the view preserves top-k ordering."""
    result = reordered_view.query(
        "SELECT * ORDER BY COSINE_SIMILARITY(embedding, ARRAY[1.0, 7.0]) "
        "DESC LIMIT 1"
    )
    assert [int(row["id"]) for row in result] == [7]
