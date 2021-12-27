import hub
from hub.core.version_control.commit_node import CommitNode
from hub.constants import FIRST_COMMIT_ID
from hub.util.version_control import _merge_commit_node_maps


def test_merge_commit_node_map():

    root = CommitNode("main", FIRST_COMMIT_ID)
    a = CommitNode("main", "a")
    b = CommitNode("main", "b")
    c = CommitNode("main", "c")
    e = CommitNode("main", "e")
    root.add_successor(a, "commit a")
    root.add_successor(b, "commit b")
    a.add_successor(c, "commit c")
    c.add_successor(e, "commit e")
    map1 = {
        FIRST_COMMIT_ID: root,
        "a": a,
        "b": b,
        "c": c,
        "e": e,
    }

    root = CommitNode("main", FIRST_COMMIT_ID)
    a = CommitNode("main", "a")
    b = CommitNode("main", "b")
    d = CommitNode("main", "d")
    f = CommitNode("main", "f")
    root.add_successor(a, "commit a")
    root.add_successor(b, "commit b")
    b.add_successor(d, "commit d")
    d.add_successor(f, "commit f")

    map2 = {
        FIRST_COMMIT_ID: root,
        "a": a,
        "b": b,
        "d": d,
        "f": f,
    }

    merged = _merge_commit_node_maps(map1, map2)

    assert set(merged.keys()) == set((FIRST_COMMIT_ID, "a", "b", "c", "d", "e", "f"))
    get_children = lambda node: set(c.commit_id for c in node.children)
    assert get_children(merged[FIRST_COMMIT_ID]) == set(("a", "b"))
    assert get_children(merged["a"]) == set(("c"))
    assert get_children(merged["b"]) == set(("d"))
    assert get_children(merged["c"]) == set(("e"))
    assert get_children(merged["d"]) == set(("f"))
