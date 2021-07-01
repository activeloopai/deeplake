from hub.util.subscript_namedtuple import subscript_namedtuple as namedtuple


def test_namedtuple():
    T = namedtuple("T", ["a", "b", "c", "d"])
    t1 = T(1, 2, 3, 4)
    t2 = T(b=2, a=1, d=4, c=3)
    assert t1["a"] == 1
    assert t1["b"] == 2
    assert t1["c"] == 3
    assert t1["d"] == 4
    assert t2["a"] == 1
    assert t2["b"] == 2
    assert t2["c"] == 3
    assert t2["d"] == 4
    assert t1 == t2
    assert len(t1) == len(t2) == 4
    assert list(t1.keys()) == list(t2.keys()) == ["a", "b", "c", "d"]
    a1, b1, c1, d1 = t1
    a2, b2, c2, d2 = t2
    assert a1 == a2 == 1
    assert b1 == b2 == 2
    assert c1 == c2 == 3
    assert d1 == d2 == 4
