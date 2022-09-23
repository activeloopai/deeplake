from deeplake.util.iterable_ordered_dict import IterableOrderedDict as OrderedDict
import pickle


def test_ordereddict():
    T = OrderedDict
    t1 = T()
    t1["b"] = 2
    t1["a"] = 1
    t1["d"] = 4
    t1["c"] = 3

    t2 = T(b=2, a=1, d=4, c=3)

    assert t1 == t2
    assert t1["b"] == 2
    assert t1["c"] == 3
    assert t1["d"] == 4
    assert t2["a"] == 1
    assert t2["b"] == 2
    assert t2["c"] == 3
    assert t2["d"] == 4
    assert t1 == t2
    assert len(t1) == len(t2) == 4
    assert list(t1.keys()) == list(t2.keys()) == ["b", "a", "d", "c"]
    b1, a1, d1, c1 = t1
    b2, a2, d2, c2 = t2
    assert a1 == a2 == 1
    assert b1 == b2 == 2
    assert c1 == c2 == 3
    assert d1 == d2 == 4

    pickled = pickle.dumps(t2)
    t2 = pickle.loads(pickled)

    assert t1 == t2
    assert t1["b"] == 2
    assert t1["c"] == 3
    assert t1["d"] == 4
    assert t2["a"] == 1
    assert t2["b"] == 2
    assert t2["c"] == 3
    assert t2["d"] == 4
    assert t1 == t2
    assert len(t1) == len(t2) == 4
    assert list(t1.keys()) == list(t2.keys()) == ["b", "a", "d", "c"]
    b1, a1, d1, c1 = t1
    b2, a2, d2, c2 = t2
    assert a1 == a2 == 1
    assert b1 == b2 == 2
    assert c1 == c2 == 3
    assert d1 == d2 == 4
