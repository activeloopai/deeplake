from hub.core.query.autocomplete import autocomplete
import hub


def _test_ds():
    ds = hub.dataset("mem://x")
    ds.create_tensor("a")
    ds.create_tensor("b")
    ds.create_tensor("c")
    ds.create_tensor("def")
    ds.create_tensor("g/h/i")
    ds.create_tensor("j/k/l")
    return ds


def test_empty_query():
    ds = _test_ds()
    q = ""
    resp = autocomplete(q, ds)
    assert resp["tokens"] == []
    suggestions = [s["string"] for s in resp["suggestions"]]
    assert suggestions == ["a", "b", "c", "def", "g", "j"]
    assert resp["replace"] == ""


def test_tensor_name():
    ds = _test_ds()
    for q in ["a", "b", "c", "def"]:
        resp = autocomplete(q, ds)
        tokens = resp["tokens"]
        assert len(tokens) == 1
        token = tokens[0]
        assert token["string"] == q
        assert token["start"] == 0
        assert token["end"] == len(q)
        assert token["type"] == "TENSOR"
        suggestions = resp["suggestions"]
        assert resp["replace"] == ""
        assert suggestions == [
            {"string": ".contains", "type": "METHOD"},
            {"string": ".max", "type": "PROPERTY"},
            {"string": ".mean", "type": "PROPERTY"},
            {"string": ".min", "type": "PROPERTY"},
            {"string": ".shape", "type": "PROPERTY"},
            {"string": ".size", "type": "PROPERTY"},
            {"string": " ==", "type": "OP"},
            {"string": " >", "type": "OP"},
            {"string": " <", "type": "OP"},
            {"string": " >=", "type": "OP"},
            {"string": " <=", "type": "OP"},
            {"string": " !=", "type": "OP"},
        ]


def test_tensor_name_partial():
    ds = _test_ds()
    for q in ["d", "de"]:
        resp = autocomplete(q, ds)
        tokens = resp["tokens"]
        assert len(tokens) == 1
        token = tokens[0]
        assert token["string"] == q, (q, token)
        assert token["start"] == 0
        assert token["end"] == len(q)
        assert token["type"] == "UNKNOWN"
        assert resp["replace"] == q
        suggestions = resp["suggestions"]
        assert suggestions == [{"string": "def", "type": "TENSOR"}]


def test_group():
    ds = _test_ds()
    q = "g"
    resp = autocomplete(q, ds)
    suggestions = resp["suggestions"]
    assert suggestions == [
        {
            "string": ".h",
            "type": "GROUP",
        }
    ]
    q = "g.h"
    resp = autocomplete(q, ds)
    suggestions = resp["suggestions"]
    assert suggestions == [
        {
            "string": ".i",
            "type": "TENSOR",
        }
    ]

    q = "g.h.i"
    resp = autocomplete(q, ds)
    suggestions = resp["suggestions"]
    assert suggestions == [
        {"string": ".contains", "type": "METHOD"},
        {"string": ".max", "type": "PROPERTY"},
        {"string": ".mean", "type": "PROPERTY"},
        {"string": ".min", "type": "PROPERTY"},
        {"string": ".shape", "type": "PROPERTY"},
        {"string": ".size", "type": "PROPERTY"},
        {"string": " ==", "type": "OP"},
        {"string": " >", "type": "OP"},
        {"string": " <", "type": "OP"},
        {"string": " >=", "type": "OP"},
        {"string": " <=", "type": "OP"},
        {"string": " !=", "type": "OP"},
    ]

    q = "g.h.k"
    resp = autocomplete(q, ds)
    assert resp["suggestions"] == []


def test_keyword():
    ds = _test_ds()
    q = "'x' in "
    resp = autocomplete(q, ds)
    suggestions = [s["string"] for s in resp["suggestions"]]
    assert "a" in suggestions
    assert "b" in suggestions
    assert "c" in suggestions
    assert "def" in suggestions
    assert "g" in suggestions
    assert "j" in suggestions


def test_const():
    ds = _test_ds()
    for q in ["True", "'abcd'", "123"]:
        resp = autocomplete(q, ds)
        assert resp["suggestions"] == []


def test_property():
    ds = _test_ds()
    q = "g.h.i.mean"
    resp = autocomplete(q, ds)
    suggestions = [s["string"] for s in resp["suggestions"]]
    assert " ==" in suggestions


def test_method():
    ds = _test_ds()
    q = "g.h.i.contains"
    resp = autocomplete(q, ds)
    suggestions = [s["string"] for s in resp["suggestions"]]
    assert "(" in suggestions
