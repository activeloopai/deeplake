import numpy as np
import hub


def test_tag_tensors(memory_ds):
    ds = memory_ds
    with ds:
        ds.create_tensor("x", htype="tag")
        assert ds.x.is_tag_tensor
        ds.x.add_tag("a")
        ds.x.add_tag("b")
        ds.x.a.append(1)
        ds.x.b.append(1)
        ds.x.a.append(2)
        ds.x.b.append(3)
        assert ds.x._tags == ["default", "a", "b"]
        assert ds.x.a[0].numpy() == ds.x.b[0].numpy()
        assert ds.x.a[1].numpy() == ds.x.b[1].numpy() - 1
        ds.x[0].sample_default_tag = "a"
        ds.x[1].sample_default_tag = "b"
        assert ds.x[0].sample_default_tag == "a"
        assert ds.x[1].sample_default_tag == "b"
        assert ds.x.sample_default_tag == ["a", "b"]
        np.testing.assert_array_equal(ds.x.numpy(), np.array([[1], [3]]))
        assert ds.x.is_tag_tensor
        assert ds.x.default_tag == "default"
        materialized = ds.x.materialize("materialized")
        assert ds.x.is_tag_tensor, (ds.x.group_index, ds.x.meta.tag_tensors)
        np.testing.assert_array_equal(ds.x.numpy(), materialized.numpy())
        ds.create_tensor("y", htype="tag")
        ds.y.add_tag("a")
        ds.y.add_tag("b")
        ds.y.add_tag("c")
        ds.y.a[0] = 0
        ds.y.b[0] = 1
        ds.y.c[0] = 2
        ds.y.a[1] = 2
        ds.y.b[1] = 1
        ds.y.c[1] = 0
        ds.y.a[2] = 0
        ds.y.b[2] = 2
        ds.y.c[2] = 1
        aggr_fn = lambda x: np.max([x.a, x.b, x.c])
        ds.y.materialize("d", aggr_fn)
        np.testing.assert_array_equal(ds.y.d, np.ones((3, 1)) * 2)
