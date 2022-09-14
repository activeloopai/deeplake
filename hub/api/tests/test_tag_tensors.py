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
