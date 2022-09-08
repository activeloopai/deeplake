import hub


def test_tag_tensors(memory_ds):
    ds = memory_ds
    with ds:
        ds.create_tensor("x", htype="tag")
        ds.x.add_tag("a")
        ds.x.add_tag("b")
