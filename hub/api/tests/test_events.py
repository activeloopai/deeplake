import hub
import numpy as np


def test_commit_checkout_event(hub_cloud_ds):
    hub_cloud_ds.commit()
    hub_cloud_ds.checkout("abc", create=True)


def test_query_progress_event(hub_cloud_ds):
    with hub_cloud_ds as ds:
        ds.create_tensor("labels")
        ds.labels.append([0])
        ds.labels.append([1])
    ds.commit()
    result = ds.filter("labels == 0", progressbar=False, save_result=True)
    assert len(result) == 1


def test_compute_progress_event(hub_cloud_ds):
    with hub_cloud_ds as ds:
        ds.create_tensor("abc")

    @hub.compute
    def func(sample_in, samples_out):
        samples_out.abc.append(sample_in * np.ones((2, 2)))

    ls = list(range(10))

    func().eval(ls, hub_cloud_ds)
    for i in range(10):
        np.testing.assert_array_equal(ds.abc[i].numpy(), i * np.ones((2, 2)))


# test is empty as pytorch events aren't sent currently
def test_pytorch_progress_event(hub_cloud_ds):
    pass
