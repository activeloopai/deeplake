import hub
import pytest
import numpy as np
from click.testing import CliRunner
from hub.core.storage.memory import MemoryProvider
from hub.util.remove_cache import remove_memory_cache
from hub.util.check_installation import ray_installed
from hub.util.exceptions import InvalidOutputDatasetError, TransformError
from hub.tests.common import parametrize_num_workers
from hub.util.transform import get_pbar_description
import hub


# github actions can only support 2 workers
TRANSFORM_TEST_NUM_WORKERS = 2

all_compressions = pytest.mark.parametrize("sample_compression", [None, "png", "jpeg"])

schedulers = ["threaded", "processed"]
schedulers = schedulers + ["ray"] if ray_installed() else schedulers
all_schedulers = pytest.mark.parametrize("scheduler", schedulers)
commit_or_not = pytest.mark.parametrize("do_commit", [True, False])


@hub.compute
def fn1(sample_in, samples_out, mul=1, copy=1):
    for _ in range(copy):
        samples_out.image.append(np.ones((337, 200)) * sample_in * mul)
        samples_out.label.append(np.ones((1,)) * sample_in * mul)


@hub.compute
def fn2(sample_in, samples_out, mul=1, copy=1):
    for _ in range(copy):
        samples_out.image.append(sample_in.image.numpy() * mul)
        samples_out.label.append(sample_in.label.numpy() * mul)


@hub.compute
def fn3(sample_in, samples_out, mul=1, copy=1):
    for _ in range(copy):
        samples_out.image.append(np.ones((1310, 2087)) * sample_in * mul)
        samples_out.label.append(np.ones((13,)) * sample_in * mul)


@hub.compute
def fn4(sample_in, samples_out):
    samples_out.image.append(sample_in.image)
    samples_out.image.append(sample_in.image.numpy() * 2)
    samples_out.label.append(sample_in.label)
    samples_out.label.append(sample_in.label.numpy() * 2)


@hub.compute
def fn5(sample_in, samples_out, mul=1, copy=1):
    for _ in range(copy):
        samples_out.x["y"].z.image.append(sample_in.z.y.x.image.numpy() * mul)
        samples_out.x.y.z["label"].append(sample_in.z.y.x.label.numpy() * mul)


@hub.compute
def fn6(sample_in, samples_out, mul=1, copy=1):
    for _ in range(copy):
        samples_out.append(
            {
                "image": sample_in.image.numpy() * mul,
                "label": sample_in.label.numpy() * mul,
            }
        )


@hub.compute
def read_image(sample_in, samples_out):
    samples_out.image.append(hub.read(sample_in))


@hub.compute
def crop_image(sample_in, samples_out, copy=1):
    for _ in range(copy):
        samples_out.image.append(sample_in.image.numpy()[:100, :100, :])


@hub.compute
def filter_tr(sample_in, sample_out):
    if sample_in % 2 == 0:
        sample_out.image.append(sample_in * np.ones((100, 100)))


@hub.compute
def populate_cc_bug(sample_in, samples_out):
    samples_out.xyz.append(sample_in)
    return samples_out


@hub.compute
def inplace_transform(sample_in, samples_out):
    samples_out.img.append(2 * sample_in.img.numpy())
    samples_out.img.append(3 * sample_in.img.numpy())
    samples_out.label.append(2 * sample_in.label.numpy())
    samples_out.label.append(3 * sample_in.label.numpy())


def check_target_array(ds, index, target):
    np.testing.assert_array_equal(
        ds.img[index].numpy(), target * np.ones((200, 200, 3))
    )
    np.testing.assert_array_equal(ds.label[index].numpy(), target * np.ones((1,)))


@all_schedulers
@pytest.mark.parametrize(
    "ds",
    ["memory_ds", "local_ds", "s3_ds"],
    indirect=True,
)
def test_single_transform_hub_dataset(ds, scheduler):
    data_in = hub.dataset("./test/single_transform_hub_dataset", overwrite=True)
    with data_in:
        data_in.create_tensor("image")
        data_in.create_tensor("label")
        for i in range(1, 100):
            data_in.image.append(i * np.ones((i, i)))
            data_in.label.append(i * np.ones((1,)))
    ds_out = ds
    ds_out.create_tensor("image")
    ds_out.create_tensor("label")
    if (
        isinstance(remove_memory_cache(ds.storage), MemoryProvider)
        and scheduler != "threaded"
    ):
        # any scheduler other than `threaded` will not work with a dataset stored in memory
        with pytest.raises(InvalidOutputDatasetError):
            fn2(copy=1, mul=2).eval(
                data_in,
                ds_out,
                num_workers=TRANSFORM_TEST_NUM_WORKERS,
                progressbar=False,
                scheduler=scheduler,
            )
        data_in.delete()
        return

    fn2(copy=1, mul=2).eval(
        data_in,
        ds_out,
        num_workers=TRANSFORM_TEST_NUM_WORKERS,
        scheduler=scheduler,
        progressbar=False,
    )
    assert len(ds_out) == 99
    for index in range(1, 100):
        np.testing.assert_array_equal(
            ds_out[index - 1].image.numpy(), 2 * index * np.ones((index, index))
        )
        np.testing.assert_array_equal(
            ds_out[index - 1].label.numpy(), 2 * index * np.ones((1,))
        )

    assert ds_out.image.shape_interval.lower == (99, 1, 1)
    assert ds_out.image.shape_interval.upper == (99, 99, 99)
    data_in.delete()


def test_groups(local_ds):
    with CliRunner().isolated_filesystem():
        with hub.dataset("./test/transform_hub_in_generic") as data_in:
            data_in.create_tensor("data/image")
            data_in.create_tensor("data/label")
            for i in range(1, 100):
                data_in.data.image.append(i * np.ones((i, i)))
                data_in.data.label.append(i * np.ones((1,)))
        data_in = hub.dataset("./test/transform_hub_in_generic")
        ds_out = local_ds
        ds_out.create_tensor("stuff/image")
        ds_out.create_tensor("stuff/label")

        data_in = data_in.data
        ds_out = ds_out.stuff

        fn2(copy=1, mul=2).eval(
            data_in, ds_out, num_workers=TRANSFORM_TEST_NUM_WORKERS, progressbar=False
        )
        assert len(ds_out) == 99
        for index in range(1, 100):
            np.testing.assert_array_equal(
                ds_out[index - 1].image.numpy(), 2 * index * np.ones((index, index))
            )
            np.testing.assert_array_equal(
                ds_out[index - 1].label.numpy(), 2 * index * np.ones((1,))
            )

        assert ds_out.image.shape_interval.lower == (99, 1, 1)
        assert ds_out.image.shape_interval.upper == (99, 99, 99)


def test_groups_2(local_ds):
    with CliRunner().isolated_filesystem():
        with hub.dataset("./test/transform_hub_in_generic") as data_in:
            data_in.create_tensor("data/z/y/x/image")
            data_in.create_tensor("data/z/y/x/label")
            for i in range(1, 100):
                data_in.data.z.y.x.image.append(i * np.ones((i, i)))
                data_in.data.z.y.x.label.append(i * np.ones((1,)))
        data_in = hub.dataset("./test/transform_hub_in_generic")
        ds_out = local_ds
        ds_out.create_tensor("stuff/x/y/z/image")
        ds_out.create_tensor("stuff/x/y/z/label")

        data_in = data_in.data
        ds_out = ds_out.stuff

        fn5(copy=1, mul=2).eval(
            data_in, ds_out, num_workers=TRANSFORM_TEST_NUM_WORKERS, progressbar=False
        )
        assert len(ds_out) == 99
        for index in range(1, 100):
            np.testing.assert_array_equal(
                ds_out.x.y.z.image[index - 1].numpy(),
                2 * index * np.ones((index, index)),
            )
            np.testing.assert_array_equal(
                ds_out.x.y.z.label[index - 1].numpy(), 2 * index * np.ones((1,))
            )

        assert ds_out.x.y.z.image.shape_interval.lower == (99, 1, 1)
        assert ds_out.x.y.z.image.shape_interval.upper == (99, 99, 99)


@parametrize_num_workers
@all_schedulers
def test_single_transform_hub_dataset_htypes(local_ds, num_workers, scheduler):
    data_in = hub.dataset("./test/single_transform_hub_dataset_htypes", overwrite=True)
    with data_in:
        data_in.create_tensor("image", htype="image", sample_compression="png")
        data_in.create_tensor("label", htype="class_label")
        for i in range(1, 100):
            data_in.image.append(i * np.ones((i, i), dtype="uint8"))
            data_in.label.append(i * np.ones((1,), dtype="uint32"))
    ds_out = local_ds
    ds_out.create_tensor("image")
    ds_out.create_tensor("label")
    fn2(copy=1, mul=2).eval(
        data_in, ds_out, num_workers=num_workers, progressbar=False, scheduler=scheduler
    )
    assert len(ds_out) == 99
    for index in range(1, 100):
        np.testing.assert_array_equal(
            ds_out[index - 1].image.numpy(), 2 * index * np.ones((index, index))
        )
        np.testing.assert_array_equal(
            ds_out[index - 1].label.numpy(), 2 * index * np.ones((1,))
        )

    assert ds_out.image.shape_interval.lower == (99, 1, 1)
    assert ds_out.image.shape_interval.upper == (99, 99, 99)
    data_in.delete()


@all_schedulers
def test_chain_transform_list_small(local_ds, scheduler):
    ls = list(range(100))
    ds_out = local_ds
    ds_out.create_tensor("image")
    ds_out.create_tensor("label")
    pipeline = hub.compose([fn1(mul=5, copy=2), fn2(mul=3, copy=3)])
    pipeline.eval(
        ls,
        ds_out,
        num_workers=TRANSFORM_TEST_NUM_WORKERS,
        progressbar=False,
        scheduler=scheduler,
    )
    assert len(ds_out) == 600
    for i in range(100):
        for index in range(6 * i, 6 * i + 6):
            np.testing.assert_array_equal(
                ds_out[index].image.numpy(), 15 * i * np.ones((337, 200))
            )
            np.testing.assert_array_equal(
                ds_out[index].label.numpy(), 15 * i * np.ones((1,))
            )


@all_schedulers
def test_chain_transform_list_big(local_ds, scheduler):
    ls = [i for i in range(2)]
    ds_out = local_ds
    ds_out.create_tensor("image")
    ds_out.create_tensor("label")
    pipeline = hub.compose([fn3(mul=5, copy=2), fn2(mul=3, copy=3)])
    pipeline.eval(
        ls,
        ds_out,
        num_workers=TRANSFORM_TEST_NUM_WORKERS,
        progressbar=False,
        scheduler=scheduler,
    )
    assert len(ds_out) == 12
    for i in range(2):
        for index in range(6 * i, 6 * i + 6):
            np.testing.assert_array_equal(
                ds_out[index].image.numpy(), 15 * i * np.ones((1310, 2087))
            )
            np.testing.assert_array_equal(
                ds_out[index].label.numpy(), 15 * i * np.ones((13,))
            )


@all_schedulers
@commit_or_not
def test_add_to_non_empty_dataset(local_ds, scheduler, do_commit):
    ls = [i for i in range(100)]
    ds_out = local_ds
    ds_out.create_tensor("image")
    ds_out.create_tensor("label")
    pipeline = hub.compose([fn1(mul=5, copy=2), fn2(mul=3, copy=3)])
    with ds_out:
        for i in range(10):
            ds_out.image.append(i * np.ones((10, 10)))
            ds_out.label.append(i * np.ones((1,)))
        if do_commit:
            ds_out.commit()

    pipeline.eval(
        ls,
        ds_out,
        num_workers=TRANSFORM_TEST_NUM_WORKERS,
        progressbar=False,
        scheduler=scheduler,
    )
    assert len(ds_out) == 610
    for i in range(10):
        np.testing.assert_array_equal(ds_out[i].image.numpy(), i * np.ones((10, 10)))
        np.testing.assert_array_equal(ds_out[i].label.numpy(), i * np.ones((1,)))
    for i in range(100):
        for index in range(10 + 6 * i, 10 + 6 * i + 6):
            np.testing.assert_array_equal(
                ds_out[index].image.numpy(), 15 * i * np.ones((337, 200))
            )
            np.testing.assert_array_equal(
                ds_out[index].label.numpy(), 15 * i * np.ones((1,))
            )

    diff = ds_out.diff(as_dict=True)["tensor"]
    change = {
        "image": {
            "data_updated": set(),
            "info_updated": False,
            "data_transformed_in_place": False,
        },
        "label": {
            "data_updated": set(),
            "info_updated": False,
            "data_transformed_in_place": False,
        },
    }
    if do_commit:
        change["image"]["cleared"] = False
        change["label"]["cleared"] = False
        change["image"]["created"] = False
        change["label"]["created"] = False
        change["image"]["data_added"] = [10, 610]
        change["label"]["data_added"] = [10, 610]
    else:
        change["image"]["cleared"] = False
        change["label"]["cleared"] = False
        change["image"]["created"] = True
        change["label"]["created"] = True
        change["image"]["data_added"] = [0, 610]
        change["label"]["data_added"] = [0, 610]
    assert diff == change


@all_schedulers
@all_compressions
def test_transform_hub_read(local_ds, cat_path, sample_compression, scheduler):
    data_in = [cat_path] * 10
    ds_out = local_ds
    ds_out.create_tensor("image", htype="image", sample_compression=sample_compression)

    read_image().eval(
        data_in,
        ds_out,
        num_workers=TRANSFORM_TEST_NUM_WORKERS,
        progressbar=False,
        scheduler=scheduler,
    )
    assert len(ds_out) == 10
    for i in range(10):
        assert ds_out.image[i].numpy().shape == (900, 900, 3)
        np.testing.assert_array_equal(ds_out.image[i].numpy(), ds_out.image[0].numpy())


@all_schedulers
@all_compressions
def test_transform_hub_read_pipeline(local_ds, cat_path, sample_compression, scheduler):
    data_in = [cat_path] * 10
    ds_out = local_ds
    ds_out.create_tensor("image", htype="image", sample_compression=sample_compression)
    pipeline = hub.compose([read_image(), crop_image(copy=2)])
    pipeline.eval(
        data_in,
        ds_out,
        num_workers=TRANSFORM_TEST_NUM_WORKERS,
        progressbar=False,
        scheduler=scheduler,
    )
    assert len(ds_out) == 20
    for i in range(20):
        assert ds_out.image[i].numpy().shape == (100, 100, 3)
        np.testing.assert_array_equal(ds_out.image[i].numpy(), ds_out.image[0].numpy())


def test_hub_like(local_ds, scheduler="threaded"):
    with CliRunner().isolated_filesystem():
        data_in = local_ds
        with data_in:
            data_in.create_tensor("image", htype="image", sample_compression="png")
            data_in.create_tensor("label", htype="class_label")
            for i in range(1, 100):
                data_in.image.append(i * np.ones((i, i), dtype="uint8"))
                data_in.label.append(i * np.ones((1,), dtype="uint32"))
        ds_out = hub.like("./transform_hub_like", data_in)
        fn2(copy=1, mul=2).eval(
            data_in,
            ds_out,
            num_workers=TRANSFORM_TEST_NUM_WORKERS,
            progressbar=False,
            scheduler=scheduler,
        )
        assert len(ds_out) == 99
        for index in range(1, 100):
            np.testing.assert_array_equal(
                ds_out[index - 1].image.numpy(), 2 * index * np.ones((index, index))
            )
            np.testing.assert_array_equal(
                ds_out[index - 1].label.numpy(), 2 * index * np.ones((1,))
            )

        assert ds_out.image.shape_interval.lower == (99, 1, 1)
        assert ds_out.image.shape_interval.upper == (99, 99, 99)


def test_transform_empty(local_ds):
    local_ds.create_tensor("image")

    ls = list(range(10))
    filter_tr().eval(ls, local_ds, progressbar=False)

    assert len(local_ds) == 5

    for i in range(5):
        np.testing.assert_array_equal(
            local_ds[i].image.numpy(), 2 * i * np.ones((100, 100))
        )


def test_pbar_description():
    assert get_pbar_description([fn1()]) == "Evaluating fn1"
    assert get_pbar_description([fn1(), fn2()]) == "Evaluating [fn1, fn2]"
    assert get_pbar_description([fn1(), fn1()]) == "Evaluating [fn1, fn1]"
    assert (
        get_pbar_description([fn1(), fn1(), read_image()])
        == "Evaluating [fn1, fn1, read_image]"
    )


def test_bad_transform(memory_ds):
    ds = memory_ds
    ds.create_tensor("x")
    ds.create_tensor("y")
    with ds:
        ds.x.extend(np.random.rand(10, 1))
        ds.y.extend(np.random.rand(10, 1))
    ds2 = hub.like("mem://dummy2", ds)

    @hub.compute
    def fn_filter(sample_in, sample_out):
        sample_out.y.append(sample_in.y.numpy())
        return sample_out

    with pytest.raises(TransformError):
        fn_filter().eval(ds, ds2, progressbar=False)


def test_transform_persistance(local_ds_generator, num_workers=2, scheduler="threaded"):
    data_in = hub.dataset("./test/single_transform_hub_dataset_htypes", overwrite=True)
    with data_in:
        data_in.create_tensor("image", htype="image", sample_compression="png")
        data_in.create_tensor("label", htype="class_label")
        for i in range(1, 100):
            data_in.image.append(i * np.ones((i, i), dtype="uint8"))
            data_in.label.append(i * np.ones((1,), dtype="uint32"))
    ds_out = local_ds_generator()
    ds_out.create_tensor("image")
    ds_out.create_tensor("label")
    if (
        isinstance(remove_memory_cache(ds_out.storage), MemoryProvider)
        and scheduler != "threaded"
        and num_workers > 0
    ):
        # any scheduler other than `threaded` will not work with a dataset stored in memory
        # num_workers = 0 automatically does single threaded irrespective of the scheduler
        with pytest.raises(InvalidOutputDatasetError):
            fn2(copy=1, mul=2).eval(
                data_in,
                ds_out,
                num_workers=num_workers,
                scheduler=scheduler,
                progressbar=False,
            )
        data_in.delete()
        return
    fn2(copy=1, mul=2).eval(
        data_in, ds_out, num_workers=num_workers, scheduler=scheduler, progressbar=False
    )

    def test_ds_out():
        assert len(ds_out) == 99
        for index in range(1, 100):
            np.testing.assert_array_equal(
                ds_out[index - 1].image.numpy(), 2 * index * np.ones((index, index))
            )
            np.testing.assert_array_equal(
                ds_out[index - 1].label.numpy(), 2 * index * np.ones((1,))
            )

        assert ds_out.image.shape_interval.lower == (99, 1, 1)
        assert ds_out.image.shape_interval.upper == (99, 99, 99)

    test_ds_out()
    ds_out = local_ds_generator()
    test_ds_out()

    data_in.delete()


def test_ds_append_in_transform(memory_ds):
    ds = memory_ds
    data_in = hub.dataset("./test/single_transform_hub_dataset", overwrite=True)
    with data_in:
        data_in.create_tensor("image")
        data_in.create_tensor("label")
        for i in range(1, 100):
            data_in.image.append(i * np.ones((i, i)))
            data_in.label.append(i * np.ones((1,)))
    ds_out = ds
    ds_out.create_tensor("image")
    ds_out.create_tensor("label")

    fn6(copy=1, mul=2).eval(
        data_in, ds_out, num_workers=2, scheduler="threaded", progressbar=False
    )
    assert len(ds_out) == 99
    for index in range(1, 100):
        np.testing.assert_array_equal(
            ds_out[index - 1].image.numpy(), 2 * index * np.ones((index, index))
        )
        np.testing.assert_array_equal(
            ds_out[index - 1].label.numpy(), 2 * index * np.ones((1,))
        )

    assert ds_out.image.shape_interval.lower == (99, 1, 1)
    assert ds_out.image.shape_interval.upper == (99, 99, 99)
    data_in.delete()


def test_transform_pass_through():
    data_in = hub.dataset("mem://ds1")
    data_in.create_tensor("image", htype="image", sample_compression="png")
    data_in.create_tensor("label", htype="class_label")
    for i in range(1, 100):
        data_in.image.append(i * np.ones((i, i), dtype="uint8"))
        data_in.label.append(i * np.ones((1,), dtype="uint32"))
    ds_out = hub.dataset("mem://ds2")
    ds_out.create_tensor("image", htype="image", sample_compression="png")
    ds_out.create_tensor("label", htype="class_label")
    fn4().eval(data_in, ds_out, num_workers=2, scheduler="threaded", progressbar=False)
    for i in range(len(data_in)):
        np.testing.assert_array_equal(
            data_in[i].image.numpy(), ds_out[i * 2].image.numpy()
        )
        np.testing.assert_array_equal(
            data_in[i].label.numpy(), ds_out[i * 2].label.numpy()
        )
        np.testing.assert_array_equal(
            data_in[i].image.numpy() * 2, ds_out[i * 2 + 1].image.numpy()
        )
        np.testing.assert_array_equal(
            data_in[i].label.numpy() * 2, ds_out[i * 2 + 1].label.numpy()
        )


def test_inplace_transform(local_ds_generator):
    ds = local_ds_generator()

    with ds:
        ds.create_tensor("img")
        ds.create_tensor("label")
        for i in range(10):
            if i == 5:
                ds.img.append(np.zeros((200, 200, 3)))
            else:
                ds.img.append(np.ones((200, 200, 3)))
            ds.label.append(1)
        a = ds.commit()
        assert len(ds) == 10
        for i in range(10):
            if i != 5:
                check_target_array(ds, i, 1)
        ds.img[5] = np.ones((200, 200, 3))
        b = ds.commit()

        inplace_transform().eval(
            ds, num_workers=TRANSFORM_TEST_NUM_WORKERS, progressbar=False
        )
        assert ds.img.chunk_engine.num_samples == len(ds) == 20

        for i in range(20):
            target = 2 if i % 2 == 0 else 3
            check_target_array(ds, i, target)

        diff = ds.diff(as_dict=True)["tensor"]
        change = {
            "img": {
                "created": False,
                "cleared": False,
                "data_added": [0, 20],
                "data_updated": set(),
                "data_transformed_in_place": True,
                "info_updated": False,
            },
            "label": {
                "created": False,
                "cleared": False,
                "data_added": [0, 20],
                "data_updated": set(),
                "data_transformed_in_place": True,
                "info_updated": False,
            },
        }
        assert diff == change

        ds.checkout(b)
        assert len(ds) == 10
        for i in range(10):
            check_target_array(ds, i, 1)

    ds = local_ds_generator()
    assert len(ds) == 20
    for i in range(20):
        target = 2 if i % 2 == 0 else 3
        check_target_array(ds, i, target)

    ds.checkout(b)
    assert len(ds) == 10
    for i in range(10):
        check_target_array(ds, i, 1)


def test_inplace_transform_without_commit(local_ds_generator):
    ds = local_ds_generator()

    with ds:
        ds.create_tensor("img")
        ds.create_tensor("label")
        for _ in range(10):
            ds.img.append(np.ones((200, 200, 3)))
            ds.label.append(1)
        assert len(ds) == 10
        for i in range(10):
            check_target_array(ds, i, 1)

        inplace_transform().eval(
            ds, num_workers=TRANSFORM_TEST_NUM_WORKERS, progressbar=False
        )
        assert ds.img.chunk_engine.num_samples == len(ds) == 20

        for i in range(20):
            target = 2 if i % 2 == 0 else 3
            check_target_array(ds, i, target)

    ds = local_ds_generator()
    assert len(ds) == 20
    for i in range(20):
        target = 2 if i % 2 == 0 else 3
        check_target_array(ds, i, target)


def test_inplace_transform_non_head(local_ds_generator):
    ds = local_ds_generator()
    with ds:
        ds.create_tensor("img")
        ds.create_tensor("label")
        for _ in range(10):
            ds.img.append(np.ones((200, 200, 3)))
            ds.label.append(1)
        assert len(ds) == 10
        for i in range(10):
            check_target_array(ds, i, 1)
        a = ds.commit()
        for _ in range(5):
            ds.img.append(np.ones((200, 200, 3)))
            ds.label.append(1)
        assert len(ds) == 15
        for i in range(15):
            check_target_array(ds, i, 1)

        ds.checkout(a)

        # transforming non-head node
        inplace_transform().eval(
            ds, num_workers=TRANSFORM_TEST_NUM_WORKERS, progressbar=False
        )
        br = ds.branch

        assert len(ds) == 20
        for i in range(20):
            target = 2 if i % 2 == 0 else 3
            check_target_array(ds, i, target)

        ds.checkout(a)
        assert len(ds) == 10
        for i in range(10):
            check_target_array(ds, i, 1)

        ds.checkout("main")
        assert len(ds) == 15
        for i in range(15):
            check_target_array(ds, i, 1)

    ds = local_ds_generator()
    assert len(ds) == 15
    for i in range(15):
        check_target_array(ds, i, 1)

    ds.checkout(a)
    assert len(ds) == 10
    for i in range(10):
        check_target_array(ds, i, 1)

    ds.checkout(br)
    assert len(ds) == 20
    for i in range(20):
        target = 2 if i % 2 == 0 else 3
        check_target_array(ds, i, target)


def test_inplace_transform_clear_chunks(local_ds_generator):
    ds = local_ds_generator()

    with ds:
        ds.create_tensor("img")
        ds.create_tensor("label")

        for _ in range(10):
            ds.img.append(np.ones((500, 500, 3)))
            ds.label.append(np.ones(3))

    prev_chunks = set(
        [
            f"{tensor.key}/chunks/{chunk}"
            for tensor in [ds.img, ds.label]
            for chunk in tensor.chunk_engine.list_all_chunks()
        ]
    )
    inplace_transform().eval(ds)
    after_chunks = set(
        [
            f"{tensor.key}/chunks/{chunk}"
            for tensor in [ds.img, ds.label]
            for chunk in tensor.chunk_engine.list_all_chunks()
        ]
    )

    # all chunks where replaced
    assert len(after_chunks.intersection(prev_chunks)) == 0

    # test all new chunks where created
    for chunk in after_chunks:
        assert ds.storage[chunk] is not None

    # test all old chunks where removed
    for chunk in prev_chunks:
        try:
            assert ds.storage[chunk] is None
        except KeyError:
            pass


def test_transform_skip_ok(local_ds_generator):
    ds = local_ds_generator()
    ls = list(range(100))
    with ds:
        ds.create_tensor("image")
        ds.create_tensor("label")
        ds.create_tensor("unused")

    pipeline = hub.compose([fn1(mul=5, copy=2), fn2(mul=3, copy=3)])
    pipeline.eval(
        ls,
        ds,
        num_workers=TRANSFORM_TEST_NUM_WORKERS,
        progressbar=False,
        scheduler="processed",
        skip_ok=True,
    )
    for i in range(100):
        for index in range(6 * i, 6 * i + 6):
            np.testing.assert_array_equal(
                ds.image[index].numpy(), 15 * i * np.ones((337, 200))
            )
            np.testing.assert_array_equal(
                ds.label[index].numpy(), 15 * i * np.ones((1,))
            )

    assert len(ds.unused) == 0

    # test persistence
    ds = local_ds_generator()
    for i in range(100):
        for index in range(6 * i, 6 * i + 6):
            np.testing.assert_array_equal(
                ds.image[index].numpy(), 15 * i * np.ones((337, 200))
            )
            np.testing.assert_array_equal(
                ds.label[index].numpy(), 15 * i * np.ones((1,))
            )
    assert len(ds.unused) == 0


def test_inplace_transform_skip_ok(local_ds_generator):
    ds = local_ds_generator()

    with ds:
        ds.create_tensor("img")
        ds.create_tensor("label")
        ds.create_tensor("unused")
        ds.img.extend(np.ones((10, 200, 200, 3)))
        ds.label.extend([1 for _ in range(10)])
        ds.unused.extend(5 * np.ones((10, 10, 10)))
        for i in range(10):
            check_target_array(ds, i, 1)

    inplace_transform().eval(
        ds, num_workers=TRANSFORM_TEST_NUM_WORKERS, progressbar=False, skip_ok=True
    )
    assert ds.img.chunk_engine.num_samples == 20

    for i in range(20):
        target = 2 if i % 2 == 0 else 3
        check_target_array(ds, i, target)

    assert len(ds.unused) == 10
    np.testing.assert_array_equal(ds.unused.numpy(), 5 * np.ones((10, 10, 10)))

    # test persistence
    ds = local_ds_generator()

    assert ds.img.chunk_engine.num_samples == 20

    for i in range(20):
        target = 2 if i % 2 == 0 else 3
        check_target_array(ds, i, target)

    assert len(ds.unused) == 10
    np.testing.assert_array_equal(ds.unused.numpy(), 5 * np.ones((10, 10, 10)))


def test_chunk_compression_bug(local_ds):
    xyz = np.zeros((480, 640), dtype=np.float32)
    length = 55
    dataset = [xyz] * length
    with local_ds as ds:
        ds.create_tensor("xyz", chunk_compression="lz4")
        populate_cc_bug().eval(dataset, ds, num_workers=2, scheduler="threaded")

    for index in range(length):
        np.testing.assert_array_equal(ds.xyz[index].numpy(), xyz)


@hub.compute
def sequence_transform(inp, out):
    out.x.append([np.ones(inp)] * inp)


def test_sequence_htype_with_transform(local_ds):
    ds = local_ds
    with ds:
        ds.create_tensor("x", htype="sequence")
        assert ds.x.dtype is None
        assert ds.x.htype == "sequence[None]"
        sequence_transform().eval(list(range(1, 11)), ds, TRANSFORM_TEST_NUM_WORKERS)
    for i in range(10):
        np.testing.assert_array_equal(ds.x[i].numpy(), np.ones((i + 1, i + 1)))
    assert ds.x.dtype == np.ones(1).dtype
    assert ds.x.htype == "sequence[generic]"


def test_htype_dtype_after_transform(local_ds):
    ds = local_ds
    with ds:
        ds.create_tensor("image")
        assert ds.image.htype is None
        assert ds.image.dtype is None
        ds.create_tensor("label")
        fn3().eval(list(range(10)), ds, TRANSFORM_TEST_NUM_WORKERS)
    assert ds.image.htype == "generic"
    assert ds.image.dtype == np.ones(1).dtype
