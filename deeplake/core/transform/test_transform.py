import deeplake
import pytest
import numpy as np
from click.testing import CliRunner
from deeplake.core.storage.memory import MemoryProvider
from deeplake.core.version_control.test_version_control import (
    compare_dataset_diff,
    compare_tensor_diff,
    get_default_tensor_diff,
    get_default_dataset_diff,
)
from deeplake.util.remove_cache import remove_memory_cache
from deeplake.util.exceptions import (
    AllSamplesSkippedError,
    EmptyTensorError,
    InvalidOutputDatasetError,
    SampleExtendingError,
    TransformError,
)
from deeplake.tests.common import parametrize_num_workers
from deeplake.util.transform import get_pbar_description
import deeplake
import gc
import re
from deeplake.tests.common import get_dummy_data_path


# github actions can only support 2 workers
TRANSFORM_TEST_NUM_WORKERS = 2

all_compressions = pytest.mark.parametrize("sample_compression", [None, "png", "jpeg"])

schedulers = ["threaded", "processed"]
all_schedulers = pytest.mark.parametrize("scheduler", schedulers)
commit_or_not = pytest.mark.parametrize("do_commit", [True, False])


@deeplake.compute
def fn1(sample_in, samples_out, mul=1, copy=1):
    for _ in range(copy):
        samples_out.image.append(np.ones((337, 200), dtype=np.uint8) * sample_in * mul)
        samples_out.label.append(np.ones((1,), dtype=np.uint32) * sample_in * mul)


@deeplake.compute
def fn2(sample_in, samples_out, mul=1, copy=1):
    for _ in range(copy):
        samples_out.image.append(sample_in.image.numpy() * mul)
        samples_out.label.append(sample_in.label.numpy() * mul)


@deeplake.compute
def fn3(sample_in, samples_out, mul=1, copy=1):
    for _ in range(copy):
        samples_out.image.append(np.ones((1310, 2087)) * sample_in * mul)
        samples_out.label.append(np.ones((13,)) * sample_in * mul)


@deeplake.compute
def fn4(sample_in, samples_out):
    samples_out.image.append(sample_in.image)
    samples_out.image.append(sample_in.image.numpy() * 2)
    samples_out.label.append(sample_in.label)
    samples_out.label.append(sample_in.label.numpy() * 2)


@deeplake.compute
def fn5(sample_in, samples_out, mul=1, copy=1):
    for _ in range(copy):
        samples_out.x["y"].z.image.append(sample_in.z.y.x.image.numpy() * mul)
        samples_out.x.y.z["label"].append(sample_in.z.y.x.label.numpy() * mul)


@deeplake.compute
def fn6(sample_in, samples_out, mul=1, copy=1):
    for _ in range(copy):
        samples_out.append(
            {
                "image": sample_in.image.numpy() * mul,
                "label": sample_in.label.numpy() * mul,
            }
        )


@deeplake.compute
def read_image(sample_in, samples_out):
    samples_out.image.append(deeplake.read(sample_in))


@deeplake.compute
def crop_image(sample_in, samples_out, copy=1):
    for _ in range(copy):
        samples_out.image.append(sample_in.image.numpy()[:100, :100, :])


@deeplake.compute
def filter_tr(sample_in, sample_out):
    if sample_in % 2 == 0:
        sample_out.image.append(sample_in * np.ones((100, 100)))


@deeplake.compute
def populate_cc_bug(sample_in, samples_out):
    samples_out.xyz.append(sample_in)
    return samples_out


@deeplake.compute
def inplace_transform(sample_in, samples_out):
    samples_out.img.append(2 * sample_in.img.numpy())
    samples_out.img.append(3 * sample_in.img.numpy())
    samples_out.label.append(2 * sample_in.label.numpy())
    samples_out.label.append(3 * sample_in.label.numpy())


@deeplake.compute
def unequal_transform(sample_in, samples_out):
    samples_out.x.append(sample_in.x.numpy() * 2)
    if sample_in.y.numpy().size > 0:
        samples_out.y.append(sample_in.y.numpy() * 2)


@deeplake.compute
def add_text(sample_in, samples_out):
    samples_out.abc.append(sample_in)


@deeplake.compute
def add_link(sample_in, samples_out):
    samples_out.abc.append(deeplake.link(sample_in))


@deeplake.compute
def add_image(sample_in, samples_out):
    samples_out.image.append(np.random.randint(0, 255, (1310, 2087, 3), dtype=np.uint8))


@deeplake.compute
def add_images(i, sample_out):
    for _ in range(5):
        image = deeplake.read(get_dummy_data_path("images/flower.png"))
        sample_out.append({"image": image})


@deeplake.compute
def small_transform(sample_in, samples_out):
    samples_out.image.append(sample_in)


@deeplake.compute
def fn_aggregate(samples_in, samples_out, key, values):
    values.append(samples_in[key].numpy().mean())


def check_target_array(ds, index, target):
    np.testing.assert_array_equal(
        ds.img[index].numpy(), target * np.ones((200, 200, 3))
    )
    np.testing.assert_array_equal(ds.label[index].numpy(), target * np.ones((1,)))


def retrieve_objects_from_memory(object_type=deeplake.core.sample.Sample):
    total_n_of_occurences = 0
    gc_objects = gc.get_objects()
    for item in gc_objects:
        try:
            if isinstance(item, object_type):
                total_n_of_occurences += 1
        except ReferenceError:
            pass  # weakly-referenced object which no longer exists
    return total_n_of_occurences


@pytest.mark.slow
@all_schedulers
@pytest.mark.parametrize(
    "ds",
    ["memory_ds", "local_ds", "s3_ds"],
    indirect=True,
)
def test_single_transform_deeplake_dataset(ds, scheduler):
    data_in = deeplake.dataset(
        "./test/single_transform_deeplake_dataset", overwrite=True
    )
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
        with deeplake.dataset("./test/transform_deeplake_in_generic") as data_in:
            data_in.create_tensor("data/image")
            data_in.create_tensor("data/label")
            for i in range(1, 100):
                data_in.data.image.append(i * np.ones((i, i)))
                data_in.data.label.append(i * np.ones((1,)))
        data_in = deeplake.dataset("./test/transform_deeplake_in_generic")
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
        with deeplake.dataset("./test/transform_deeplake_in_generic") as data_in:
            data_in.create_tensor("data/z/y/x/image")
            data_in.create_tensor("data/z/y/x/label")
            for i in range(1, 100):
                data_in.data.z.y.x.image.append(i * np.ones((i, i)))
                data_in.data.z.y.x.label.append(i * np.ones((1,)))
        data_in = deeplake.dataset("./test/transform_deeplake_in_generic")
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


@pytest.mark.slow
@parametrize_num_workers
@all_schedulers
def test_single_transform_deeplake_dataset_htypes(local_ds, num_workers, scheduler):
    data_in = deeplake.dataset(
        "./test/single_transform_deeplake_dataset_htypes", overwrite=True
    )
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
            ds_out[index - 1].image.numpy(), 2 * index * np.ones((index, index, 1))
        )
        np.testing.assert_array_equal(
            ds_out[index - 1].label.numpy(), 2 * index * np.ones((1,))
        )

    assert ds_out.image.shape_interval.lower == (99, 1, 1, 1)
    assert ds_out.image.shape_interval.upper == (99, 99, 99, 1)
    data_in.delete()


@pytest.mark.slow
@all_schedulers
def test_chain_transform_list_small(local_ds, scheduler):
    ls = list(range(100))
    ds_out = local_ds
    ds_out.create_tensor("image")
    ds_out.create_tensor("label")
    pipeline = deeplake.compose([fn1(mul=5, copy=2), fn2(mul=3, copy=3)])
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
                ds_out[index].image.numpy(),
                np.asarray(15 * i * np.ones((337, 200)), dtype=np.uint8),
            )
            np.testing.assert_array_equal(
                ds_out[index].label.numpy(), 15 * i * np.ones((1,))
            )


@pytest.mark.slow
@all_schedulers
def test_chain_transform_list_big(local_ds, scheduler):
    ls = [i for i in range(2)]
    ds_out = local_ds
    ds_out.create_tensor("image")
    ds_out.create_tensor("label")
    pipeline = deeplake.compose([fn3(mul=5, copy=2), fn2(mul=3, copy=3)])
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


@pytest.mark.slow
@all_schedulers
@commit_or_not
def test_add_to_non_empty_dataset(local_ds, scheduler, do_commit):
    ls = [i for i in range(100)]
    ds_out = local_ds
    ds_out.create_tensor("image")
    ds_out.create_tensor("label")
    pipeline = deeplake.compose([fn1(mul=5, copy=2), fn2(mul=3, copy=3)])
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
                ds_out[index].image.numpy(),
                np.asarray(15 * i * np.ones((337, 200)), dtype=np.uint8),
            )
            np.testing.assert_array_equal(
                ds_out[index].label.numpy(), 15 * i * np.ones((1,))
            )

    expected_tensor_diff = {
        "commit_id": local_ds.pending_commit_id,
        "image": get_default_tensor_diff(),
        "label": get_default_tensor_diff(),
    }

    expected_dataset_diff = get_default_dataset_diff(local_ds.pending_commit_id)

    if do_commit:
        expected_tensor_diff["image"]["data_added"] = [10, 610]
        expected_tensor_diff["label"]["data_added"] = [10, 610]
    else:
        expected_tensor_diff["image"]["created"] = True
        expected_tensor_diff["label"]["created"] = True
        expected_tensor_diff["image"]["data_added"] = [0, 610]
        expected_tensor_diff["label"]["data_added"] = [0, 610]

    diff = ds_out.diff(as_dict=True)
    tensor_diff = diff["tensor"]
    dataset_diff = diff["dataset"]
    compare_tensor_diff([expected_tensor_diff], tensor_diff)
    compare_dataset_diff([expected_dataset_diff], dataset_diff)


@pytest.mark.slow
@all_schedulers
@all_compressions
def test_transform_deeplake_read(local_ds, cat_path, sample_compression, scheduler):
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


@pytest.mark.slow
@all_schedulers
@all_compressions
def test_transform_deeplake_read_pipeline(
    local_ds, cat_path, sample_compression, scheduler
):
    data_in = [cat_path] * 10
    ds_out = local_ds
    ds_out.create_tensor("image", htype="image", sample_compression=sample_compression)
    pipeline = deeplake.compose([read_image(), crop_image(copy=2)])
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


def test_deeplake_like(local_ds, scheduler="threaded"):
    with CliRunner().isolated_filesystem():
        data_in = local_ds
        with data_in:
            data_in.create_tensor("image", htype="image", sample_compression="png")
            data_in.create_tensor("label", htype="class_label")
            for i in range(1, 100):
                data_in.image.append(i * np.ones((i, i), dtype="uint8"))
                data_in.label.append(i * np.ones((1,), dtype="uint32"))
        ds_out = deeplake.like("./transform_deeplake_like", data_in)
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
                ds_out[index - 1].image.numpy(), 2 * index * np.ones((index, index, 1))
            )
            np.testing.assert_array_equal(
                ds_out[index - 1].label.numpy(), 2 * index * np.ones((1,))
            )

        assert ds_out.image.shape_interval.lower == (99, 1, 1, 1)
        assert ds_out.image.shape_interval.upper == (99, 99, 99, 1)


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
    ds2 = deeplake.like("mem://dummy2", ds)

    @deeplake.compute
    def fn_filter(sample_in, sample_out):
        sample_out.y.append(sample_in.y.numpy())
        return sample_out

    with pytest.raises(TransformError):
        fn_filter().eval(ds, ds2, progressbar=False)


def test_transform_persistance(local_ds_generator, num_workers=2, scheduler="threaded"):
    data_in = deeplake.dataset(
        "./test/single_transform_deeplake_dataset_htypes", overwrite=True
    )
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
                ds_out[index - 1].image.numpy(), 2 * index * np.ones((index, index, 1))
            )
            np.testing.assert_array_equal(
                ds_out[index - 1].label.numpy(), 2 * index * np.ones((1,))
            )

        assert ds_out.image.shape_interval.lower == (99, 1, 1, 1)
        assert ds_out.image.shape_interval.upper == (99, 99, 99, 1)

    test_ds_out()
    ds_out = local_ds_generator()
    test_ds_out()

    data_in.delete()


def test_ds_append_in_transform(memory_ds):
    ds = memory_ds
    data_in = deeplake.dataset(
        "./test/single_transform_deeplake_dataset", overwrite=True
    )
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


@pytest.mark.slow
def test_transform_pass_through():
    data_in = deeplake.dataset("mem://ds1")
    data_in.create_tensor("image", htype="image", sample_compression="png")
    data_in.create_tensor("label", htype="class_label")
    for i in range(1, 100):
        data_in.image.append(i * np.ones((i, i), dtype="uint8"))
        data_in.label.append(i * np.ones((1,), dtype="uint32"))
    ds_out = deeplake.dataset("mem://ds2")
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

        expected_tensor_diff = {
            "commit_id": ds.pending_commit_id,
            "img": get_default_tensor_diff(),
            "label": get_default_tensor_diff(),
        }
        expected_dataset_diff = get_default_dataset_diff(ds.pending_commit_id)
        expected_tensor_diff["img"]["data_added"] = [0, 20]
        expected_tensor_diff["img"]["data_transformed_in_place"] = True
        expected_tensor_diff["label"]["data_added"] = [0, 20]
        expected_tensor_diff["label"]["data_transformed_in_place"] = True

        diff = ds.diff(as_dict=True)
        tensor_diff = diff["tensor"]
        dataset_diff = diff["dataset"]

        compare_tensor_diff([expected_tensor_diff], tensor_diff)
        compare_dataset_diff([expected_dataset_diff], dataset_diff)

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


@pytest.mark.slow
def test_inplace_transform_bug(local_ds_generator):
    @deeplake.compute
    def construct(sample_in, sample_out):
        sample_out.append({"positive": [1, 2, 3], "negative": [4, 5, 6]})

    ds = local_ds_generator()
    with ds:
        ds.create_tensor("id")
        ds.id.extend(list(range(10)))

        ds.create_tensor("positive")
        ds.create_tensor("negative")

    for _ in range(0, ds.max_len):
        construct().eval(
            ds,
            num_workers=2,
            skip_ok=True,
            check_lengths=False,
            pad_data_in=True,
        )

    np.testing.assert_array_equal(
        ds.positive.numpy(aslist=True), [np.array([1, 2, 3])] * 10
    )
    np.testing.assert_array_equal(
        ds.negative.numpy(aslist=True), [np.array([4, 5, 6])] * 10
    )


def test_inplace_transform_bug_2(local_ds_generator):
    @deeplake.compute
    def tform(sample_in, sample_out):
        sample_out.text2.append(sample_in.text.text())

    ds = local_ds_generator()
    with ds:
        ds.create_tensor("text", htype="text", sample_compression="lz4")
        ds.text.extend(["abcd", "efgh", "hijk"] * 10)
        ds.create_tensor("text2", htype="text", sample_compression="lz4")
        tform().eval(ds[["text"]], ds, num_workers=2, check_lengths=False)

    np.testing.assert_array_equal(ds.text.text(), ["abcd", "efgh", "hijk"] * 10)
    np.testing.assert_array_equal(ds.text2.text(), ["abcd", "efgh", "hijk"] * 10)


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


@pytest.mark.slow
def test_transform_skip_ok(local_ds_generator):
    ds = local_ds_generator()
    ls = list(range(100))
    with ds:
        ds.create_tensor("image")
        ds.create_tensor("label")
        ds.create_tensor("unused")

    pipeline = deeplake.compose([fn1(mul=5, copy=2), fn2(mul=3, copy=3)])
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
                ds.image[index].numpy(),
                np.asarray(15 * i * np.ones((337, 200)), dtype=np.uint8),
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
                ds.image[index].numpy(),
                np.asarray(15 * i * np.ones((337, 200)), dtype=np.uint8),
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


@deeplake.compute
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


def test_transform_pad_data_in(local_ds):
    with local_ds as ds:
        ds.create_tensor("x")
        ds.create_tensor("y")
        ds.x.extend(list(range(10)))
        ds.y.extend(list(range(5)))
    ds2 = deeplake.dataset("./data/unequal2", overwrite=True)
    ds2.create_tensor("x")
    ds2.create_tensor("y")

    unequal_transform().eval(ds, ds2, pad_data_in=True, skip_ok=True)
    assert len(ds2.x) == 10
    assert len(ds2.y) == 5
    assert len(ds2) == 5
    for i in range(10):
        x = ds2[i].x.numpy()
        np.testing.assert_equal(x, 2 * i)
        if i < 5:
            y = ds2[i].y.numpy()
            np.testing.assert_equal(y, 2 * i)

    for i, dsv in enumerate(ds2):
        x, y = dsv.x.numpy(), dsv.y.numpy()
        np.testing.assert_equal(x, 2 * i)
        np.testing.assert_equal(y, 2 * i)


def test_transform_bug_text(local_ds):
    with local_ds as ds:
        ds.create_tensor("abc", htype="text")
        ls = ["hello"] * 10
        add_text().eval(ls, ds, num_workers=2)
        assert len(ds) == 10
        ds.pop(6)
        assert len(ds) == 9

        for i in range(9):
            assert ds[i].abc.numpy() == "hello"


def test_transform_bug_link(local_ds, cat_path):
    with local_ds as ds:
        ds.create_tensor("abc", htype="link[image]", sample_compression="jpg")
        ls = [cat_path] * 10
        add_link().eval(ls, ds, num_workers=2)
        assert len(ds) == 10
        ds.pop(6)
        assert len(ds) == 9

        for i in range(9):
            assert ds[i].abc.numpy().shape == (900, 900, 3)
            assert ds[i].abc.shape == (900, 900, 3)


@pytest.mark.slow
def test_tensor_dataset_memory_leak(local_ds):
    local_ds.create_tensor("image", htype="image", sample_compression="png")
    add_images().eval(list(range(100)), local_ds, scheduler="threaded")

    n = retrieve_objects_from_memory()
    assert n == 0


def test_transform_info(local_ds_generator):
    ds = local_ds_generator()
    with ds:
        ds.create_tensor("image")
        small_transform().eval(range(1), ds)
        ds.info["test"] = 123
        assert ds.info["test"] == 123
    ds = local_ds_generator()
    assert ds.info["test"] == 123


@pytest.mark.slow
@parametrize_num_workers
@all_compressions
@pytest.mark.parametrize(
    "ds",
    ["memory_ds", "local_ds", "s3_ds"],
    indirect=True,
)
def test_read_only_dataset_aggregation_image(ds, sample_compression, num_workers):
    scheduler = "serial"
    i_start = 0
    i_stop = 100
    with ds:
        ds.create_tensor("image", htype="image", sample_compression=sample_compression)
        for i in range(i_start, i_stop):
            ds.image.append(i * np.ones((9, 16), dtype="uint8"))
    ds.read_only = True

    values = []
    fn_aggregate(key="image", values=values).eval(
        ds,
        num_workers=num_workers,
        progressbar=False,
        scheduler=scheduler,
        read_only_ok=True,
    )
    assert len(values) == i_stop - i_start
    assert np.array(values).mean() == (i_start + i_stop - 1) / 2  # half-open interval


@parametrize_num_workers
@pytest.mark.parametrize(
    "ds",
    ["memory_ds", "local_ds", pytest.param("s3_ds", marks=pytest.mark.slow)],
    indirect=True,
)
def test_read_only_dataset_aggregation_label(ds, num_workers):
    scheduler = "serial"

    i_start = 0
    i_stop = 100
    with ds:
        ds.create_tensor("label", htype="class_label")
        for i in range(i_start, i_stop):
            ds.label.append(i)
    ds.read_only = True

    values = []
    fn_aggregate(key="label", values=values).eval(
        ds,
        num_workers=num_workers,
        progressbar=False,
        scheduler=scheduler,
        read_only_ok=True,
    )
    assert len(values) == i_stop - i_start
    assert np.array(values).mean() == (i_start + i_stop - 1) / 2  # half-open interval


@parametrize_num_workers
@all_schedulers
@pytest.mark.parametrize(
    "ds",
    ["local_ds", pytest.param("s3_ds", marks=pytest.mark.slow)],
    indirect=True,
)
def test_read_only_dataset_raise(ds, scheduler, num_workers):
    with ds:
        ds.create_tensor("label", htype="class_label")
        ds.label.append(1)
    ds.read_only = True

    with pytest.raises(InvalidOutputDatasetError):
        values = []
        fn_aggregate(key="label", values=values).eval(
            ds, num_workers=num_workers, progressbar=False, scheduler=scheduler
        )


def test_read_only_dataset_raise_if_output_dataset(memory_ds):
    data_in = memory_ds

    with data_in:
        data_in.create_tensor("label", htype="class_label")
        data_in.label.append(1)

    data_out = deeplake.dataset(
        "mem://test_read_only_dataset_raise_if_output_dataset", overwrite=True
    )
    data_out.read_only = True

    with pytest.raises(InvalidOutputDatasetError):
        values = []
        fn_aggregate(key="label", values=values).eval(
            data_in, data_out, progressbar=False, read_only_ok=True
        )


@pytest.mark.parametrize(
    "compression", [{"sample_compression": "lz4"}, {"chunk_compression": "lz4"}, {}]
)
@pytest.mark.parametrize(
    "data", [[1] * 100 + [2] * 100 + [None] * 300, [None] * 300 + [3] * 200]
)
def test_empty_sample_transform_1(local_ds, compression, data):
    @deeplake.compute
    def upload(sample_in, sample_out):
        sample_out.x.append(sample_in)

    with local_ds as ds:
        ds.create_tensor("x", **compression)

        upload().eval(
            data,
            ds,
            num_workers=2,
        )
        assert len(ds.x) == 500


def test_classlabel_transform_bug(local_ds):
    @deeplake.compute
    def upload(sample_in, sample_out):
        sample_out.x.append(sample_in)

    with local_ds as ds:
        ds.create_tensor("x", htype="class_label", dtype="int32")

        upload().eval([-1], ds)

        assert len(ds.x) == 1
        np.testing.assert_array_equal(ds.x[0], -1)


@pytest.mark.slow
def test_downsample_transform(local_ds):
    with local_ds as ds:
        ds.create_tensor(
            "image", htype="image", sample_compression="jpeg", downsampling=(2, 3)
        )

        add_image().eval(list(range(10)), ds, num_workers=TRANSFORM_TEST_NUM_WORKERS)
        tensors = [
            "image",
            "_image_downsampled_2",
            "_image_downsampled_4",
            "_image_downsampled_8",
        ]
        expected_shapes = [
            (1310, 2087, 3),
            (655, 1043, 3),
            (327, 521, 3),
            (163, 260, 3),
        ]
        for tensor, shape in zip(tensors, expected_shapes):
            assert len(ds[tensor]) == 10
            for i in range(10):
                assert ds[tensor][i].shape == shape


def test_rechunk_post_transform(local_ds):
    with local_ds as ds:
        ds.create_tensor("image", htype="image", sample_compression="jpg")
        ds.create_tensor("label", htype="class_label")

    fn1().eval(list(range(100)), ds, num_workers=4)

    label_num_chunks = ds.label.chunk_engine.num_chunks

    assert label_num_chunks == 1

    image_num_chunks = ds.image.chunk_engine.num_chunks

    assert image_num_chunks == 4


def test_none_rechunk_post_transform(local_ds):
    @deeplake.compute
    def upload(stuff, ds):
        ds.abc.append(None)

    with local_ds as ds:
        ds.create_tensor("abc")

    upload().eval(list(range(100)), ds, num_workers=2)

    num_chunks = ds.abc.chunk_engine.num_chunks

    assert num_chunks == 2


@pytest.mark.slow
@pytest.mark.parametrize("scheduler", ["serial", "threaded", "processed"])
def test_transform_checkpointing(local_ds, scheduler):
    @deeplake.compute
    def upload(i, ds):
        if i == 45:
            raise Exception("test")
        ds.abc.append(i)

    @deeplake.compute
    def double(data_in, ds):
        ds.abc.append(data_in.abc * 2)

    data_in = list(range(100))

    with local_ds as ds:
        ds.create_tensor("abc")

    # not divisible by num_workers
    with pytest.raises(ValueError):
        upload().eval(
            data_in, ds, num_workers=2, scheduler=scheduler, checkpoint_interval=51
        )

    # greater than len(data_in)
    with pytest.raises(ValueError):
        upload().eval(
            data_in, ds, num_workers=2, scheduler=scheduler, checkpoint_interval=102
        )

    # less than 10% of data_in, shows warning
    with pytest.warns(UserWarning, match="10%"):
        with pytest.raises(TransformError):
            upload().eval(
                data_in, ds, num_workers=2, scheduler=scheduler, checkpoint_interval=8
            )

    assert len(ds.abc) == 40
    assert ds.abc.numpy(aslist=True) == list(range(40))
    with pytest.raises(ValueError):
        double().eval(ds, num_workers=2, scheduler=scheduler, checkpoint_interval=10)

    # fix input data
    data_in[45] = 0

    upload().eval(
        data_in[40:], ds, num_workers=2, scheduler=scheduler, checkpoint_interval=10
    )
    assert ds.abc.numpy(aslist=True) == data_in


@pytest.mark.parametrize("bad_sample_index", [10, 50])
def test_transform_checkpoint_store_data(local_ds_generator, bad_sample_index):
    @deeplake.compute
    def upload(i, ds):
        ds.abc.append(i)

    samples = list(range(100))
    samples.insert(bad_sample_index, "bad sample")

    with pytest.raises(TransformError):
        with local_ds_generator() as ds:
            ds.create_tensor("abc")
            upload().eval(
                samples,
                ds,
                num_workers=TRANSFORM_TEST_NUM_WORKERS,
                checkpoint_interval=20,
            )

    ds = local_ds_generator()

    nsamples = 0 if bad_sample_index == 10 else 40
    assert len(ds.abc) == nsamples
    last_checkpoint = ds.version_state["commit_node"].parent
    assert last_checkpoint.is_checkpoint == True
    assert last_checkpoint.total_samples_processed == nsamples


def create_test_ds(path):
    ds = deeplake.empty(path, overwrite=True)
    ds.create_tensor("images", htype="image", sample_compression="jpg")
    ds.create_tensor("boxes", htype="bbox")
    ds.create_tensor("labels", htype="class_label")
    return ds


class BadSample:
    # will pass shape check in transform tensor
    shape = (250, 250, 3)


@pytest.mark.slow
@all_schedulers
@pytest.mark.parametrize("method", ["ds", "multiple", "checkpointed"])
@pytest.mark.parametrize("error_at", ["transform", "chunk_engine"])
def test_ds_append_errors(
    local_path, compressed_image_paths, scheduler, method, error_at
):
    @deeplake.compute
    def upload(item, ds):
        images = (
            deeplake.read(item["images"])
            if isinstance(item["images"], str)
            else item["images"]
        )
        if method == "ds" or method == "checkpointed":
            ds.append(
                {
                    "labels": np.zeros(10, dtype=np.uint32),
                    "boxes": np.ones((len(item["boxes"]), 4), dtype=np.float32),
                    "images": images,
                }
            )
        elif method == "multiple":
            # test rolling back multiple samples
            ds.labels.append(np.zeros(10, dtype=np.uint32))
            ds.boxes.append(np.ones((len(item["boxes"]), 4), dtype=np.float32))
            ds.labels.append(np.zeros(10, dtype=np.uint32))
            ds.boxes.append(np.ones((len(item["boxes"]), 4), dtype=np.float32))
            ds.images.append(images)
            ds.images.append(images)

    ds = create_test_ds(local_path)

    images = compressed_image_paths["jpeg"][:2]

    samples = []
    for i in range(20):
        samples.append({"images": images[i % 2], "boxes": range(i + 1)})

    if error_at == "transform":
        # errors out in transform dataset / tensor
        bad_sample = {"images": "bad_path", "boxes": [1, 2, 3]}
        err_msg = re.escape(
            f"Transform failed at index 17 of the input data on the item: {bad_sample}."
        )
    else:
        # errors out in chunk engine
        bad_sample = {"images": BadSample(), "boxes": [1, 2, 3]}
        err_msg = re.escape(f"Transform failed at index 17 of the input data.")

    if method == "checkpointed":
        err_msg += re.escape(
            " Last checkpoint: 10 samples processed. You can slice the input to resume from this point."
        )
    err_msg += re.escape(
        " See traceback for more details."
        " If you wish to skip the samples that cause errors, please specify `ignore_errors=True`."
    )

    samples.insert(17, bad_sample)

    with pytest.raises(TransformError, match=err_msg) as e:
        upload().eval(
            samples,
            ds,
            num_workers=TRANSFORM_TEST_NUM_WORKERS,
            scheduler=scheduler,
            checkpoint_interval=10 if method == "checkpointed" else 0,
        )

    ds = create_test_ds(local_path)

    upload().eval(
        samples,
        ds,
        num_workers=TRANSFORM_TEST_NUM_WORKERS,
        scheduler=scheduler,
        ignore_errors=True,
        checkpoint_interval=10 if method == "checkpointed" else 0,
    )

    if method == "ds" or method == "checkpointed":
        assert ds["images"][::2].numpy().shape == (10, *deeplake.read(images[0]).shape)
        assert ds["images"][1::2].numpy().shape == (10, *deeplake.read(images[1]).shape)

        assert len(ds["boxes"]) == 20
        assert ds["boxes"].meta.min_shape == [1, 4]
        assert ds["boxes"].meta.max_shape == [20, 4]

        assert ds["labels"].numpy().shape == (20, 10)
    elif method == "multiple":
        data = ds["images"]

        images_0 = np.concatenate([data[i : i + 2].numpy() for i in range(0, 40, 4)])
        image_0_shape = deeplake.read(images[0]).shape
        assert images_0.shape == (20, *image_0_shape)

        images_1 = np.concatenate([data[i : i + 2].numpy() for i in range(2, 40, 4)])
        image_1_shape = deeplake.read(images[1]).shape
        assert images_1.shape == (20, *image_1_shape)

        assert len(ds["boxes"]) == 40
        assert ds["boxes"].meta.min_shape == [1, 4]
        assert ds["boxes"].meta.max_shape == [20, 4]

        assert ds["labels"].numpy().shape == (40, 10)


def test_ds_update(local_ds):
    @deeplake.compute
    def update_ds(sample_in, ds):
        i = sample_in.pop("index")
        ds[i].update(sample_in)

    with local_ds as ds:
        ds.create_tensor("abc")
        ds.create_tensor("xyz")

        ds.abc.extend([1, 2, 3, 4, 5])
        ds.xyz.extend([1, 2, 3, 4, 5])

    samples = [{"abc": 1, "xyz": 2, "index": 0}, {"abc": 3, "xyz": 4, "index": 1}]

    with pytest.raises(TransformError) as e:
        update_ds().eval(samples, ds, num_workers=TRANSFORM_TEST_NUM_WORKERS)
        assert isinstance(e.__cause__, NotImplementedError)


def test_all_samples_skipped(local_ds):
    @deeplake.compute
    def upload(stuff, ds):
        if isinstance(stuff["images"], str):
            sample = deeplake.read(stuff["images"])
        else:
            sample = stuff["images"]
        ds.images.append(sample)

    with local_ds as ds:
        ds.create_tensor("images", htype="image", sample_compression="png")

    samples = (
        [{"images": "bad_path"}] * 10
        + [{"images": BadSample()}] * 20
        + [{"images": "bad_path"}] * 10
    )

    with pytest.raises(AllSamplesSkippedError) as e:
        upload().eval(
            samples, ds, num_workers=TRANSFORM_TEST_NUM_WORKERS, ignore_errors=True
        )


def test_transform_numpy_only(local_ds):
    @deeplake.compute
    def upload(i, ds):
        ds.abc.extend(i * np.ones((10, 5, 5)))

    with local_ds as ds:
        ds.create_tensor("abc")

        upload().eval(list(range(100)), ds, num_workers=2)

    assert len(local_ds) == 1000

    for i in range(100):
        np.testing.assert_array_equal(
            ds.abc[i * 10 : (i + 1) * 10].numpy(), i * np.ones((10, 5, 5))
        )


@deeplake.compute
def add_samples(i, ds, flower_path):
    ds.abc.extend(i * np.ones((5, 5, 5)))
    ds.images.extend([deeplake.read(flower_path) for _ in range(5)])


@deeplake.compute
def mul_by_2(sample_in, samples_out):
    samples_out.abc.append(2 * sample_in.abc.numpy())
    samples_out.images.append(sample_in.images.numpy() - 1)


@pytest.mark.slow
def test_pipeline(local_ds, flower_path):
    pipeline = deeplake.compose([add_samples(flower_path), mul_by_2()])

    flower_arr = np.array(deeplake.read(flower_path))

    with local_ds as ds:
        ds.create_tensor("abc")
        ds.create_tensor("images", htype="image", sample_compression="png")

        pipeline.eval(list(range(10)), ds, num_workers=2)

    assert len(local_ds) == 50

    for i in range(10):
        np.testing.assert_array_equal(
            ds.abc[i * 5 : (i + 1) * 5].numpy(), i * 2 * np.ones((5, 5, 5))
        )
        np.testing.assert_array_equal(
            ds.images[i * 5 : (i + 1) * 5].numpy(),
            np.tile(flower_arr - 1, (5, 1, 1, 1)),
        )


def test_pad_data_in_bug(local_ds):
    @deeplake.compute
    def upload(stuff, ds):
        append_dict = {}
        for tensor in ds.tensors:
            append_dict[tensor] = stuff[tensor]

        ds.append(append_dict)

    with local_ds as ds:
        ds.create_tensor("abc", htype="class_label")
        ds.create_tensor("xyz")

        ds.abc.extend([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        ds.xyz.extend([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

    ds2 = deeplake.empty(local_ds.path + "_2", overwrite=True)
    ds2.create_tensor("abc", htype="class_label")
    ds2.create_tensor("xyz")

    upload().eval(ds, ds2, num_workers=TRANSFORM_TEST_NUM_WORKERS, pad_data_in=True)

    assert len(ds2) == 11
    np.testing.assert_array_equal(ds2.abc[:10].numpy(), ds.abc.numpy())
    np.testing.assert_array_equal(ds2.abc[10].numpy(), np.zeros((0,)))
    np.testing.assert_array_equal(ds2.xyz.numpy(), ds.xyz.numpy())

    ds2.delete()


def test_no_corruption(local_ds):
    @deeplake.compute
    def upload(stuff, ds):
        ds.append(stuff)

    with local_ds as ds:
        ds.create_tensor("images", htype="image", sample_compression="png")
        ds.create_tensor("labels", htype="class_label")

    samples = (
        [
            {
                "images": np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8),
                "labels": 1,
            }
            for _ in range(20)
        ]
        + ["bad_sample"]
    ) * 2

    with pytest.raises(TransformError):
        upload().eval(samples, ds, num_workers=TRANSFORM_TEST_NUM_WORKERS)

    assert ds.images.numpy().shape == (40, 10, 10, 3)
    assert ds.labels.numpy().shape == (40, 1)


def test_ds_append_empty(local_ds):
    @deeplake.compute
    def upload(stuff, ds):
        ds.append(stuff, append_empty=True)

    with local_ds as ds:
        ds.create_tensor("images", htype="image", sample_compression="png")
        ds.create_tensor("label1", htype="class_label")
        ds.create_tensor("label2", htype="class_label")

    samples = [
        {"images": np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8), "label1": 1}
        for _ in range(20)
    ]

    upload().eval(samples, ds, num_workers=TRANSFORM_TEST_NUM_WORKERS)

    with pytest.raises(EmptyTensorError):
        ds.label2.numpy()

    ds.label2.append(1)

    np.testing.assert_array_equal(ds.label2[:20].numpy(), np.array([]).reshape((20, 0)))


def test_catch_value_error(local_path):
    @deeplake.compute
    def upload(sample_in, ds, class_names):
        label = class_names.index(sample_in)

        ds.append(
            {
                "abc": sample_in,
                "label": label,
            }
        )

    ds = deeplake.empty(local_path, overwrite=True)
    ds.create_tensor("abc")
    ds.create_tensor("label", htype="class_label")

    class_names = [-1, 0, 1, 2, 3]

    with pytest.raises(TransformError) as e:
        upload(class_names=class_names).eval(
            [0] * 10 + [1] * 10 + [10] * 10, ds, num_workers=TRANSFORM_TEST_NUM_WORKERS
        )
        assert e.index == 20
        assert e.sample == 10


def test_transform_summary(local_ds, capsys):
    @deeplake.compute
    def upload(sample_in, sample_out):
        sample_out.images.append(sample_in)

    with local_ds as ds:
        ds.create_tensor("images", htype="image", sample_compression="jpg")

    samples = (
        ["bad_sample"]
        + [np.random.randint(0, 255, (10, 10), dtype=np.uint8) for _ in range(8)]
        + ["bad_sample"]
    ) * 2

    upload().eval(
        samples,
        ds,
        num_workers=TRANSFORM_TEST_NUM_WORKERS,
        progressbar=False,
        ignore_errors=True,
    )

    captured = capsys.readouterr()
    assert captured.out == (
        "No. of samples successfully processed: 16 (80.0%)\n"
        "No. of samples skipped: 4 (20.0%)\n"
    )

    samples = [np.random.randint(0, 255, (10, 10), dtype=np.uint8) for _ in range(8)]
    upload().eval(
        samples,
        ds,
        num_workers=TRANSFORM_TEST_NUM_WORKERS,
        progressbar=False,
        ignore_errors=True,
    )

    captured = capsys.readouterr()
    assert captured.out == (
        "No. of samples successfully processed: 8 (100.0%)\n"
        "No. of samples skipped: 0 (0.0%)\n"
    )

    # no summary if ignore_errors=False
    samples = [np.random.randint(0, 255, (10, 10), dtype=np.uint8) for _ in range(8)]
    upload().eval(
        samples, ds, num_workers=TRANSFORM_TEST_NUM_WORKERS, progressbar=False
    )

    captured = capsys.readouterr()
    assert captured.out == ""


def test_transform_extend(local_ds):
    # skip_ok is not supported with extend
    @deeplake.compute
    def bad_upload(batch, ds):
        ds.extend(batch, skip_ok=True)

    @deeplake.compute
    def upload(batch, ds):
        ds.extend(batch, append_empty=True)

    with local_ds as ds:
        ds.create_tensor("abc")
        ds.create_tensor("xyz")
        ds.abc.append(np.ones((10, 10)))
        ds.xyz.append(1)

    batches = [{"abc": np.ones((5, 10, 10))} for _ in range(10)]

    with pytest.raises(TransformError):
        bad_upload().eval(batches, ds, num_workers=TRANSFORM_TEST_NUM_WORKERS)

    ds.xyz.append(2)
    # unequal tensor lengths
    with pytest.raises(TransformError):
        upload().eval(batches, ds, num_workers=TRANSFORM_TEST_NUM_WORKERS)

    ds.abc.append(np.ones((10, 10)))
    # items should be dicts
    with pytest.raises(TransformError):
        upload().eval([1, 2, 3, 4, 5], ds, num_workers=TRANSFORM_TEST_NUM_WORKERS)

    upload().eval(batches, ds, num_workers=TRANSFORM_TEST_NUM_WORKERS)

    assert len(ds) == 52
    assert ds.abc.numpy().shape == (52, 10, 10)
    assert ds.xyz.shape == (52, None)
    assert ds.xyz[:2].numpy().shape == (2, 1)
    assert ds.xyz[2:].numpy().shape == (50, 0)
