import pytest

from deeplake.constants import MB
import numpy as np
import random
import deeplake
import json


@pytest.mark.slow
def test_rechunk(local_ds):
    with local_ds as ds:
        ds.create_tensor("abc")
        for _ in range(10):
            ds.abc.append(np.ones((10, 10)))
        for i in range(5, 10):
            ds.abc[i] = np.ones((1000, 1000))

        for i in range(10):
            target = np.ones((10, 10)) if i < 5 else np.ones((1000, 1000))
            np.testing.assert_array_equal(ds.abc[i].numpy(), target)

        original_num_chunks = ds.abc.chunk_engine.num_chunks
        assert original_num_chunks == 3
        ds.rechunk()
        new_num_chunks = ds.abc.chunk_engine.num_chunks
        assert new_num_chunks == 6

        assert len(ds.abc) == 10
        for i in range(10):
            target = np.ones((10, 10)) if i < 5 else np.ones((1000, 1000))
            np.testing.assert_array_equal(ds.abc[i].numpy(), target)
        assert len(ds.abc) == 10

        ds.create_tensor("xyz")
        for _ in range(10):
            ds.xyz.append(np.ones((1000, 1000)))

        assert len(ds.xyz) == 10
        for i in range(10):
            ds.xyz[i] = np.ones((100, 100))

        original_num_chunks = ds.xyz.chunk_engine.num_chunks
        assert original_num_chunks == 1
        assert len(ds.xyz) == 10
        ds.rechunk("xyz")
        new_num_chunks = ds.xyz.chunk_engine.num_chunks
        assert new_num_chunks == 1

        ds.create_tensor("compr", chunk_compression="lz4")
        for _ in range(100):
            ds.compr.append(np.random.randint(0, 255, size=(175, 350, 3)))

        assert len(ds.compr) == 100
        for i in range(100):
            ds.compr[i] = np.random.randint(0, 3, size=(10, 10, 10))
        assert len(ds.compr) == 100
        for i in range(100):
            ds.compr[i] = np.random.randint(0, 255, size=(175, 350, 3))
        assert len(ds.compr) == 100


def test_rechunk_2(local_ds):
    with local_ds as ds:
        ds.create_tensor("compr", dtype="int64")
        for _ in range(100):
            ds.compr.append(np.random.randint(0, 255, size=(175, 350, 3)))

        assert len(ds.compr) == 100
        for i in range(100):
            ds.compr[i] = np.random.randint(0, 3, size=(10, 10, 10))
        assert len(ds.compr) == 100
        assert ds.compr.chunk_engine.num_chunks == 1


def test_rechunk_3(local_ds):
    NUM_TEST_SAMPLES = 100
    deeplake.constants._ENABLE_RANDOM_ASSIGNMENT = True
    test_sample = np.random.randint(0, 255, size=(600, 600, 3), dtype=np.uint8)
    with local_ds as ds:
        ds.create_tensor("test", dtype="uint8")
        r = list(range(NUM_TEST_SAMPLES))
        random.seed(20)
        random.shuffle(r)
        for i in r:
            ds.test[i] = test_sample


state_json = "[3, [2968816850, 431958457, 3847918079, 920773653, 1183269698, 3333547174, 3999706037, 1600744715, 2669055074, 1706523597, 1238696894, 1203705021, 3434659079, 2721990810, 660905437, 1191778566, 862008228, 2427091368, 2655392196, 1399784847, 1127214736, 2143117458, 3635781574, 487195879, 1363754106, 907651704, 1281530659, 1083761032, 1645929773, 541031376, 1982471788, 261820361, 2474899712, 3080157925, 1028166783, 3658443606, 1221585999, 2592453311, 3964449013, 3105884803, 1808181838, 3435841320, 2361241581, 1170598344, 3073508480, 4261542533, 2909087953, 3399413119, 2489477194, 726375815, 2131337135, 1528702063, 167542317, 2794159906, 3852693278, 3613909660, 2834957178, 3793140572, 2726155373, 1050401223, 117835526, 3052361156, 3217924937, 3471647280, 1597845398, 4103109978, 3789546747, 4062429700, 4146175419, 818066899, 3736037478, 725408823, 4226452293, 1183542338, 835382304, 3233157926, 2416413477, 3702267425, 1569851580, 3732237547, 262892585, 3689353066, 2098750845, 2040005579, 1136853512, 1502464462, 1777524963, 2618333477, 2436486401, 1569676253, 3627265984, 325180196, 2113005301, 817615225, 2962295101, 375924726, 3718445534, 2279480893, 1037239560, 917786365, 957338238, 2151191932, 3415643009, 110487343, 1867573783, 3583571837, 2580933075, 959168037, 1907666394, 2253743682, 500472427, 149533349, 2247015961, 1719097563, 1306410297, 1997775245, 2201678829, 2927508311, 164026485, 2981180937, 2933317093, 2433532212, 138627514, 1667933307, 2884346874, 3892941191, 3415848146, 3898709064, 3477970589, 3139746372, 3009168051, 1514295675, 4265816852, 2700623184, 492162904, 1335965144, 3126903160, 1858110560, 2057420470, 946796336, 4246181075, 2351350904, 247095999, 297163097, 434169658, 2961647993, 587010360, 3458903972, 3254985060, 714773714, 1181220788, 1828500662, 548775665, 3430424222, 4167202923, 224300373, 1713350212, 2605328497, 926344259, 3091562428, 1798073330, 1093984950, 2101877576, 2684706546, 4286439368, 2383785642, 1392797839, 3610917621, 1309823001, 738619683, 206431818, 4293012910, 1035359031, 3356076176, 2077429285, 2076545129, 3507387271, 4223449480, 381210301, 1433015687, 3125723175, 2464304890, 646161800, 2245403546, 3082606507, 3497509210, 3801951349, 2396421350, 3361722208, 717197667, 3921475366, 2631698227, 4137525437, 2617872221, 1867412737, 3060405755, 2476986291, 2077208014, 3456876274, 1867217480, 2596595964, 980265161, 2040790606, 3024275403, 2671508469, 3572659345, 3619685495, 932459772, 108669208, 3891626345, 1864096802, 2387543341, 2467331637, 2086255756, 1489222066, 3986426318, 1978387952, 337145338, 1041840027, 2835672809, 3847930424, 2482827576, 511280181, 631577380, 4071205065, 2672492832, 2037225172, 1340096038, 3481114236, 377972666, 3028528639, 880010493, 2716210357, 517891927, 1364510586, 1423149719, 722380462, 927628354, 2675708928, 1565348538, 3955317752, 2687180997, 3233202176, 1095891725, 4271170096, 8660952, 844189839, 3259773026, 135964880, 3045014153, 3523353340, 2724909332, 953080134, 2195159353, 2000688528, 3605887026, 2695787617, 2075152434, 780447473, 3788723387, 4243452227, 2239860364, 3327329107, 2827451849, 4248334587, 928691463, 1330524739, 1696266944, 3846552858, 3548966813, 3931413317, 3088683145, 987649708, 3176258324, 402390526, 3436600831, 830715685, 2925497332, 485034341, 1062553349, 3189585664, 1544564715, 1156630401, 3080421830, 1638959544, 2395133163, 1525052532, 2091899573, 756402703, 3670165568, 3742379501, 3758451768, 3099476036, 2018925245, 2035907712, 4235165130, 235332165, 3141712875, 1864168550, 3122710145, 3574915384, 2207914672, 3495242283, 3585972346, 1492473581, 2029331589, 3374108444, 3031965359, 2516873798, 2359880339, 278614617, 866341224, 555253435, 3620952705, 2022870201, 3674645572, 3285523308, 29273653, 1080897666, 4249527990, 2469877256, 4000034927, 3379643741, 413532346, 2989155103, 528961423, 1470320532, 36461535, 3377565300, 2187192427, 379454561, 3669762915, 2945558006, 3539620136, 279466236, 4287218235, 348458251, 2852037207, 3588976106, 789625068, 1241419409, 3415466153, 1255634812, 2977282009, 1492571279, 1801637264, 80745935, 2865741220, 3007830692, 4077864741, 4201716559, 1832447912, 1568397490, 2947236761, 2257166944, 378909525, 3071967460, 2621933059, 3483594045, 3253997895, 1115465227, 1236174644, 822694611, 1579848671, 944850524, 1932720163, 871076109, 1266236331, 1905534914, 843429232, 1760289335, 2183914985, 174622510, 679615266, 97406504, 1540358701, 3459737283, 3302460591, 137387572, 3662078630, 1562722201, 3919284131, 876009268, 1841865865, 3688019530, 598348714, 3486649980, 501081913, 4036698561, 1253234116, 4154700213, 3903047613, 328639512, 1773338136, 3849405181, 1885174350, 1563213397, 3816172589, 2496048832, 1573513442, 999717432, 2173133521, 1473907714, 462242591, 819184928, 2069999561, 30962715, 1072381842, 624950321, 1847911910, 1656677615, 207940511, 1354333881, 3714102430, 826137826, 3250538791, 361418714, 1554515668, 3585861539, 1846106123, 2377968951, 25285774, 1507232799, 4071273741, 1156231246, 1744430138, 523018058, 1848674213, 1090050218, 2748938699, 2492751030, 3880392889, 3708185546, 4095009209, 3981544091, 2219660577, 100343492, 3238528050, 3254014806, 1746906770, 978787723, 2893565075, 1925357899, 4282034486, 3953815381, 372551821, 388274557, 2285825757, 2324467082, 4249279208, 3800233824, 1109968902, 85807688, 3880723083, 4203513269, 302025155, 3089507976, 3616904597, 3258639180, 3827134948, 3237221787, 1583005461, 3710788986, 1522576412, 2933449353, 3049258443, 3814012930, 3538677396, 4003784590, 3762531975, 599050420, 4017218987, 791587229, 3662034378, 1721982982, 417774824, 1733683175, 237527310, 4271266287, 3968311575, 2171465595, 77042717, 3276868520, 1256394578, 2187423251, 742377209, 1046331351, 2169488005, 3783588365, 2963516767, 2089956317, 4063397688, 1855310102, 1047694362, 3131180012, 150826672, 3776469023, 1007148605, 3651019347, 1395348536, 2107397804, 2668530275, 354992822, 2800544860, 2823645010, 4169302982, 2554146124, 1512228546, 3043260708, 1771592953, 1520156387, 2140335665, 602127123, 565915844, 165988844, 1060393175, 3566559505, 3630093037, 3369680940, 2115064893, 3092893654, 4042965700, 2657207125, 2747277058, 1268710566, 1926124671, 3081419055, 47305475, 1816639007, 2116219533, 2861920779, 2248154127, 632356954, 1109449007, 1091499796, 3507453098, 824658085, 3604486540, 566941974, 1276127539, 125759458, 1799473365, 435808977, 3628051677, 643686390, 1698991679, 809908741, 89531519, 2216577288, 1237531493, 1654598707, 1852039436, 2403607291, 1445461232, 1563692368, 1196184680, 4222018552, 1652109242, 1474459779, 961948380, 2523431910, 3431618821, 3260054037, 67464367, 3175674921, 3159487806, 1111689649, 2199160083, 2404034127, 871421595, 56753711, 2645519241, 1038662576, 2521011573, 1659347930, 2421770774, 2591278606, 3190926254, 10573582, 1772508679, 303337957, 448506681, 2015545829, 3998663892, 194347, 1284434421, 3396359767, 2015447499, 3116624524, 2308507369, 3951895905, 553324237, 1187217007, 3623765420, 2665852413, 1150920945, 2038185487, 2901323853, 393391334, 69190108, 1175343015, 1131848533, 2959193680, 73055199, 3845689890, 2988415822, 1657083654, 1430172899, 1839034063, 2747388955, 2266327841, 2208646566, 749538176, 2241592715, 2658942134, 2271884215, 2516771389, 2929759575, 2738509550, 2939437564, 384775706, 1623627328, 2041819626, 1163327548, 607], null]"


@pytest.mark.slow
def test_rechunk_4(local_ds):
    NUM_TEST_SAMPLES = 1000
    MIN_SAMPLE_SIZE_MB = 0.1
    MAX_SAMPLE_SIZE_MB = 4
    deeplake.constants._ENABLE_RANDOM_ASSIGNMENT = True
    random.seed(1337)
    state = json.loads(state_json)
    random.setstate((state[0], tuple(state[1]), state[2]))

    with local_ds as ds:
        ds.create_tensor(
            "test",
            dtype="uint8",
            chunk_compression="lz4",
            max_chunk_size=32 * MB,
            tiling_threshold=16 * MB,
        )
        r = list(range(NUM_TEST_SAMPLES))
        random.shuffle(r)
        with ds:
            for i in r:
                test_sample_width = random.randint(
                    int((MIN_SAMPLE_SIZE_MB * 1e6 / 3) ** 0.5),
                    int((MAX_SAMPLE_SIZE_MB * 1e6 / 3) ** 0.5),
                )
                test_sample_r = np.random.randint(
                    0,
                    255,
                    size=(test_sample_width, test_sample_width, 3),
                    dtype=np.uint8,
                )
                ds.test[i] = test_sample_r


@deeplake.compute
def add_sample_in(sample_in, samples_out):
    samples_out.abc.append(sample_in)


def test_rechunk_text(local_ds_generator):
    with local_ds_generator() as ds:
        ds.create_tensor("abc", "text")
        add_sample_in().eval(
            ["hello", "world", "abc", "def", "ghi", "yo"],
            ds,
            num_workers=2,
            disable_rechunk=True,
        )

        assert len(ds.abc.chunk_engine.chunk_id_encoder.array) == 2
        ds.abc[0] = "bye"
        np.testing.assert_array_equal(
            ds.abc.numpy(),
            np.array([["bye"], ["world"], ["abc"], ["def"], ["ghi"], ["yo"]]),
        )
        assert len(ds.abc.chunk_engine.chunk_id_encoder.array) == 1

    ds = local_ds_generator()
    np.testing.assert_array_equal(
        ds.abc.numpy(),
        np.array([["bye"], ["world"], ["abc"], ["def"], ["ghi"], ["yo"]]),
    )
    assert len(ds.abc.chunk_engine.chunk_id_encoder.array) == 1


def test_rechunk_json(local_ds_generator):
    with local_ds_generator() as ds:
        ds.create_tensor("abc", "json")
        add_sample_in().eval(
            [{"one": "if"}, {"two": "elif"}, {"three": "else"}],
            ds,
            num_workers=2,
            disable_rechunk=True,
        )

        assert len(ds.abc.chunk_engine.chunk_id_encoder.array) == 2
        ds.abc[0] = {"four": "finally"}
        np.testing.assert_array_equal(
            ds.abc.numpy(),
            np.array([[{"four": "finally"}], [{"two": "elif"}], [{"three": "else"}]]),
        )
        assert len(ds.abc.chunk_engine.chunk_id_encoder.array) == 1

    ds = local_ds_generator()
    np.testing.assert_array_equal(
        ds.abc.numpy(),
        np.array([[{"four": "finally"}], [{"two": "elif"}], [{"three": "else"}]]),
    )
    assert len(ds.abc.chunk_engine.chunk_id_encoder.array) == 1


def test_rechunk_list(local_ds_generator):
    with local_ds_generator() as ds:
        ds.create_tensor("abc", "list")
        add_sample_in().eval(
            [["hello", "world"], ["abc", "def", "ghi"], ["yo"]],
            ds,
            num_workers=2,
            disable_rechunk=True,
        )

        assert len(ds.abc.chunk_engine.chunk_id_encoder.array) == 2
        ds.abc[0] = ["bye"]
        np.testing.assert_array_equal(ds.abc[0].numpy(), np.array(["bye"]))
        np.testing.assert_array_equal(
            ds.abc[1].numpy(), np.array(["abc", "def", "ghi"])
        )
        np.testing.assert_array_equal(ds.abc[2].numpy(), np.array(["yo"]))
        assert len(ds.abc.chunk_engine.chunk_id_encoder.array) == 1

    ds = local_ds_generator()
    np.testing.assert_array_equal(ds.abc[0].numpy(), np.array(["bye"]))
    np.testing.assert_array_equal(ds.abc[1].numpy(), np.array(["abc", "def", "ghi"]))
    np.testing.assert_array_equal(ds.abc[2].numpy(), np.array(["yo"]))
    assert len(ds.abc.chunk_engine.chunk_id_encoder.array) == 1


def test_rechunk_link(local_ds_generator, cat_path, flower_path, color_image_paths):
    dog_path = color_image_paths["jpeg"]
    with local_ds_generator() as ds:
        ds.create_tensor("abc", "link[image]", sample_compression="jpg")
        add_sample_in().eval(
            [
                deeplake.link(dog_path),
                deeplake.link(flower_path),
                deeplake.link(cat_path),
            ],
            ds,
            num_workers=2,
        )
        assert len(ds.abc.chunk_engine.chunk_id_encoder.array) == 2
        ds.abc[0] = deeplake.link(cat_path)

        assert ds.abc[0].numpy().shape == (900, 900, 3)
        assert ds.abc[1].numpy().shape == (513, 464, 4)
        assert ds.abc[2].numpy().shape == (900, 900, 3)
        assert len(ds.abc.chunk_engine.chunk_id_encoder.array) == 1

    ds = local_ds_generator()
    assert ds.abc[0].numpy().shape == (900, 900, 3)
    assert ds.abc[1].numpy().shape == (513, 464, 4)
    assert ds.abc[2].numpy().shape == (900, 900, 3)
    assert len(ds.abc.chunk_engine.chunk_id_encoder.array) == 1
    assert ds.abc.chunk_engine.creds_encoder.num_samples == 3


@pytest.mark.slow
def test_rechunk_cloud_link(local_ds_generator):
    s3_path_1 = "s3://test-bucket/test-1.jpeg"
    s3_path_2 = "s3://test-bucket/test-2.jpeg"
    with local_ds_generator() as ds:
        ds.create_tensor(
            "abc",
            htype="link[image]",
            sample_compression="jpeg",
            create_shape_tensor=False,
            create_sample_info_tensor=False,
            verify=False,
        )
        ds.add_creds_key("my_s3_key_1")
        ds.add_creds_key("my_s3_key_2")
        ds.populate_creds("my_s3_key_1", {})
        ds.populate_creds("my_s3_key_2", {})

        add_sample_in().eval(
            [deeplake.link(s3_path_1, "my_s3_key_1")] * 3, ds, num_workers=2
        )
        assert len(ds.abc.chunk_engine.chunk_id_encoder.array) == 2
        ds.abc[0] = deeplake.link(s3_path_2, "my_s3_key_2")

        assert len(ds.abc.chunk_engine.chunk_id_encoder.array) == 1

        sample_0 = ds.abc[0]._linked_sample()
        assert sample_0.path == s3_path_2, sample_0.creds_key == "my_s3_key_2"

        sample_1 = ds.abc[1]._linked_sample()
        assert sample_1.path == s3_path_1, sample_1.creds_key == "my_s3_key_1"

        sample_2 = ds.abc[2]._linked_sample()
        assert sample_2.path == s3_path_1, sample_2.creds_key == "my_s3_key_1"

    ds = local_ds_generator()
    assert len(ds.abc.chunk_engine.chunk_id_encoder.array) == 1
    sample_0 = ds.abc[0]._linked_sample()
    assert sample_0.path == s3_path_2, sample_0.creds_key == "my_s3_key_2"

    sample_1 = ds.abc[1]._linked_sample()
    assert sample_1.path == s3_path_1, sample_1.creds_key == "my_s3_key_1"

    sample_2 = ds.abc[2]._linked_sample()
    assert sample_2.path == s3_path_1, sample_2.creds_key == "my_s3_key_1"

    assert ds.abc.chunk_engine.creds_encoder.num_samples == 3


@deeplake.compute
def add_samples(sample_in, samples_out):
    samples_out.labels.append(np.ones((200,), dtype=np.int64))


def test_rechunk_vc_bug(local_ds):
    ds = local_ds
    with ds:
        ds.create_tensor("labels", dtype="int64")
    add_samples().eval(list(range(200)), ds, num_workers=2)
    ds.commit()
    add_samples().eval(list(range(100)), ds, num_workers=2)
    ds.commit()
    ds.checkout("alt", True)
    ds.labels[8] = ds.labels[8].numpy()
    ds.commit()
    np.testing.assert_array_equal(
        ds.labels.numpy(), np.ones((300, 200), dtype=np.int64)
    )
    ds.checkout("main")
    np.testing.assert_array_equal(
        ds.labels.numpy(), np.ones((300, 200), dtype=np.int64)
    )


def test_rechunk_text_like_lz4(local_ds):
    @deeplake.compute
    def upload(stuff, ds):
        ds.append(stuff)

    with local_ds as ds:
        ds.create_tensor("text", htype="text", chunk_compression="lz4")
        ds.create_tensor("json", htype="json", chunk_compression="lz4")
        ds.create_tensor("list", htype="list", chunk_compression="lz4")

        samples = [{"text": "hello", "json": {"a": 1, "b": 3}, "list": [1, 2, 3]}] * 10
        samples[8] = {
            "text": "hello world",
            "json": {"a": 2, "b": 4},
            "list": [4, 5, 6],
        }

        upload().eval(samples, ds, num_workers=2, disable_rechunk=True)

    assert ds.text.chunk_engine.num_chunks == 2
    assert ds.json.chunk_engine.num_chunks == 2
    assert ds.list.chunk_engine.num_chunks == 2

    ds.pop()

    assert ds.text.chunk_engine.num_chunks == 1
    assert ds.json.chunk_engine.num_chunks == 1
    assert ds.list.chunk_engine.num_chunks == 1

    assert ds.text[-1].data()["value"] == "hello world"
    assert ds.json[-1].data()["value"] == {"a": 2, "b": 4}
    assert ds.list[-1].data()["value"] == [4, 5, 6]
