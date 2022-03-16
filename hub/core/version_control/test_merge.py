def test_merge(local_ds):
    with local_ds as ds:
        ds.create_tensor("image")
        ds.image.append(1)
        a = ds.commit()
        ds.image[0] = 2
        b = ds.commit()
        assert ds.image[0].numpy() == 2
        ds.checkout(a)
        ds.checkout("alt", create=True)
        ds.image[0] = 3
        assert ds.image[0].numpy() == 3
        f = ds.commit()

        ds.checkout("main")
        assert ds.image[0].numpy() == 2

        ds.merge(f, conflict_resolution="theirs")
        assert ds.image[0].numpy() == 3
        c = ds.commit()

        ds.image[0] = 4
        assert ds.image[0].numpy() == 4
        d = ds.commit()
        ds.checkout("alt")
        assert ds.image[0].numpy() == 3

        ds.image[0] = 0
        assert ds.image[0].numpy() == 0
        g = ds.commit()

        ds.merge("main", conflict_resolution="theirs")
        assert ds.image[0].numpy() == 4
        h = ds.commit()

        ds.image[0] = 5
        assert ds.image[0].numpy() == 5
        ds.image.append(10)
        i = ds.commit()

        ds.image[0] = 6
        assert ds.image[0].numpy() == 6

        ds.checkout("main")
        assert ds.image[0].numpy() == 4

        ds.merge("alt", conflict_resolution="theirs")
        assert ds.image[0].numpy() == 6
        assert ds.image[1].numpy() == 10
