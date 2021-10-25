def test_sign_wall(hub_cloud_ds_generator):
    ds = hub_cloud_ds_generator()

    ds.create_tensor("images")
    ds.images.append([1, 2, 3])

    ds.info.terms_of_access = "access is only grantable to nerds"

    pass
