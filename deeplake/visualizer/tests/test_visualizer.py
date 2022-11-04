import numpy as np
import requests
from deeplake.visualizer.visualizer import visualizer
from deeplake.tests.dataset_fixtures import *
import deeplake as dp


@pytest.mark.parametrize(
    "ds_generator",
    [
        "local_ds_generator",
        "s3_ds_generator",
        "gcs_ds_generator",
        "hub_cloud_ds_generator",
    ],
    indirect=True,
)
def test_local_server(ds_generator):
    ds = ds_generator()
    ds.create_tensor("images", htype="image", sample_compression="jpg")
    ds.images.append(np.random.randint(0, 255, size=(400, 400, 3), dtype="uint8"))
    ds.add_creds_key("my_creds")
    temp_creds = {"access_key": "serious_stuff", "secret": "very_secret_stuff"}
    ds.populate_creds("my_creds", temp_creds)
    id = visualizer.add(ds.storage)
    link_creds_id = visualizer.add_link_creds(ds.link_creds)
    assert visualizer.start_server() == f"http://localhost:{visualizer.port}/"
    assert visualizer.is_server_running
    url = f"http://localhost:{visualizer.port}/{id}/"
    response = requests.request("HEAD", url + "dataset_meta.json")
    assert response.status_code == 200
    response = requests.request("HEAD", url + "not_exists/not_exists")
    assert response.status_code == 404
    response = requests.request("GET", url + "dataset_meta.json")
    assert response.status_code == 206
    j = response.json()
    assert "tensors" in j
    assert "images" in j["tensors"]

    response = requests.request("GET", url + "images/tensor_meta.json")
    assert response.status_code == 206
    j = response.json()
    assert j["sample_compression"] == "jpeg"
    assert j["length"] == 1
    assert j["htype"] == "image"
    assert j["dtype"] == "uint8"

    response = requests.request("GET", url + "images/chunks_index/unsharded")
    assert response.status_code == 206
    c1 = response.content
    assert len(c1) == 18 + len(dp.__version__)
    response = requests.request(
        "GET", url + "images/chunks_index/unsharded", headers={"Range": "bytes 2-3;"}
    )
    assert response.status_code == 206
    c2 = response.content
    assert len(c2) == 2
    assert c1[2] == c2[0]
    assert c1[3] == c2[1]

    response = requests.request("GET", url + "not_exists/not_exists")
    assert response.status_code == 404

    creds_url = f"http://localhost:{visualizer.port}/creds/{link_creds_id}/"
    response = requests.request("GET", creds_url + "my_creds")
    r = response.json()
    assert len(r.keys()) == len(temp_creds.keys())
    for k in r.keys():
        assert r[k] == temp_creds[k]

    response = requests.request("GET", creds_url + "not_my_creds")
    assert response.status_code == 404
