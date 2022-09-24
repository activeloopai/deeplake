import numpy as np
import deeplake
import wandb


def test_wandb(hub_cloud_path, hub_cloud_dev_token):
    run = wandb.init(mode="offline")
    ds = deeplake.empty(hub_cloud_path, token=hub_cloud_dev_token, overwrite=True)
    with ds:
        ds.create_tensor("image", htype="image", sample_compression="jpeg")
        ds.create_tensor("label")
        for _ in range(100):
            ds.image.append(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
            ds.label.append(np.random.randint(0, 2, dtype=np.uint8))

    run.finish()
    run = wandb.init(mode="offline")
    ds = deeplake.load(hub_cloud_path, token=hub_cloud_dev_token)
    ds.image[0].numpy()
    run.finish()
