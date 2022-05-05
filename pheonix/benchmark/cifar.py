import hub
import tqdm


def hub_dataset(
    path: str,
    batch_size: int = 128,
    local: bool = True,
    local_path: str = "./data/tmp_dataset",
):
    # copy the dataset to local machine
    if not hub.exists(local_path) and local:
        hub.copy(path, local_path)
    else: 
        local_path = path

    ds = hub.load(local_path)

    return ds.pytorch(batch_size=batch_size)

def loop(dataloader):
    for el in tqdm.tqdm(dataloader):
        # TODO add model pass
        pass


if __name__ == "__main__":
    dataloader = hub_dataset("hub://activeloop/cifar100-train", local=False)
    loop(dataloader)
