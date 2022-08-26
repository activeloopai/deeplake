from torchvision import transforms


def repeat_shape(images_tensor, dataloader_params):
    repeat_tform = transforms.Lambda(lambda x: x.repeat(int(3 / x.shape[0]), 1, 1))

    if dataloader_params is None:
        dataloader_params, dataloader_params["transform"] = {}, {}
        dataloader_params["transform"][images_tensor] = repeat_tform

    elif (
        "transform" in dataloader_params
        and images_tensor not in dataloader_params["transform"]
    ):
        dataloader_params["transform"][images_tensor] = repeat_tform

    else:

        tform = transforms.Compose(
            [dataloader_params["transform"][images_tensor], repeat_tform]
        )

        dataloader_params["transform"][images_tensor] = tform

    return dataloader_params
