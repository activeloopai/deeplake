from torchvision import transforms


def repeat_shape(images_tensor, transform):
    """
    This function adds additional lambda transform to the transform object.
    This transofrm repeats the values of the images tensor and brings each image to the same number of channels.
    """
    repeat_tform = transforms.Lambda(lambda x: x.repeat(int(3 / x.shape[0]), 1, 1))

    if transform is None:
        transform = {images_tensor: repeat_tform}

    elif isinstance(transform, dict) and images_tensor not in transform:
        transform[images_tensor] = repeat_tform

    else:
        transform[images_tensor] = transforms.Compose(
            [transform[images_tensor], repeat_tform]
        )

    return transform
