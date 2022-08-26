
from torchvision import transforms


def repeat_shape(images_tensor, transform):

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
