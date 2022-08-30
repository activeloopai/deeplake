from torchvision import transforms

from hub.util.dataset import map_tensor_keys


def repeat_image_shape(images_tensor, transform):
    """
    This function adds an additional lambda transformation to the `transform`.
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


def is_image_tensor(image_tensor_htype):
    supported_image_htypes = set(
        ["image", "image.rgb", "image.gray", "generic"],
    )
    return image_tensor_htype in supported_image_htypes


def is_label_tensor(label_tensor_htype):
    supported_label_htypes = set(
        ["class_label", "generic"],
    )
    return label_tensor_htype in supported_label_htypes


def assert_valid_tensors(dataset, images_tensor, labels_tensor):
    """Asserts that the images tensor and labels tensor are valid."""
    image_tensor_htype, label_tensor_htype = (
        dataset[images_tensor].htype,
        dataset[labels_tensor].htype,
    )

    if not is_image_tensor(image_tensor_htype):
        raise TypeError(
            f'The images tensor has an unsupported htype: {image_tensor_htype}. In general, the images tensor must be of type "image".'
        )

    if not is_label_tensor(label_tensor_htype):
        raise TypeError(
            f'The labels tensor has an unsupported htype: {label_tensor_htype}. In general, the labels tensor must be of type "class_label".'
        )


def get_dataset_tensors(dataset, transform, tensors):
    """
    This function returns the tensors of a dataset. If `tensors` list is not provided, it will try to get them from the `transform`.
    """

    if tensors is not None:
        tensors = map_tensor_keys(dataset, tensors)

    # Try to get the tensors from the transform.
    elif transform and isinstance(transform, dict):
        tensors = map_tensor_keys(
            dataset,
            [k for k in transform.keys() if k != "index"],
        )

    # Map the images and labels tensors.
    try:
        images_tensor, labels_tensor = tensors
    except ValueError:
        raise ValueError(
            "Could not find the images and labels tensors. Please provide the images and labels tensors in `tensors` or `transform`."
        )

    assert_valid_tensors(
        dataset=dataset, images_tensor=images_tensor, labels_tensor=labels_tensor
    )

    return [images_tensor, labels_tensor]
