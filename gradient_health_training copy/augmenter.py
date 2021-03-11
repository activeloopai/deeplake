from imgaug import augmenters as iaa

augmenter = iaa.Sequential(
    [
        iaa.Fliplr(0.5),
    ],
    random_order=True,
)
