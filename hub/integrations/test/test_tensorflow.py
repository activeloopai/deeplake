import hub
import tensorflow

def test_tensorflow():
    print('testing tensorflow')

    # Create arrays
    images = hub.array((10, 100, 100), name='name1', dtype='uint8')
    labels = hub.array((10, 1), name='name2', dtype='uint8')

    # Create dataset
    ds = hub.dataset({
        'images': images,
        'labels': labels
    }, name='test/loaders:dataset')

    # Transform to Pytorch
    train_dataset = ds.to_tensorflow()

    assert len(train_dataset) == 10

    for image, label in train_dataset:
        assert len(image) == 100
        assert len(image[0]) == 100
        assert len(label) == 1        

    train_dataset = train_dataset.batch(32, drop_remainder=True);

    # Loop over attributes
    for _, (images, labels) in train_dataset.enumerate():
        # assert len(images) == 32
        # assert len(labels) == 32
        assert len(images[0]) == 100
        assert len(images[0][0]) == 100
        assert len(labels[0]) == 1
        break

    print('pass')
