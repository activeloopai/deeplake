import hub
import torch

def test_pytorch():
    print('testing pytorch')

    # Create arrays
    images = hub.array((100, 100, 100),
                       name='test/dataloaders:images', dtype='uint8')
    labels = hub.array(
        (100, 1), name='test/dataloaders:labels', dtype='uint8')

    # Create dataset
    ds = hub.dataset({
        'images': images,
        'labels': labels
    }, name='test/loaders:dataset')

    # Transform to Pytorch
    train_dataset = ds.to_pytorch()

    # Create data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, num_workers=1,
        pin_memory=True
    )

    # Loop over attributes
    for i, (images, labels) in enumerate(train_loader):
        print(images.shape, labels.shape)
        assert (images.shape == (32, 100, 100))
        assert (labels.shape == (32, 1))
        break

    print('pass')
