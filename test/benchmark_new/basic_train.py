import time
import torch
import hub
from hub.schema import Tensor
from helper import report
import numpy as np
from PIL import Image
from pathlib import Path
import os
import tensorflow as tf

from torch import nn, optim
import torchvision.models as models


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == "resnet101":
        """ Resnet18
        """
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_1(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(
            512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size


class PytorchDataset(torch.utils.data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(
        self,
        samples,
        width=256,
        load_image=True,
        image_path="results/Parallel150KB.png",
    ):
        "Initialization"
        self.samples = samples
        self.width = width
        self.load_image = load_image
        self.image_path = image_path

    def __len__(self):
        "Denotes the total number of samples"
        return self.samples

    def __getitem__(self, index):
        "Generates one sample of data"
        if self.load_image:
            folder = Path(__file__).parent
            path = os.path.join(folder, self.image_path)
            with open(path, "rb") as f:
                img = Image.open(f)
                inp = img.convert("RGB")
                inp = np.array(inp)[: self.width, : self.width]
                inp = np.transpose(inp, (2, 0, 1))
        else:
            inp = np.random.rand(3, self.width, self.width)
            inp = (255 * inp).astype("uint8")
        # objs = {"input": inp, "label": np.random.rand(1).astype("uint8")}

        # objs = {k: torch.tensor(v) for k, v in objs.items()}
        objs = (torch.tensor(inp), torch.tensor(
            np.random.rand(1).astype("uint8")))
        return objs

    def collate_fn(self, batch):
        batch = tuple(batch)
        keys = tuple(batch[0].keys())
        ans = {key: [item[key] for item in batch] for key in keys}

        for key in keys:
            ans[key] = torch.stack(ans[key], dim=0, out=None)
        return ans


def get_dataset_from_hub(samples=1, read_from_fs=False, pytorch=False):
    """
    Build dataset and transform to pytorch or tensorflow
    """
    my_schema = {"img": Tensor(shape=(256, 256, 3)), "label": "uint8"}
    ds = hub.Dataset("kristina/benchmarking",
                     shape=(samples,), schema=my_schema)
    if read_from_fs:
        ds = hub.Dataset("./tmp/benchmarking",
                         shape=(samples,), schema=my_schema)
    ds = ds.to_pytorch() if pytorch else ds.to_tensorflow()
    return ds


def TensorflowDataset(samples=100, load_image=False, image_path=""):
    def tf_gen(width=256):
        "Generates one sample of data"
        for i in range(samples):
            if load_image:
                folder = Path(__file__).parent
                path = os.path.join(folder, image_path)
                with open(path, "rb") as f:
                    img = Image.open(f)
                    inp = img.convert("RGB")
                    inp = np.array(inp)[:width, :width]
            else:
                inp = np.random.rand(width, width, 3)
                inp = (255 * inp).astype("uint8")
            objs = {"input": inp, "label": np.random.rand(1).astype("uint8")}
            yield objs

    ds = tf.data.Dataset.from_generator(
        tf_gen,
        output_types={
            "input": tf.dtypes.as_dtype("uint8"),
            "label": tf.dtypes.as_dtype("uint8"),
        },
        output_shapes={"input": [256, 256, 3], "label": [1]},
    )
    return ds


def dataset_loader(
    samples=1, read_from_fs=False, img_path="/tmp/test.png", pytorch=True
):
    """
    Returns tensorflow or pytorch dataset
    """
    inp = np.random.rand(256, 256, 3)
    inp = (255 * inp).astype("uint8")
    img = Image.fromarray(inp)
    img.save(img_path)

    Dataset = PytorchDataset if pytorch else TensorflowDataset
    ds = Dataset(samples=samples, load_image=read_from_fs, image_path=img_path)
    return ds


def train(net, train_dataloader, criterion, optimizer):
    running_loss = 0.0
    batch_time = 0
    compute_time = 0
    t1 = time.time()
    for i, data in enumerate(train_dataloader, 0):
        batch_time += time.time() - t1
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimizefor batch in ds:
        inputs = inputs.float().to('cuda')

        t1 = time.time()
        outputs = net(inputs)
        compute_time = time.time() - t1

        loss = criterion(outputs, labels.to('cuda').argmax(1))
        loss.backward()
        optimizer.step()

        t1 = time.time()
    return batch_time / len(train_dataloader), compute_time / len(train_dataloader)


def train_hub(samples=100, backend="hub:pytorch", read_from_fs=False, batch_size=64):
    """
    Looping over empty space
    """
    if "hub" in backend:
        ds = get_dataset_from_hub(
            samples=samples,
            read_from_fs=read_from_fs,
            pytorch="pytorch" in backend,
        )
    else:
        ds = dataset_loader(
            samples=samples,
            read_from_fs=read_from_fs,
            pytorch="pytorch" in backend,
        )

    if "pytorch" in backend:
        ds = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=8,
            # collate_fn=ds.collate_fn if "collate_fn" in dir(ds) else None,
        )
    else:
        ds = ds.batch(batch_size)

    model_names = ['resnet18', 'resnet101', 'vgg', 'squeezenet', 'densenet']
    for model_name in model_names:
        net, input_size = initialize_model(
            model_name, num_classes=1, feature_extract=False, use_pretrained=False)
        net = net.to('cuda')
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        t1 = time.time()
        batch, compute = train(net, ds, criterion, optimizer)
        print({"Batch time: ": batch, "Compute time": compute})
        t2 = time.time()

        print({
            "name": f"{backend} loading from {'FS' if read_from_fs else 'RAM'}",
            "model_name": model_name,
            # "samples": len(ds),
            "overall": t2 - t1,
            # "iterations": len(ds),
        })


if __name__ == "__main__":
    n_samples = 256
    params = [
        # {"samples": n_samples, "backend": "pytorch", "read_from_fs": False},
        {"samples": n_samples, "backend": "pytorch", "read_from_fs": True},
        # {"samples": n_samples, "backend": "hub:pytorch", "read_from_fs": False},
        # {"samples": n_samples, "backend": "hub:pytorch", "read_from_fs": True},
        # {"samples": n_samples, "backend": "tensorflow", "read_from_fs": False},
        # {"samples": n_samples, "backend": "tensorflow", "read_from_fs": True},
        # {"samples": n_samples, "backend": "hub:tensorflow", "read_from_fs": False},
        # {"samples": n_samples, "backend": "hub:tensorflow", "read_from_fs": True},
    ]
    logs = [train_hub(**args) for args in params]
    # report(logs)
