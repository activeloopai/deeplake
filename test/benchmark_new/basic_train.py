import time
import io
import torch
import hub
from hub.schema import Tensor
from hub.store.store import get_fs_and_path
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
    model_ft = None
    input_size = 0

    if model_name == "resnet18":
        """Resnet18"""
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == "resnet101":
        """Resnet18"""
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    elif model_name == "alexnet":
        """Alexnet"""
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """VGG11_bn"""
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """Squeezenet"""
        model_ft = models.squeezenet1_1(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(
            512, num_classes, kernel_size=(1, 1), stride=(1, 1)
        )
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """Densenet"""
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
        fs=None,
    ):
        "Initialization"
        self.samples = samples
        self.width = width
        self.load_image = load_image
        self.image_path = image_path
        self.fs = fs

    def __len__(self):
        "Denotes the total number of samples"
        return self.samples

    def __getitem__(self, index):
        "Generates one sample of data"
        if self.load_image:
            if self.image_path.startswith("s3") and not self.fs:
                return {}

            with self.fs.open(self.image_path, "rb") as f:
                img = Image.open(f)
                inp = img.convert("RGB")
                inp = np.array(inp)[: self.width, : self.width]
                inp = np.transpose(inp, (2, 0, 1))
        else:
            inp = np.random.rand(3, self.width, self.width)
            inp = (255 * inp).astype("uint8")
        objs = (torch.tensor(inp), torch.tensor(np.random.rand(1).astype("uint8")))
        return objs


class MyDataset(torch.utils.data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, hub_ds, transform=None):
        "Initialization"
        self.ds = hub_ds
        self._transform = transform

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.ds)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        ID = self.ds[index]
        ID = self._transform(ID) if self._transform else ID
        # Load data and get label
        X = ID["img"]
        y = ID["label"]
        return X, y


def get_dataset_from_hub(samples=1, read_from_fs=False, pytorch=False):
    """
    Build dataset and transform to pytorch or tensorflow
    """
    my_schema = {"img": Tensor(shape=(3, 256, 256)), "label": "uint8"}
    if not read_from_fs:
        ds = hub.Dataset(
            "kristina/benchmarking",
            shape=(samples,),
            schema=my_schema,
            cache=False,
        )
    else:
        ds = hub.Dataset(
            "s3://snark-test/benchmarking",
            shape=(samples,),
            schema=my_schema,
            cache=False,
        )
    for i in range(samples):
        ds["img", i] = np.random.rand(3, 256, 256)
        ds["label", i] = 0
    ds_hub = ds.to_pytorch() if pytorch else ds.to_tensorflow()
    ds = MyDataset(ds_hub)
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
    buff = io.BytesIO()
    img.save(buff, "JPEG")
    buff.seek(0)
    fs, path = get_fs_and_path(img_path)
    with fs.open(img_path, "wb") as f:
        f.write(buff.read())

    Dataset = PytorchDataset if pytorch else TensorflowDataset
    ds = Dataset(samples=samples, load_image=read_from_fs, image_path=img_path, fs=fs)
    return ds


def train(net, train_dataloader, criterion, optimizer):
    batch_time = 0
    compute_time = 0
    t2 = time.time()
    for i, data in enumerate(train_dataloader, 0):
        batch_time += time.time() - t2
        inputs, labels = data

        t1 = time.time()
        optimizer.zero_grad()
        inputs = inputs.float().to("cuda")
        outputs = net(inputs)

        if len(labels.shape) > 1:
            loss = criterion(outputs, labels.to("cuda").argmax(1))
        else:
            loss = criterion(outputs, labels.to("cuda").long())
        loss.backward()
        optimizer.step()
        compute_time += time.time() - t1
        t2 = time.time()
    return batch_time / len(train_dataloader), compute_time / len(train_dataloader)


def train_hub(samples=100, backend="hub:pytorch", read_from_fs=False):
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
            img_path="s3://snark-test/benchmarks/test_img.jpeg",
            pytorch="pytorch" in backend,
        )

    model_names = ["resnet18", "resnet101", "vgg", "squeezenet", "densenet"]
    for model_name in model_names:
        if model_name in ("resnet18", "squeezenet"):
            batch_size = 256
        else:
            batch_size = 64
        if "pytorch" in backend:
            ds_loader = torch.utils.data.DataLoader(
                ds,
                batch_size=batch_size,
                num_workers=8,
            )
        else:
            ds_loader = ds.batch(batch_size)
        net, input_size = initialize_model(
            model_name, num_classes=1, feature_extract=False, use_pretrained=False
        )
        net = net.to("cuda")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
        t1 = time.time()
        batch, compute = train(net, ds_loader, criterion, optimizer)
        print({"Batch time: ": batch, "Compute time": compute})
        t2 = time.time()

        print(
            {
                "name": f"{backend} loading from {'FS' if read_from_fs else 'RAM'}",
                "model_name": model_name,
                "overall": t2 - t1,
            }
        )


if __name__ == "__main__":
    n_samples = 256
    params = [
        # {"samples": n_samples, "backend": "pytorch", "read_from_fs": False},
        {"samples": n_samples, "backend": "pytorch", "read_from_fs": True},
        {"samples": n_samples, "backend": "hub:pytorch", "read_from_fs": False},
        # {"samples": n_samples, "backend": "hub:pytorch", "read_from_fs": True},
    ]
    logs = [train_hub(**args) for args in params]
