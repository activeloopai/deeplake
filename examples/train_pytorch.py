"""Basic example of training pytorch model on hub.Dataset
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import hub
from hub.training.model import Model


def example_to_pytorch():
    ds = hub.Dataset("activeloop/fashion_mnist_train")
    torch_ds = ds.to_pytorch(output_type=list)
    torch_dataloader = torch.utils.data.DataLoader(
        torch_ds,
        batch_size=8,
    )
    return torch_dataloader


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(1, 32, 3, 1)
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(5408, 10)

    def forward(self, x):
        x = self.conv(x.float())
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(trainloader: torch.utils.data.DataLoader, net: nn.Module):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(2):
        print(f"Epoch {epoch}")
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            X, y = data
            X = X.permute(0, 3, 1, 2)
            optimizer.zero_grad()
            outputs = net(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Loss {loss.item()}")
    print("Finished Training")


train_dataloader = example_to_pytorch()
net = Net()
model_cl = Model(net)
train(train_dataloader, net)
model_cl.store("/tmp/")
