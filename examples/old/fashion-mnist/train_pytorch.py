from hub import dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x.float()), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(model, train_loader, optimizer):
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        data = batch["data"]
        data = torch.unsqueeze(data, 1)
        labels = batch["labels"]
        labels = labels.type(torch.LongTensor)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()


def test(model, test_loader):
    model.eval()
    print("Evaluating on Test Set")
    test_loss = correct = 0
    with torch.no_grad():
        for batch in test_loader:
            data = batch["data"]
            data = torch.unsqueeze(data, 1)
            labels = batch["labels"]
            labels = labels.type(torch.LongTensor)
            output = model(data)
            test_loss += F.nll_loss(output, labels, reduction="sum").item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    print(
        "Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def main():
    EPOCHS = 3
    BATCH_SIZE = 64
    LEARNING_RATE = 0.01
    MOMENTUM = 0.5
    torch.backends.cudnn.enabled = False
    random_seed = 2
    torch.manual_seed(random_seed)

    # Load data
    ds = dataset.load("mnist/fashion-mnist")

    # Transform into pytorch
    # max_text_len is an optional argument that sets the maximum length of text labels, default is 30
    ds = ds.to_pytorch(max_text_len=15)

    # Splitting back into the original train and test sets, instead of random split
    train_dataset = torch.utils.data.Subset(ds, range(60000))
    test_dataset = torch.utils.data.Subset(ds, range(60000, 70000))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, collate_fn=ds.collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, collate_fn=ds.collate_fn
    )

    model = CNN()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    for epoch in range(EPOCHS):
        print(f"Starting Training Epoch {epoch}")
        train(model, train_loader, optimizer)
        print(f"Training Epoch {epoch} finished\n")
        test(model, test_loader)

    # sanity check to see outputs of model
    for batch in test_loader:
        print("\nNamed Labels:", dataset.get_text(batch["named_labels"]))
        print("\nLabels:", batch["labels"])

        data = batch["data"]
        data = torch.unsqueeze(data, 1)

        output = model(data)
        pred = output.data.max(1)[1]
        print("\nPredictions:", pred)
        break


if __name__ == "__main__":
    main()
