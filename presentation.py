import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class IrisDataSet(Dataset):
    def __init__(self):
        train_set_x = np.loadtxt('./Iris/train/x.txt', dtype=np.float32)
        train_set_y = np.loadtxt('./Iris/train/y.txt', dtype=np.float32)
        self.len = train_set_x.shape[0]
        self.x_data = torch.from_numpy(train_set_x)
        train_set_y = train_set_y.reshape(120, 1)
        self.y_data = torch.argmax(torch.from_numpy(train_set_y), dim=1)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = IrisDataSet()
train_loader = DataLoader(dataset=dataset, batch_size=16)


class IrisModule(torch.nn.Module):
    def __init__(self):
        super(IrisModule, self).__init__()
        self.linear1 = torch.nn.Linear(2, 4)
        self.linear2 = torch.nn.Linear(4, 2)
        self.linear3 = torch.nn.Linear(2, 3)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.linear3(x)
        return x


model_sgd = IrisModule()
model_lbfgs = IrisModule()
criterion = torch.nn.CrossEntropyLoss()
optimizer_sgd = torch.optim.SGD(model_sgd.parameters(), lr=0.1)
optimizer_lbfgs = torch.optim.LBFGS(model_lbfgs.parameters())

sgd_losses = []
lbfgs_losses = []


def trainWithSGD(epoch):
    for epoch in range(epoch):
        for data in train_loader:
            inputs, labels = data
            outputs = model_sgd(inputs)
            loss = criterion(outputs, labels)
            sgd_losses.append(loss.item())
            optimizer_sgd.zero_grad()
            loss.backward()
            optimizer_sgd.step()


def trainWithLBFGS(epoch):
    for epoch in range(epoch):
        for data in train_loader:
            inputs, labels = data
            optimizer_lbfgs.zero_grad()

            def closure():
                optimizer_lbfgs.zero_grad()
                outputs = model_lbfgs(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                lbfgs_losses.append(loss.item())
                return loss

            optimizer_lbfgs.step(closure)


if __name__ == '__main__':
    trainWithSGD(10)
    trainWithLBFGS(10)
    plt.figure(figsize=(10, 5))
    plt.plot(sgd_losses, label='SGD Loss', color='black')
    plt.plot(lbfgs_losses, label='LBFGS Loss', color='red')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.show()
