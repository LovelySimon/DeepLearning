import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class ExamDataSet(Dataset):
    def __init__(self):
        train_set_x = np.loadtxt('./Exam/train/x.txt', dtype=np.float32)
        train_set_y = np.loadtxt('./Exam/train/y.txt', dtype=np.float32)
        self.len = train_set_x.shape[0]
        self.x_data = torch.from_numpy(train_set_x)
        train_set_y = train_set_y.reshape(64, 1)
        self.y_data = torch.from_numpy(train_set_y)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = ExamDataSet()
train_loader = DataLoader(dataset=dataset, batch_size=16)


class EModule(torch.nn.Module):
    def __init__(self):
        super(EModule, self).__init__()
        self.linear1 = torch.nn.Linear(2, 4)
        self.linear2 = torch.nn.Linear(4, 2)
        self.linear3 = torch.nn.Linear(2, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        ans = self.sigmoid(self.linear3(x))
        return ans


model = EModule()
model2 = EModule()
criterion = torch.nn.BCELoss(reduction='mean')
learning_rate = 0.01
optimizer = torch.optim.SGD(model2.parameters(), lr=0.01)  # 优化器,设置学习率
gd_losses = []
sgd_losses = []


def trainWithGD(epoch):
    for epoch in range(epoch):
        for data in train_loader:
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            gd_losses.append(loss.item())
            model.zero_grad()
            loss.backward()
            with torch.no_grad():
                for parameters in model.parameters():
                    parameters -= learning_rate * parameters.grad


def trainWithSGD(epoch):
    for epoch in range(epoch):
        for data in train_loader:
            inputs, labels = data
            outputs = model2(inputs)
            loss = criterion(outputs, labels)
            sgd_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    trainWithGD(100)
    trainWithSGD(100)
    plt.figure(figsize=(10, 5))
    plt.plot(gd_losses, label='Gradient Descent Loss', color='red')
    plt.plot(sgd_losses, label='SGD Loss', color='black')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.show()
