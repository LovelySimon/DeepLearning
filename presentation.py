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


class BFGSOptimizer:
    def __init__(self, params, lr=1e-3, max_iter=100):
        self.params = list(params)
        self.lr = lr
        self.max_iter = max_iter
        self.hessian_inv = None
        self.prev_params = None
        self.prev_grad = None

    def step(self, closure):
        # 计算当前的损失和梯度
        loss = closure()
        current_grad = torch.cat([p.grad.view(-1) for p in self.params])

        if self.hessian_inv is None:
            # 初始化 Hessian 的逆矩阵为单位矩阵
            self.hessian_inv = torch.eye(current_grad.size(0))

        if self.prev_grad is None:
            self.prev_grad = current_grad.clone()
            self.prev_params = torch.cat([p.data.view(-1) for p in self.params])
            return loss

        # 计算参数变化和梯度变化
        param_diff = torch.cat([p.data.view(-1) for p in self.params]) - self.prev_params
        grad_diff = current_grad - self.prev_grad

        # 更新 Hessian 的逆矩阵
        y = grad_diff.view(-1, 1)
        s = param_diff.view(-1, 1)
        ys = torch.matmul(y.t(), s)
        ss = torch.matmul(s.t(), s)

        if ys.item() > 1e-10:  # 避免数值不稳定
            self.hessian_inv += (1 / ys.item()) * (torch.matmul(s, s.t()) / ss.item()) - (
                        torch.matmul(self.hessian_inv @ y, y.t() @ self.hessian_inv) / ys.item())

        # 计算搜索方向
        search_direction = -self.hessian_inv @ current_grad.view(-1, 1)

        start = 0
        for p in self.params:
            numel = p.numel()
            p.data += self.lr * search_direction[start:start + numel].view(p.size())
            start += numel

        # 存储当前状态
        self.prev_params = torch.cat([p.data.view(-1) for p in self.params])
        self.prev_grad = current_grad.clone()

        return loss


model_sgd = IrisModule()
model_lbfgs = IrisModule()
model_bfgs = IrisModule()
criterion = torch.nn.CrossEntropyLoss()
optimizer_sgd = torch.optim.SGD(model_sgd.parameters(), lr=0.1)
optimizer_lbfgs = torch.optim.LBFGS(model_lbfgs.parameters())
optimizer_bfgs = BFGSOptimizer(model_bfgs.parameters(), lr=1.0)
sgd_losses = []
lbfgs_losses = []
bfgs_losses = []


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


def trainwithBFGS(epoch):
    for epoch in range(epoch):
        for data in train_loader:
            inputs, labels = data

            def closure():
                outputs = model_bfgs(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                bfgs_losses.append(loss.item())
                return loss

            optimizer_bfgs.step(closure)


if __name__ == '__main__':
    trainWithSGD(10)
    trainWithLBFGS(10)
    trainwithBFGS(10)
    plt.figure(figsize=(10, 5))
    plt.plot(sgd_losses, label='SGD Loss', color='black')
    plt.plot(lbfgs_losses, label='LBFGS Loss', color='red')
    plt.plot(bfgs_losses, label='BFGS Loss', color='blue')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.show()
