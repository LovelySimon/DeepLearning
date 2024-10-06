import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class DiabetesDataSet(Dataset):  # DataSet为抽象类不能直接实例化
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = DiabetesDataSet('diabetes.csv.gz')
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)  # shuffle用来设置是否随机读取batch组


class Model(torch.nn.Module):  # 所有的module都要继承于该类
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def forward(self, x):  # 函数重写
        O1 = self.relu(self.linear1(x))
        O2 = self.relu(self.linear2(O1))
        ans = self.sigmoid(self.linear3(O2))
        return ans


model = Model()
criterion = torch.nn.BCELoss(size_average=True)  # 求损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # 优化器,设置学习率

for epoch in range(100):
    for i, data in enumerate(train_loader, 0):
        x_data, y_data = data
        # forward
        y_hat = model(x_data)
        loss = criterion(y_hat, y_data)
        print(epoch, i, loss.item())
        # backward
        optimizer.zero_grad()  # !因为梯度下降的backward中的grad会被累加，所以在backward之前将grad置为0
        loss.backward()  # 自动反向传播
        # update
        optimizer.step()  # 更新
