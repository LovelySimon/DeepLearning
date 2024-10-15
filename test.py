import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


# 自定义数据集
class ExamDataSet(Dataset):
    def __init__(self):
        train_set_x = np.loadtxt('./Exam/test/x.txt', dtype=np.float32)
        train_set_y = np.loadtxt('./Exam/test/y.txt', dtype=np.float32).reshape(-1, 1)
        self.len = train_set_x.shape[0]
        self.x_data = torch.from_numpy(train_set_x)
        self.y_data = torch.from_numpy(train_set_y)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


# 数据集和数据加载器
dataset = ExamDataSet()
train_loader = DataLoader(dataset=dataset, batch_size=16)


# 定义模型
class EModule(torch.nn.Module):
    def __init__(self):
        super(EModule, self).__init__()
        self.linear1 = torch.nn.Linear(2, 4)
        self.linear2 = torch.nn.Linear(4, 2)
        self.linear3 = torch.nn.Linear(2, 1)
        self.linear1.weight.data.fill_(0.1)  # 将linear1的权重初始化为0.01
        self.linear1.bias.data.fill_(0.0)  # 将linear1的偏置初始化为0.0
        self.linear2.weight.data.fill_(0.1)  # 将linear2的权重初始化为0.01
        self.linear2.bias.data.fill_(0.0)  # 将linear2的偏置初始化为0.0
        self.linear3.weight.data.fill_(0.1)  # 将linear3的权重初始化为0.01
        self.linear3.bias.data.fill_(0.0)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


# 创建模型
model1 = EModule()
model2 = EModule()

# 定义损失函数和优化器
criterion = torch.nn.BCELoss()
learning_rate = 0.1
optimizer = torch.optim.SGD(model2.parameters(), lr=learning_rate)

# 画图的准备工作
plt.ion()  # 开启交互模式
fig, ax = plt.subplots()
x_data = []
loss_data1 = []
loss_data2 = []
line1, = ax.plot(x_data, loss_data1, label='Train with GD', color='b')
line2, = ax.plot(x_data, loss_data2, label='Train with SGD', color='r')
ax.set_xlim(0, 100)  # 设置x轴范围
ax.set_ylim(0.6, 0.8)  # 设置y轴范围
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
ax.legend()
ax.grid()


# 训练模型的函数
def train(epoch):
    for ep in range(epoch):
        for data in train_loader:
            inputs, labels = data
            # 训练第一个模型
            outputs1 = model1(inputs)
            loss1 = criterion(outputs1, labels)
            loss1.backward()
            for param in model1.parameters():
                param.data -= learning_rate * param.grad.data
            model1.zero_grad()

            # 训练第二个模型
            outputs2 = model2(inputs)
            loss2 = criterion(outputs2, labels)
            optimizer.zero_grad()
            loss2.backward()
            optimizer.step()

        # 收集损失数据
        loss_data1.append(loss1.item())
        loss_data2.append(loss2.item())
        x_data.append(ep + 1)

        # 动态更新图形
        line1.set_xdata(x_data)
        line1.set_ydata(loss_data1)
        line2.set_xdata(x_data)
        line2.set_ydata(loss_data2)
        ax.draw_artist(line1)
        ax.draw_artist(line2)
        fig.canvas.flush_events()  # 刷新图形

        print(f'Epoch [{ep + 1}/{epoch}], Loss1: {loss1.item():.4f}, Loss2: {loss2.item():.4f}')

    plt.ioff()  # 关闭交互模式
    plt.show()  # 显示最终图形


# 开始训练
if __name__ == '__main__':
    train(100)
