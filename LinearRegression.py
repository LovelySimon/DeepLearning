import torch

x_data = torch.Tensor([[1.0], [2.0], [3.0], [4.0]])
y_data = torch.Tensor([[2.000], [2.500], [2.900], [3.147]])


class LinearModel(torch.nn.Module):  # 所有的module都要继承于该类
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):  # 函数重写
        y_pred = self.linear(x)
        return y_pred


model = LinearModel()  # 可调用对象
criterion = torch.nn.MSELoss(reduction='sum')  # 求损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 优化器,设置学习率

for epoch in range(100):
    y_hat = model(x_data)
    loss = criterion(y_hat, y_data)
    print(epoch, loss)
    optimizer.zero_grad()  # !因为梯度下降的backward中的grad会被累加，所以在backward之前将grad置为0
    loss.backward()  # 自动反向传播
    optimizer.step()  # 更新

print('w=', model.linear.weight.item())
print('b=', model.linear.bias.item())
# 测试集
x_test = torch.Tensor([[5.0]])
y_test = model(x_test)
print('y_hat=', y_test.data)
