import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

train_df = pd.read_csv('./titanic/train.csv')
test_df = pd.read_csv('./titanic/test.csv')

train_df = train_df.drop(['Ticket', 'Cabin', 'Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin', 'Name'], axis=1)

age_mean = (train_df['Age'].sum() + test_df['Age'].sum()) / (train_df['Age'].count() + test_df['Age'].count())

# 直接对 DataFrame 进行操作
train_df['Sex'] = train_df['Sex'].map({'female': 1, 'male': 0}).astype(int)
train_df['Age'].fillna(age_mean, inplace=True)

freq_port = train_df.Embarked.dropna().mode()[0]
train_df['Embarked'].fillna(freq_port, inplace=True)

test_df['Sex'] = test_df['Sex'].map({'female': 1, 'male': 0}).astype(int)

test_df['Age'].fillna(age_mean, inplace=True)
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

train_df['Embarked'] = train_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
test_df['Embarked'] = test_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)


class TitanicDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.features = dataframe.drop('Survived', axis=1).values  # 特征
        self.labels = dataframe['Survived'].values  # 标签

    def __getitem__(self, index):
        x = torch.tensor(self.features[index], dtype=torch.float32)  # 特征转为 tensor
        y = torch.tensor(self.labels[index], dtype=torch.float32)  # 标签转为 tensor
        return x, y

    def __len__(self):
        return len(self.dataframe)


train_data = TitanicDataset(train_df)
train_loader = DataLoader(train_data, batch_size=32, num_workers=2, shuffle=True)


class Model(torch.nn.Module):  # 所有的module都要继承于该类
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(7, 10)
        self.linear2 = torch.nn.Linear(10, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()

    def forward(self, x):  # 函数重写
        O1 = self.relu(self.linear1(x))
        O2 = self.relu(self.linear2(O1))
        ans = self.sigmoid(self.linear3(O2))
        return ans


model = Model()
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    for i, data in enumerate(train_loader, 0):
        x_data, y_data = data
        y_data = y_data.unsqueeze(1)
        # forward
        y_hat = model(x_data)
        loss = criterion(y_hat, y_data)
        print(epoch, i, loss.item())
        # backward
        optimizer.zero_grad()  # !因为梯度下降的backward中的grad会被累加，所以在backward之前将grad置为0
        loss.backward()  # 自动反向传播
        # update
        optimizer.step()  # 更新


class TitanicTest(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.features = dataframe.drop('PassengerId', axis=1).values  # 只包含特征

    def __getitem__(self, index):
        x = torch.tensor(self.features[index], dtype=torch.float32)  # 特征转为 tensor
        return x

    def __len__(self):
        return len(self.dataframe)


test_data = TitanicTest(test_df)
test_dataloader = DataLoader(test_data, batch_size=32, num_workers=2, shuffle=False)

model.eval()  # 切换为评估模式
predictions = []

with torch.no_grad():  # 关闭梯度计算，加快预测速度
    for i, x_data in enumerate(test_dataloader, 0):
        y_hat = model(x_data)
        y_hat = y_hat.squeeze(1)  # 去掉多余的维度
        predicted = (y_hat > 0.5).int()  # 将概率转换为 0 和 1
        predictions.extend(predicted.tolist())  # 将预测结果加入列表

submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],  # 从 test_df 中取出 PassengerId
    'Survived': predictions  # 模型预测的结果
})

# 保存为 CSV 文件
submission.to_csv('submission.csv', index=False)
