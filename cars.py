import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 将图片转化为Tensor形式
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# 设置随机种子
data_dir = '/home/wh603/桌面/carsclassification/train'
train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
test_dataset = datasets.ImageFolder(root='/home/wh603/桌面/carsclassification/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=12)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=12)


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x + y)


class LittleResnet(torch.nn.Module):
    def __init__(self):
        super(LittleResnet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=5)
        self.conv3 = torch.nn.Conv2d(32, 16, kernel_size=5)
        self.mp = torch.nn.MaxPool2d(2)
        self.residual1 = ResidualBlock(16)
        self.residual2 = ResidualBlock(32)
        self.linear1 = torch.nn.Linear(11664, 5832)
        self.linear2 = torch.nn.Linear(5832, 2916)
        self.linear3 = torch.nn.Linear(2916,1468)
        self.linear4 = torch.nn.Linear(1468,734)
        self.linear5 = torch.nn.Linear(734,317)
        self.linear6 = torch.nn.Linear(317,128)
        self.linear7 = torch.nn.Linear(128,6)

    def forward(self, x):
        in_size = x.size(0)
        x = self.mp(F.relu(self.conv1(x)))
        x = self.residual1(x)
        x = F.relu(self.conv2(x))
        x = self.residual2(x)
        x = self.mp(F.relu(self.conv3(x)))
        x = x.view(in_size, -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = F.relu(self.linear4(x))
        x = F.relu(self.linear5(x))
        x = F.relu(self.linear6(x))
        return self.linear7(x)


model = LittleResnet()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.5)
losses = torch.tensor([]).to(device)


def train(epoch):
    running_loss = 0.0
    global losses
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(train_loader)
    losses = torch.cat((losses, torch.tensor([epoch_loss]).to(device)))  # 将新计算的损失添加到 Tensor


def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print('Accuracy : %d %% [%d/%d]' % (100 * correct / total, correct, total))


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()

    plt.plot(losses.cpu().numpy(), label='Training Loss')  # 需要将损失从 GPU 转回到 CPU
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.show()
