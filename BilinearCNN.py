import torch
import torchvision.models
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights
from torchvision.models import ResNet34_Weights
from torchvision import datasets, transforms
from torch.optim import SGD
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import ssl
from torch.optim.lr_scheduler import StepLR
ssl._create_default_https_context = ssl._create_unverified_context
writer = SummaryWriter(log_dir="logs")
# --------------------------------------------------------------------------
# 对输入图片进行预处理和归一化
# --------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# ---------------------------------------------------------------------------
# 读入图片设置datasets
# ---------------------------------------------------------------------------
train_data_dir = 'C://Users//Administrator//Desktop//datacars//train'
val_data_dir = 'C://Users//Administrator//Desktop//datacars//val'
test_data_dir = 'C://Users//Administrator//Desktop//tobacco//test'
train_dataset = datasets.ImageFolder(root=train_data_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_data_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=4)
# -----------------------------------------
# 网络设计，使用预训练的resnet34和resnet50
# -----------------------------------------
class BilinearModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(BilinearModel, self).__init__()
        self.resnet34 = torchvision.models.resnet34(weights=ResNet34_Weights.DEFAULT)
        self.resnet50 = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
        # 去除全连接层和分类层 (classification layers)
        self.resnet34 = nn.Sequential(*list(self.resnet34.children())[:-2])
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-2])
        self.resnet50_reduce = torch.nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0)
        # 降维操作
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(128*128, 128)
        self.fc2 = nn.Linear(128, num_classes)
        # self._init_weights()
        # 冻结resnet网络的参数
        for param in self.resnet34.parameters():
            param.requires_grad = False
        for param in self.resnet50.parameters():
            param.requires_grad = False

    # def _init_weights(self):
    #     # 使用 Kaiming 初始化权重，偏置初始化为零
    #     nn.init.kaiming_normal_(self.fc1.weight.data)
    #     nn.init.constant_(self.fc1.bias, 0)

    #     nn.init.kaiming_normal_(self.fc2.weight.data)
    #     nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        # Extract features from both ResNet models
        feature_resnet34 = self.resnet34(x)
        feature_resnet50 = self.resnet50(x)
        feature_resnet50 = self.resnet50_reduce(feature_resnet50)  #[B,512,14,14]
        feature_resnet34 = self.conv1(feature_resnet34)
        feature_resnet34 = self.conv2(feature_resnet34)
        feature_resnet50 = self.conv1(feature_resnet50)
        feature_resnet50 = self.conv2(feature_resnet50)
        feature_resnet34 = feature_resnet34.view(feature_resnet34.size(0), 128, 10*10)
        feature_resnet50 = feature_resnet50.view(feature_resnet50.size(0), 128, 10*10)
        # Transpose feature_resnet50 to align dimensions for batch matrix multiplication
        feature_resnet50_T = feature_resnet50.transpose(1, 2)  # [B, H*W, 512]
        # Perform bilinear pooling
        bilinear_features = torch.bmm(feature_resnet34, feature_resnet50_T) / (10*10)  # [B, 512, 512]
        # Flatten the bilinear features to a 1D vector
        bilinear_features = bilinear_features.view(feature_resnet34.size(0), -1)  # [B, 512*512]
        # Pass through the fully connected layer
        bilinear_features = nn.functional.relu(self.fc1(bilinear_features))
        out = self.fc2(bilinear_features)
        return out


# Initialize model, criterion, and optimizer
model = BilinearModel(num_classes=4)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    [
        {'params': model.fc1.parameters()},
        {'params': model.fc2.parameters()},
    ],
    lr=0.001,
    momentum=0.9
)
scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
best_val_loss = float('inf')
def train(epoch):
    model.train()
    running_loss = 0.0
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
    writer.add_scalar('Training Loss', epoch_loss, epoch)
    print(f"Epoch [{epoch + 1}], Loss: {epoch_loss:.4f}")


def validate(epoch):
    model.eval()  # 设置为评估模式
    val_loss = 0.0
    correct = 0
    total = 0
    global best_val_loss
    with torch.no_grad():  # 禁用梯度计算
        for data in val_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()  # 累加 loss
            _, predicted = torch.max(outputs.data, 1)  # 获取预测类别
            total += labels.size(0)  # 总样本数
            correct += (predicted == labels).sum().item()  # 预测正确的数量
    avg_val_loss = val_loss / len(val_loader)  # 计算平均 loss
    accuracy = 100 * correct / total  # 计算准确率
    writer.add_scalar('Validation Loss', avg_val_loss, epoch)
    writer.add_scalar('Validation Accuracy', accuracy, epoch)
    print(f"Validation Epoch [{epoch + 1}], Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%")
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"Model saved with validation loss: {best_val_loss:.4f}")


def test(num_classes):
    model.load_state_dict(torch.load('best_model.pth'))
    # torch.save(model, 'BilinearCNN.pth')
    total_params = sum(p.numel() for p in model.parameters())
    print(total_params)
    model.eval()  # 设置为评估模式
    correct = 0
    total = 0
    class_correct = [0 for _ in range(num_classes)]
    class_total = [0 for _ in range(num_classes)]

    with torch.no_grad():  # 禁用梯度计算
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # 获取输出
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            # 计算整体准确率
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 计算每个类别的准确率
            for i in range(len(labels)):
                label = labels[i]
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1

    # 计算平均准确率
    overall_accuracy = 100 * correct / total

    # 计算每个类别的准确率
    class_accuracy = [(100 * class_correct[i] / class_total[i]) if class_total[i] > 0 else 0 for i in
                      range(num_classes)]

    # 打印测试集整体准确率
    print(f"Test Accuracy: {overall_accuracy:.2f}%")

    # 打印每个类别的准确率
    for i in range(num_classes):
        print(f"Class {i} Accuracy: {class_accuracy[i]:.2f}%")


if __name__ == '__main__':
    for epoch in range(100):
        train(epoch)
        validate(epoch)
        scheduler.step()
    # test(num_classes=4)
    writer.close()