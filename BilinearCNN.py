import torch
import torchvision.models
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights
from torchvision.models import ResNet34_Weights
from torchvision import datasets, transforms
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
writer = SummaryWriter(log_dir="logs")
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_data_dir = 'D:/datacars/train'
val_data_dir = 'D:/datacars/val'
test_data_dir = 'D:/datacars/test'
train_dataset = datasets.ImageFolder(root=train_data_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_data_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=4)


class BilinearModel(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(BilinearModel, self).__init__()
        self.resnet34 = torchvision.models.resnet34(weights=ResNet34_Weights.DEFAULT)
        self.resnet50 = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
        # Remove the fully connected layers (classification layers)
        self.resnet34 = nn.Sequential(*list(self.resnet34.children())[:-2])  # Remove the classifier
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-2])  # Remove the classifier
        self.resnet50_reduce = torch.nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0)
        # Define the final classifier
        self.fc = nn.Linear(512 * 512, num_classes)  # 512*512 after bilinear pooling (simplified)
        nn.init.kaiming_normal_(self.fc.weight.data)
        nn.init.constant_(self.fc.bias, val=0)
        # Freeze the layers of ResNet models if needed
        for param in self.resnet34.parameters():
            param.requires_grad = False
        for param in self.resnet50.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Extract features from both ResNet models
        feature_resnet34 = self.resnet34(x)
        feature_resnet50 = self.resnet50(x)
        feature_resnet50 = self.resnet50_reduce(feature_resnet50)
        feature_resnet34 = feature_resnet34.view(feature_resnet34.size(0), 512, 14 * 14)
        feature_resnet50 = feature_resnet50.view(feature_resnet50.size(0), 512, 14 * 14)

        # Transpose feature_resnet50 to align dimensions for batch matrix multiplication
        feature_resnet50_T = feature_resnet50.transpose(1, 2)  # [B, H*W, 512]

        # Perform bilinear pooling
        bilinear_features = torch.bmm(feature_resnet34, feature_resnet50_T) / (14 * 14)  # [B, 512, 512]

        # Flatten the bilinear features to a 1D vector
        bilinear_features = bilinear_features.view(feature_resnet34.size(0), -1)  # [B, 512*512]
        # Pass through the fully connected layer
        out = self.fc(bilinear_features)
        return out


# Initialize model, criterion, and optimizer
model = BilinearModel(num_classes=6)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = SGD(model.fc.parameters(), lr=0.01, momentum=0.9)
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
    model.eval()  # 设置为评估模式
    correct = 0
    total = 0
    class_correct = [0 for _ in range(num_classes)]
    class_total = [0 for _ in range(num_classes)]

    feature_matrix = []  # 存储特征矩阵
    labels_list = []  # 存储对应的标签

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

            # 提取并保存特征矩阵和对应标签
            feature_matrix.append(outputs.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

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

    # 将特征矩阵转为 numpy 数组并保存
    feature_matrix = np.vstack(feature_matrix)
    labels_list = np.concatenate(labels_list)
    print(f"Feature Matrix Shape: {feature_matrix.shape}")

    # 写入 TensorBoard
    writer.add_scalar('Test Accuracy', overall_accuracy)
    for i in range(num_classes):
        writer.add_scalar(f'Class {i} Accuracy', class_accuracy[i])

    # 将特征矩阵保存到 TensorBoard
    for i in range(len(feature_matrix)):
        writer.add_embedding(feature_matrix[i].reshape(1, -1), metadata=[str(labels_list[i])], global_step=i,
                             tag="Test Feature Matrix")


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        validate(epoch)
    test(num_classes=6)
    writer.close()