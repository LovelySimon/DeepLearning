import torch
import torchvision.models
from matplotlib import pyplot as plt
from torch.hub import load_state_dict_from_url
from torchvision import datasets, transforms
from torch import nn
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((448*448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

data_dir = '/home/wh603/桌面/carsclassification/train'
train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
test_dataset = datasets.ImageFolder(root='/home/wh603/桌面/carsclassification/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True)


class BCNN(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(BCNN, self).__init__()
        features = torchvision.models.resnet34(pretrained=pretrained)
        self.conv = nn.Sequential(*list(features.children())[:-2])
        self.fc = nn.Linear(512*512, num_classes)

        if pretrained:
            for parameter in self.conv.parameters():
                parameter.requires_grad = False
            nn.init.kaiming_normal_(self.fc.weight.data)
            nn.init.constant_(self.fc.bias, val=0)

    def forward(self, input):
        features = self.conv(input)
        features = features.view(features.size(0), 512, 14*14)
        features_T = torch.transpose(features, 1, 2)  # 将第二维度和第三维度进行交换
        features = torch.bmm(features, features_T) / (14*14)
        features = features.view(features.size(0),512*512)
        return self.fc(features)

model=BCNN(num_classes=6)
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.fc.parameters(),lr=0.01,momentum=0.9)
losses=[]

def train(epoch):
    running_loss=0.0
    for batch_idx, data in enumerate(train_loader,0):
        inputs, labels=data
        inputs, labels=inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs=model(inputs)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    epoch_loss = running_loss / len(train_loader)
    losses.append(epoch_loss)
    print(f'Epoch {epoch + 1}, Loss: {epoch_loss:.4f}')

def test():
    model.eval()
    with torch.no_grad():
        correct=0
        total=0
        for images,labels in test_loader:
            images , labels= images.to(device),labels.to(device)
            outputs=model(images)
            _,predictions = torch.max(outputs.data,dim=1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
    print('Accuracy : %d %%' % (100 * correct / total))

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
