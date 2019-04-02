import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision

epochs = 5
learning_rate = 0.001
beta1 = 0.9
beta2 = 0.99
batch_size = 100

transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()
])

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train = torchvision.datasets.CIFAR10(root='../../data',
                                     train=True,
                                     transform=transform,
                                     download=True)

test = torchvision.datasets.CIFAR10(root='../../data',
                                    train=False,
                                    transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(train,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(test,
                                          batch_size=batch_size)


def conv3x3(inp_channels, out_channels, stride=1):
    return nn.Conv2d(inp_channels, out_channels, 3, stride=stride, padding=1)


class ResidualBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(inp_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, num_classes=10):
        super(ResNet, self).__init__()
        self.inp_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.layer1 = self.make_layer(16, block[0])
        self.layer2 = self.make_layer(32, block[1], 2)
        self.layer3 = self.make_layer(64, block[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, out_channels, block, stride=1):
        downsample = None
        if self.inp_channels != out_channels or stride != 1:
            downsample = nn.Sequential(
                conv3x3(self.inp_channels, out_channels, stride),
                nn.BatchNorm2d(out_channels)
            )
        layers = []
        if downsample:
            layers.append(ResidualBlock(self.inp_channels, out_channels, stride, downsample))
        self.inp_channels = out_channels
        for i in range(0, block):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


model = ResNet([1, 1, 1])
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2))

for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch:{:d}/{:d}   Sample:{:d}/{:d}  Loss:{:4f}'.
                  format(epoch+1, epochs, i+1, len(train_loader), loss.item()))

model.eval()
with torch.no_grad():
    total = 0.0
    correct = 0.0
    for (images, labels) in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        prediction = torch.argmax(output, 1)
        total += len(labels)
        correct += (prediction == labels).sum().item()

print("Test accuracy:{:2f}%".format(100*correct/total))
