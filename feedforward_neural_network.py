import torch
from torchvision import transforms,datasets
import torch.nn as nn

epochs = 5
device = "cuda" if torch.cuda.is_available() else "cpu"
learning_rate = 0.001
batch_size = 100
input_size = 784
hidden_size = 500
num_classes = 10

torch.manual_seed(0)
torch.cuda.manual_seed(0)


train = datasets.MNIST(root='../../data',
                       train=True,
                       transform=transforms.ToTensor(),
                       download=True)

test = datasets.MNIST(root='../../data',
                      train=False,
                      transform=transforms.ToTensor(),
                      download=True)

train_loader = torch.utils.data.DataLoader(train,
                                           shuffle=True,
                                           batch_size=batch_size)

test_loader = torch.utils.data.DataLoader(test,
                                          shuffle=False,
                                          batch_size=batch_size)


class Model(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


Model = Model(input_size, hidden_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(Model.parameters(), lr=learning_rate, betas=[0.9, 0.99])

for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        labels = labels.to(device)
        images = images.view(-1, 28*28).to(device)

        outputs = Model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print("Epoch:{}/{}, images:{:d}/{}, loss:{:.3f}.".
                  format(epoch+1, epochs, i+1, len(train_loader), loss.item()))

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.view(-1, 28*28).to(device)
        labels = labels.to(device)

        outputs = Model(images)
        _, prediction = torch.max(outputs, 1)
        total += len(labels)
        correct += (prediction == labels).sum().item()

    print("Test accuracy:{:.4f}".format(correct/total))

torch.save(Model.state_dict(), 'model.ckpt')


