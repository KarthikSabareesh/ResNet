# %%
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
from custom_layers import Conv2d, BatchNorm2d, MaxPool2d, AdaptiveAvgPool2d, ReLU


class BasicBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = Conv2d(in_channels, out_channels,
                            kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = BatchNorm2d(out_channels)

        self.conv2 = Conv2d(out_channels, out_channels,
                            kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = BatchNorm2d(out_channels)

        self.conv3 = Conv2d(out_channels, out_channels *
                            self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = BatchNorm2d(out_channels*self.expansion)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))

        x = self.relu(self.batch_norm2(self.conv2(x)))

        x = self.conv3(x)
        x = self.batch_norm3(x)

        # downsample if needed
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        # add identity
        x += identity
        x = self.relu(x)

        return x

# to use for Resnt-18 or simpler versions


class Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()

        self.conv1 = Conv2d(in_channels, out_channels,
                            kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = BatchNorm2d(out_channels)
        self.conv2 = Conv2d(out_channels, out_channels,
                            kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = ReLU()

    def forward(self, x):
        identity = x.clone()

        x = self.relu(self.batch_norm2(self.conv1(x)))
        x = self.batch_norm2(self.conv2(x))

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        print(x.shape)
        print(identity.shape)
        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, ResBlock, layer_list, num_classes, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = Conv2d(num_channels, 64, kernel_size=7,
                            stride=2, padding=3, bias=False)
        self.batch_norm1 = BatchNorm2d(64)
        self.relu = ReLU()
        self.max_pool = MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
        self.layer2 = self._make_layer(
            ResBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(
            ResBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(
            ResBlock, layer_list[3], planes=512, stride=2)

        self.avgpool = AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*ResBlock.expansion, num_classes)

    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []

        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                Conv2d(self.in_channels, planes*ResBlock.expansion,
                       kernel_size=1, stride=stride),
                BatchNorm2d(planes*ResBlock.expansion)
            )

        layers.append(ResBlock(self.in_channels, planes,
                      i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion

        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))

        return nn.Sequential(*layers)


def ResNet50(num_classes, channels=3):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, channels)

# %%


transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)

trainloader = torch.utils.data.DataLoader(
    train, batch_size=128, shuffle=True, num_workers=2)

test = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)

testloader = torch.utils.data.DataLoader(
    test, batch_size=128, shuffle=False, num_workers=2)

classes = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# %%
model = ResNet50(10).to('cuda')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01,
                      momentum=0.9, weight_decay=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.1, patience=5)

EPOCHS = 50
for epoch in range(EPOCHS):
    losses = []
    running_loss = 0
    for i, inp in enumerate(trainloader):
        inputs, labels = inp
        inputs, labels = inputs.to('cuda'), labels.to('cuda')
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 100 == 0 and i > 0:
            print(f'Loss [{epoch+1}, {i}](epoch, minibatch): ',
                  running_loss / 100)
            running_loss = 0.0

    avg_loss = sum(losses)/len(losses)
    scheduler.step(avg_loss)

print('Training Complete')

# %%
correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to('cuda'), labels.to('cuda')
        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy on 10,000 test images: ', 100*(correct/total), '%')
