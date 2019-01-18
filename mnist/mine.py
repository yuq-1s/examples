from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt

DTYPE = torch.float16

def get_loader(train, args, start_idx, **kwargs):
    from scattered_sampler import ScatteredSampler
    dataset = datasets.MNIST('../data', train=train, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ]))
    dataset = sorted(dataset, key=lambda x: x[1])
    return torch.utils.data.DataLoader(
        dataset,
        sampler=ScatteredSampler(dataset, start_idx=start_idx, order=args.n_hot),
        batch_size=args.batch_size, **kwargs)

def n_hot(ids, C, n=4):
    """
    ids: (list, ndarray) shape:[batch_size]
    out_tensor:FloatTensor shape:[batch_size, depth]
    """
    # if not isinstance(ids, (list, np.ndarray)):
        # raise ValueError("ids must be 1-D list or array")
    ids = torch.LongTensor(ids).view(-1, n)
    out = torch.zeros(ids.shape[0], C, dtype=torch.float32)
    out = out.scatter_(dim=1, index=ids, value=1.)
    # assert all(out.sum(dim=1) == args.n_hot)
    return (out.transpose(0, 1) / out.sum(dim=1)).transpose(0, 1)
    # out_tensor.scatter_(1, ids, 1.0)

# 3*3 convolutino
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                    stride=stride, padding=1, bias=False)


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
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


# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(1, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[0], 2)
        self.layer3 = self.make_layer(block, 64, layers[1], 2)
        self.avg_pool = nn.AvgPool2d(8)
        # self.fc = nn.Linear(64, num_classes)
        self.fc = nn.Linear(192, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = F.log_softmax(out, dim=1)
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(11*11*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        # x = x.view(-1, 4*4*50)
        x = x.view(-1, 11*11*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def my_loss(pred, label):
    return -torch.mean(torch.sum(pred * label, dim=1))

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # plt.imshow((data[1, 0]>0).data.numpy())
        # plt.show()
        data = data.reshape((-1, 1, 28*args.n_hot, 28))
        target = n_hot(target, C=10, n=args.n_hot)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # loss = my_loss(output, target)
        loss = torch.nn.functional.l1_loss(torch.exp(output), target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data),
                len(train_loader.dataset) // args.n_hot,
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.reshape((-1, 1, 28*args.n_hot, 28))
            target = target.view(-1, args.n_hot)
            data, target = data.to(device), target.to(device)
            output = torch.exp(model(data))
            output, target = output.to("cpu"), target.to("cpu")
            test_loss += F.l1_loss(output, n_hot(target, C=10, n=args.n_hot),
                                   reduction='sum').item() # sum up batch loss
            _, pred = output.topk(args.n_hot, dim=1)
            pred, _ = pred.sort(dim=1)
            target, _ = target.sort(dim=1)
            # pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).all(dim=1).sum().item()
            # import ipdb
            # ipdb.set_trace()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset) // args.n_hot,
        100. * correct / (len(test_loader.dataset) // args.n_hot)))

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--n-hot', type=int, default=4, metavar='n',
                        help='n hot label')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    start_idx = [0, 5923, 12665, 18623, 24754, 30596, 36017, 41935, 48200, 54051]
    train_loader = get_loader(
        train=True, args=args,
        start_idx=[0, 5923, 12665, 18623, 24754, 30596, 36017, 41935, 48200,
                   54051],
        **kwargs
    )
    test_loader = get_loader(
        train=False, args=args,
        start_idx=[0, 980, 2115, 3147, 4157, 5139, 6031, 6989, 8017, 8991],
        **kwargs
    )

    # model = Net().to(device)
    model = ResNet(block=ResidualBlock, layers=[2, 2, 3, 3]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")

if __name__ == '__main__':
    main()
