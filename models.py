import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.models import densenet121
from torch.nn import Module


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10, normalize=False):
        super(ResNet, self).__init__()
        self.inchannel = 16
        self.mean = torch.from_numpy(np.array([0.4914, 0.4822, 0.4465]).reshape((1, 3, 1, 1))).cuda()
        self.std = torch.from_numpy(np.array([0.247, 0.243, 0.261]).reshape((1, 3, 1, 1))).cuda()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 16, 3, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 32, 3, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 64, 3, stride=2)
        self.fc = nn.Linear(64, num_classes)
        self.normalize = normalize


    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x, feature=False):
        if self.normalize:
            x = ((x - self.mean) / self.std).type(torch.float)
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # if feature:
        #     return out
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        if feature:
            return out
        out = self.fc(out)
        return out


class ConvNet(nn.Module):
    def __init__(self, nb_classes=43, normalize=False):
        super(ConvNet, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn_1 = nn.BatchNorm2d(num_features=32)
        self.conv_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn_2 = nn.BatchNorm2d(num_features=32)
        self.dropout_1 = nn.Dropout(0.2)
        self.conv_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn_3 = nn.BatchNorm2d(num_features=64)
        self.conv_4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn_4 = nn.BatchNorm2d(num_features=64)
        self.dropout_2 = nn.Dropout(0.2)
        self.conv_5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn_5 = nn.BatchNorm2d(num_features=128)
        self.conv_6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn_6 = nn.BatchNorm2d(num_features=128)
        self.dropout_3 = nn.Dropout(0.2)
        self.fc_1 = nn.Linear(in_features=4 * 4 * 128, out_features=nb_classes)
        self.normalize = normalize
        self.mean = torch.from_numpy(np.array([0.3337, 0.3064, 0.3171]).reshape((1, 3, 1, 1))).cuda()
        self.std = torch.from_numpy(np.array([0.2672, 0.2564, 0.2629]).reshape((1, 3, 1, 1))).cuda()


    def forward(self, x):
        if self.normalize:
            x = ((x - self.mean) / self.std).float()
        layer1 = F.relu(self.conv_1(x))
        layer1 = self.bn_1(layer1)
        layer1 = F.relu(self.conv_2(layer1))
        layer1 = self.bn_2(layer1)

        layer2 = self.dropout_1(F.max_pool2d(layer1, 2, 2))
        layer2 = F.relu(self.conv_3(layer2))
        layer2 = self.bn_3(layer2)
        layer2 = F.relu(self.conv_4(layer2))
        layer2 = self.bn_4(layer2)

        layer3 = self.dropout_2(F.max_pool2d(layer2, 2, 2))
        layer3 = F.relu(self.conv_5(layer3))
        layer3 = self.bn_5(layer3)
        layer3 = F.relu(self.conv_6(layer3))
        layer3 = self.bn_6(layer3)

        layer4 = self.dropout_3(F.max_pool2d(layer3, 2, 2)).reshape(-1, 4 * 4 * 128)
        layer4 = self.fc_1(layer4)
        return layer4


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out, x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):

    class Lambda(nn.Module):
        def __init__(self, func):
            super().__init__()
            self.func = func

        def forward(self, x):
            return self.func(x)

    def __init__(self,normalize=False):
        super(DenseNet, self).__init__()
        self.basic_model = densenet121(pretrained=True, progress=True)
        self.mean = torch.from_numpy(np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))).cuda()
        self.std = torch.from_numpy(np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))).cuda()
        self.Lambda_layer = Lambda(lambda data: ((data - self.mean) / self.std).type(torch.float))
        self.normalize = normalize
        self.nb_classes = 1000

    def forward(self, x):
        if self.normalize:
            x = self.Lambda_layer(x)
        out = self.basic_model(x)
        return out

# source: https://github.com/ChawDoe/LeNet5-MNIST-PyTorch/blob/master/model.py
class Model(Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(256, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x, feature=False):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        if feature:
            return y
        y = self.fc3(y)
        y = self.relu5(y)
        return y


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

    return BCE + KLD.mean()

class VAE(nn.Module):


    def __init__(self, latent_dim):
        super().__init__()

        self.encoder = nn.Sequential(nn.Linear(28 * 28, 256),
                                     nn.ReLU())

        self.mu = nn.Linear(256, latent_dim)
        self.logvar = nn.Linear(256, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 28 * 28))

    def encode(self, x):
        x = x.view(x.size(0), -1)
        x_encoded = self.encoder(x)
        mu, logvar = self.mu(x_encoded), self.logvar(x_encoded)
        return mu, logvar

    def sample_z(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        out = self.decoder(z)
        reshaped_out = torch.sigmoid(out).view(-1, 1, 28, 28)
        return reshaped_out

    def forward(self, x, loss=False):
        mu, logvar = self.encode(x)
        z = self.sample_z(mu, logvar)
        output = self.decode(z)
        if loss:
            return loss_function(output, x, mu, logvar)

        return output


def LeNet5(**kwargs):
    return Model()

def DenseNet121(normalize=False, **kwargs):
    return DenseNet(normalize=normalize)


def convnet_fc(normalize=False,**kwargs):
    model = ConvNet(nb_classes=43, normalize=normalize)
    # model.apply(conv_init)
    return model


def ResNet20(**kwargs):
    return ResNet(ResidualBlock)


def ResNet20_CIFAR10(normalize=False, **kwargs):
    return ResNet(ResidualBlock, num_classes=10, normalize=normalize)

def ResNet20_GTSRB(**kwargs):
    return ResNet(ResidualBlock, num_classes=43)
