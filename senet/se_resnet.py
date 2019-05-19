import chainer
from chainer import link
import chainer.functions as F
import chainer.links as L
from chainer import Sequential
from senet.se_module import SELayer


class CifarSEBasicBlock(link.Chain):
    def __init__(self, inplanes, planes, stride=1, reduction=16):
        super(CifarSEBasicBlock, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(inplanes, planes, ksize=3, stride=stride, pad=1, nobias=True)
            self.bn1 = L.BatchNormalization(planes)
            self.conv2 = L.Convolution2D(planes, planes, ksize=3, stride=1, pad=1, nobias=True)
            self.bn2 = L.BatchNormalization(planes)
            self.se = SELayer(planes, reduction)
            if inplanes != planes:
                self.downsample = Sequential(
                    L.Convolution2D(inplanes, planes, ksize=1, stride=stride, nobias=True),
                    L.BatchNormalization(planes))
            else:
                self.downsample = lambda x: x

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        out += residual
        out = F.relu(out)

        return out


class CifarSEResNet(link.Chain):
    def __init__(self, block, n_size, num_classes=10, reduction=16):
        super(CifarSEResNet, self).__init__()
        self.inplane = 16
        with self.init_scope():
            self.conv1 = L.Convolution2D(3, self.inplane, ksize=3, stride=1, pad=1, nobias=True)
            self.bn1 = L.BatchNormalization(self.inplane)
            self.layer1 = self._make_layer(block, 16, blocks=n_size, stride=1, reduction=reduction)
            self.layer2 = self._make_layer(block, 32, blocks=n_size, stride=2, reduction=reduction)
            self.layer3 = self._make_layer(block, 64, blocks=n_size, stride=2, reduction=reduction)
            self.fc = L.Linear(64, num_classes)

    def _make_layer(self, block, planes, blocks, stride, reduction):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplane, planes, stride, reduction))
            self.inplane = planes

        return Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = F.mean(x, axis=(2, 3), keepdims=True)
        x = self.fc(x)

        return x


def se_resnet20(**kwargs):
    """Constructs a ResNet-18 model.

    """
    model = CifarSEResNet(CifarSEBasicBlock, 3, **kwargs)
    return model
