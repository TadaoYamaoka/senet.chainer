from chainer import link
import chainer.functions as F
import chainer.links as L


class SELayer(link.Chain):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(channel, channel // reduction, nobias=True)
            self.l2 = L.Linear(channel // reduction, channel, nobias=True)

    def forward(self, x):
        b, c, _, _ = x.shape
        y = F.mean(x, axis=(2, 3), keepdims=True)
        y = F.relu(self.l1(y))
        y = F.sigmoid(self.l2(y))
        return x * y.reshape((b, c, 1, 1))
