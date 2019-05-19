import chainer
import chainer.links as L
from chainer import optimizers
from chainer import training
from chainer.training import extensions

from senet.baseline import resnet20
from senet.se_resnet import se_resnet20


def main():
    train, test = chainer.datasets.get_cifar10()

    train_iter = chainer.iterators.SerialIterator(train, args.batch_size)
    test_iter = chainer.iterators.SerialIterator(test, args.batch_size, repeat=False, shuffle=False)

    if args.baseline:
        model = L.Classifier(resnet20())
    else:
        model = L.Classifier(se_resnet20(num_classes=10, reduction=args.reduction))

    optimizer = optimizers.MomentumSGD(lr=1e-1)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(1e-4))

    updater = training.updaters.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epochs, 'epoch'), out=args.out)

    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    trainer.extend(extensions.dump_graph('main/loss'))

    trainer.extend(extensions.LogReport())

    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    trainer.extend(extensions.ProgressBar())

    trainer.run()


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--reduction", type=int, default=16)
    p.add_argument("--baseline", action="store_true")
    p.add_argument('--gpu', '-g', type=int, default=0)
    p.add_argument('--out', '-o', default='result')
    args = p.parse_args()
    main()
