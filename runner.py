import argparse
import math

import torch

from cll.algo import ComplementaryLoss, complementary_forward_fn
from cll.data import LoaderWithComplementaryLabels
from cll.models import MLP, LinearModel, LeNet
from contrastive.utils import WarmUpScheduler, LearningRateAdjuster, TwoCropTransform, contrastive_forward_fn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from types import SimpleNamespace

from trainer import Trainer, CLLTrainer


def parse_contrastive_args():
    args = dict(
        temp=0.07,
        epochs=100,
        # optimization
        lr_step_size=10,
        lr_gamma=0.1,
        lr=0.005,
        lr_decay_epochs=[70, 80, 90],
        lr_decay_rate=0.01,
        weight_decay=1e-4,
        momentum=0.9,
        base_lr=0.01,
        size=32,
        cosine=False,
        warmup_epochs=10,
        model='resnet50'

    )
    args = SimpleNamespace(**args)
    if args.cosine:
        eta_min = args.lr * (args.lr_decay_rate ** 3)
        args.target_lr = eta_min + (args.lr - eta_min) * (
                1 + math.cos(math.pi * args.warmup_epochs / args.epochs)) / 2
    else:
        args.target_lr = args.lr

    return args


def set_loader(args, classify=False):
    # Define data transformations
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    normalize = transforms.Normalize(mean=mean, std=std)
    if not classify:

        train_transform = TwoCropTransform(transforms.Compose([
            transforms.RandomResizedCrop(size=args.size, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize,
        ]))

        val_transform = TwoCropTransform(transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    else:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        val_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    # Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=train_transform, download=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, transform=val_transform, download=True)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    return train_loader, test_loader


def parse_cll_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-lr', help='optimizer\'s learning rate', default=5e-5, type=float)
    parser.add_argument('--batch_size', help='batch_size of ordinary labels.', default=256, type=int)
    parser.add_argument('-dataset', help='specify a dataset', default="mnist", type=str,
                        required=False)  # mnist, kmnist, fashion, cifar10
    parser.add_argument('--method', help='method type', choices=['pc', 'nn', 'scarce', 'scarce', 'forward', 'free'],
                        type=str, required=True)
    parser.add_argument('--model', help='model name', default='mlp',
                        choices=['linear', 'mlp', 'resnet', 'densenet', 'lenet', 'convnet'], type=str, required=False)
    parser.add_argument('--epochs', help='number of epochs', type=int, default=200)
    parser.add_argument('-wd', help='weight decay', default=1e-5, type=float)
    parser.add_argument('-seed', help='Random seed', default=0, type=int, required=False)
    parser.add_argument('-gpu', help='used gpu id', default='0', type=str, required=False)
    parser.add_argument('-o', help='optimizer', default='adam', type=str, required=False)
    parser.add_argument('-gen', help='the generation process of complementary labels', default='random',
                        choices=['random', 'scarce_1', 'scarce_2'], type=str, required=False)
    parser.add_argument('-run_times', help='random run times', default=0, type=int, required=False)

    args = parser.parse_args()
    return args


def prepare_dataloaders(dataset_name, batch_size):
    if dataset_name == 'mnist':
        train_dataset = datasets.MNIST(root='./dataset/mnist', train=True, transform=transforms.ToTensor(),
                                       download=True)
        test_dataset = datasets.MNIST(root='./dataset/mnist', train=False, transform=transforms.ToTensor())
    elif dataset_name == 'kmnist':
        train_dataset = datasets.KMNIST(root='./dataset/KMNIST', train=True, transform=transforms.ToTensor(),
                                        download=True)
        test_dataset = datasets.KMNIST(root='./dataset/KMNIST', train=False, transform=transforms.ToTensor())
    elif dataset_name == 'fashion':
        train_dataset = datasets.FashionMNIST(root='./dataset/FashionMnist', train=True,
                                              transform=transforms.ToTensor(), download=True)
        test_dataset = datasets.FashionMNIST(root='./dataset/FashionMnist', train=False,
                                             transform=transforms.ToTensor())
    elif dataset_name == 'cifar10':
        train_transform = transforms.Compose(
            [transforms.ToTensor(),  # transforms.RandomHorizontalFlip(), transforms.RandomCrop(32,4),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        test_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        train_dataset = datasets.CIFAR10(root='./dataset', train=True, transform=train_transform, download=True)
        test_dataset = datasets.CIFAR10(root='./dataset', train=False, transform=test_transform)

    else:
        raise ValueError(f"Dataset {dataset_name} not available")

    # Initialize DatasetWithComplementaryLabels
    data = LoaderWithComplementaryLabels(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size  # Specify batch size
    )
    # Get the loaders and class prior probabilities
    return data.get_loaders()


if __name__ == '__main__':
    args = parse_cll_args()
    ordinary_train_loader, complementary_train_loader, test_loader, ccp, dim = (
        prepare_dataloaders(args.dataset, args.batch_size))
    device = torch.device("cuda:" + args.gpu if torch.cuda.is_available() else "cpu")
    K = ordinary_train_loader.dataset.classes
    if args.model == 'mlp':
        model = MLP(dim, 500, output_dim=K)
    elif args.model == 'linear':
        model = LinearModel(dim, K)
    elif args.model == 'lenet':
        model = LeNet(K)

    meta_method = args.method
    model = model.to(device)
    if args.op == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    trainer = CLLTrainer(
        model=model,
        train_loader=complementary_train_loader,
        val_loader=None,
        optimizer=optimizer,
        loss_fn=ComplementaryLoss(K, ccp, args.method),
        classification=True,
        device=device,
        forward_fn=lambda model, inputs, labels, loss_fn, device: complementary_forward_fn(
            model, inputs, labels, loss_fn, ccp=ccp, method=args.method, device=device
        ),
        save_freq=30

    )

    trainer.fit(args.epochs, train_loader=ordinary_train_loader, test_loader=test_loader, offset=270)
