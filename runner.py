import math

from trainer import Trainer
from contrastive.losses import SupConLoss
from contrastive.utils import WarmUpScheduler, LearningRateAdjuster, TwoCropTransform, contrastive_forward_fn
from contrastive.models import LinearClassifier, SupCEResNet, SupConResNet

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights

from types import SimpleNamespace


def parse():
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


def run_supcon():
    args = parse()
    train_loader, test_loader = set_loader(args)
