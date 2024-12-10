import os
from typing import Optional, Tuple, Dict

import numpy as np
import math

import torch


class LearningRateAdjuster:
    def __init__(self, optimizer, base_lr, epochs, lr_decay_rate=0.1, lr_decay_epochs=None, cosine=False):
        """
        Handles learning rate adjustment for both cosine annealing and step decay.
        Args:
            optimizer: Optimizer to adjust.
            base_lr: Initial learning rate.
            epochs: Total number of epochs.
            lr_decay_rate: Decay rate for step-based decay.
            lr_decay_epochs: List of epochs for step decay.
            cosine: If True, use cosine annealing; otherwise, use step decay.
        """
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.epochs = epochs
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_epochs = lr_decay_epochs if lr_decay_epochs else []
        self.cosine = cosine

    def adjust(self, epoch):
        """
        Adjust the learning rate for the given epoch.
        Args:
            epoch: Current epoch number.
        """
        if self.cosine:
            eta_min = self.base_lr * (self.lr_decay_rate ** 3)
            lr = eta_min + (self.base_lr - eta_min) * (
                    1 + math.cos(math.pi * epoch / self.epochs)) / 2
        else:
            steps = np.sum(epoch > np.asarray(self.lr_decay_epochs))
            lr = self.base_lr * (self.lr_decay_rate ** steps)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_lr(self):
        """
        Get the current learning rate.
        Returns:
            float: Current learning rate.
        """
        for param_group in self.optimizer.param_groups:
            return param_group['lr']


class WarmUpScheduler:
    def __init__(self, optimizer, warmup_epochs, base_lr, target_lr):
        """
        Warm-up scheduler for linear learning rate increase.
        Args:
            optimizer: Optimizer to adjust.
            warmup_epochs: Number of warm-up epochs.
            base_lr: Starting learning rate.
            target_lr: Final learning rate after warm-up.
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.target_lr = target_lr
        self.current_epoch = 0
        self.current_batch = 0

    def step(self, epoch, batch_idx, total_batches):
        """
        Adjust learning rate for the current warm-up step.
        Args:
            epoch: Current epoch number.
            batch_idx: Current batch index within the epoch.
            total_batches: Total number of batches in the epoch.
        """
        if epoch < self.warmup_epochs:
            progress = (batch_idx + epoch * total_batches) / (self.warmup_epochs * total_batches)
            lr = self.base_lr + progress * (self.target_lr - self.base_lr)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        self.current_epoch = epoch
        self.current_batch = batch_idx

    def get_lr(self):
        """
        Get the current learning rate.
        """
        for param_group in self.optimizer.param_groups:
            return param_group['lr']


class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.count = None
        self.sum = None
        self.avg = None
        self.val = None
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save(model, optimizer, filename, epoch=0, args=None):
    print('==> Saving...')
    args = args or {}
    state = {
        'args': args,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, filename)
    del state


def contrastive_forward_fn(model, inputs, labels, loss_fn):
    images = torch.cat([inputs[0], inputs[1]], dim=0)  # Combine augmented views

    if torch.cuda.is_available():
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
    features = model(images)

    bsz = labels.shape[0]
    f1, f2 = torch.split(features, [bsz, bsz], dim=0)
    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
    loss = loss_fn(features, labels)
    return features, loss


class AccuracyContext:
    def __init__(self, compute_accuracy=True, accuracy_fn=None):
        """
        Context manager for controlling accuracy computation.
        Args:
            compute_accuracy (bool): Whether to compute accuracy.
        """
        self.compute_accuracy = compute_accuracy
        default_accuracy_fn = lambda output, labels: (torch.argmax(output, dim=1) == labels).sum().item()
        self.accuracy_fn = accuracy_fn or default_accuracy_fn
        self.total_correct = 0
        self.total_samples = 0

    def update(self, outputs, labels):
        """
        Update the total correct predictions and samples.
        Args:
            outputs (Tensor): Model outputs (logits).
            labels (Tensor): Ground truth labels.
        """
        if self.compute_accuracy:
            correct = self.accuracy_fn(outputs, labels)
            self.total_correct += correct
            self.total_samples += labels.size(0)

    def compute(self):
        """
        Compute the accuracy based on accumulated results.
        Returns:
            float: Accuracy as a percentage.
        """
        if self.compute_accuracy and self.total_samples > 0:
            return (self.total_correct / self.total_samples) * 100.0
        return 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


def load_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        checkpoint_path: str,
        device: Optional[torch.device] = None
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, Dict, int]:
    """
    Load model and optimizer from a checkpoint

    Args:
        model: The model architecture (should match the saved one)
        optimizer: The optimizer (should match the saved one)
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model to (if None, will use the same device as the checkpoint)

    Returns:
        tuple: (loaded_model, loaded_optimizer, training_state, epoch)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    # Load checkpoint on CPU to avoid GPU memory issues
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Load model state
    model.load_state_dict(checkpoint['model'])
    if device is not None:
        model = model.to(device)

    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer'])

    # If using GPU, move optimizer states to GPU
    if device is not None and 'cuda' in device.type:
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    # Get training state and epoch
    training_state = checkpoint.get('training_state', {})
    epoch = checkpoint.get('epoch', 0)

    return model, optimizer, training_state, epoch
