import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save


def linear_forward_fn(model, inputs, labels, loss_fn):
    inputs = inputs.to(model.device)
    outputs = model(inputs)
    loss = loss_fn(outputs, labels)
    return outputs, loss


class Trainer:
    def __init__(self, model,
                 train_loader, optimizer,
                 loss_fn, val_loader=None, scheduler=None, device="cuda", early_stopping=None,
                 accuracy_topk=(1,), lr_warmup_scheduler=None, lr_adjuster=None,
                 forward_fn=None, save_dir=None, classification=False, save_freq = None
                 ):
        """
        Initialize the Trainer class.
        Args:
            model: The PyTorch model to train.
            train_loader: DataLoader for the training data.
            val_loader: DataLoader for the validation data.
            optimizer: Optimizer for training.
            loss_fn: Loss function to optimize.
            device: Device to use for computation ('cuda' or 'cpu').
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.lr_warmup_scheduler = lr_warmup_scheduler
        self.lr_adjuster = lr_adjuster
        self.device = device

        self.save_freq = save_freq

        self.early_stopping = early_stopping
        self.accuracy_topk = accuracy_topk

        self.classification = classification

        #### hook
        self.forward_fn = forward_fn or linear_forward_fn

        self.save_dir = save_dir or 'output'
        tmp_save_dir = os.path.join(self.save_dir, 'models')
        os.makedirs(tmp_save_dir, exist_ok=True)

        run_id = len([d for d in os.listdir(tmp_save_dir) if os.path.isdir(os.path.join(tmp_save_dir, d))])
        tmp_save_dir = os.path.join(tmp_save_dir, 'run_{run}'.format(run= run_id))
        os.makedirs(tmp_save_dir,  exist_ok=False)

        os.makedirs(os.path.join(self.save_dir, 'models'), exist_ok=True)
        self.log_dir = os.path.join(self.save_dir, 'logs', 'run_{run}'.format(run= run_id))
        self.save_dir = tmp_save_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)  # TensorBoard writer

    def compute_accuracy(self, output, target):
        with torch.no_grad():
            maxk = max(self.accuracy_topk)
            batch_size = target.size(0)
            # Get top-k predictions
            _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
            pred = pred.t()  # Transpose for comparison
            correct = pred.eq(target.view(1, -1).expand_as(pred))  # Check top-k correctness

            # Compute top-k accuracies
            res = []
            for k in self.accuracy_topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def train_one_epoch(self, epoch):
        """
        Train the model for one epoch.
        Args:
            epoch: Current epoch number.
        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch}")

        for batch_idx, (images, labels) in enumerate(progress_bar):
            labels = labels.to(self.device)
            if self.lr_warmup_scheduler:
                self.lr_warmup_scheduler.step(epoch, batch_idx, progress_bar.total)
            outputs, loss = self.forward_fn(self.model, images, labels, self.loss_fn)
            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Log loss
            total_loss += loss.item()
            if self.classification:
                total_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
            total_samples += labels.size(0)
        avg_loss = total_loss / len(self.train_loader)
        accuracy = total_correct / total_samples
        return avg_loss, accuracy

    def evaluate_one_epoch(self, epoch):
        """
        Evaluate the model for one epoch.
        Args:
            epoch: Current epoch number.
        Returns:
            Average validation loss for the epoch.
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        progress_bar = tqdm(self.val_loader, desc=f"Validation Epoch {epoch}")

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(progress_bar):
                images, labels = images.to(self.device), labels.to(self.device)

                # Forward pass
                outputs, loss = self.forward_fn(self.model, images, labels, self.loss_fn)

                # Log loss
                total_loss += loss.item()
                if self.classification:
                    total_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
                total_samples += labels.size(0)
        avg_loss = total_loss / len(self.val_loader)
        accuracy = total_correct / total_samples
        return avg_loss, accuracy

    def fit(self, num_epochs):
        """
        Train the model for a specified number of epochs.
        Args:
            num_epochs: Total number of epochs to train.
            :param forward_fn:
        """
        val_loss = 0
        val_acc = 0.0
        for epoch in range(1, num_epochs + 1):
            if self.lr_adjuster:
                self.lr_adjuster.adjust(epoch)
            train_loss, train_acc = self.train_one_epoch(epoch)
            if self.val_loader:
                val_loss, val_acc = self.evaluate_one_epoch(epoch)

            # Log to TensorBoard
            self.writer.add_scalars("Loss", {"Train": train_loss, "Validation": val_loss}, epoch)
            self.writer.add_scalars("Accuracy", {"Train": train_acc, "Validation": val_acc}, epoch)

            # Scheduler step (if applicable)
            if self.scheduler:
                self.scheduler.step()

            # Print epoch summary
            print(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f} - "
                f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
            if self.save_freq and epoch % self.save_freq == 0:
                filename = os.path.join(self.save_dir, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
                save(self.model, self.optimizer, filename)

            if self.early_stopping and self.val_loader:
                self.early_stopping(val_loss, self.model)
                if self.early_stopping.early_stop:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break

    def predict(self, dataloader):
        """
        Predict using the trained model.
        Args:
            dataloader: DataLoader for the dataset to predict.
        Returns:
            A list of predictions.
        """
        self.model.eval()
        predictions = []

        with torch.no_grad():
            for images, _ in tqdm(dataloader, desc="Predicting"):
                images = images.to(self.device)
                outputs = self.model(images)
                predictions.append(outputs.cpu().numpy())
        return predictions


class EarlyStopping:
    def __init__(self, patience=5, mode="min", delta=0, checkpoint_path=None):
        """
        Early stopping class to monitor validation loss/accuracy and stop training.
        Args:
            patience: Number of epochs to wait without improvement.
            mode: Metric mode - "min" for loss, "max" for accuracy.
            delta: Minimum change to consider as an improvement.
            checkpoint_path: Path to save the best model.
        """
        assert mode in ["min", "max"], "Mode must be 'min' or 'max'"
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.checkpoint_path = checkpoint_path
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, metric, model):
        """
        Evaluate the metric and decide whether to stop training.
        Args:
            metric: Current epoch's validation loss or accuracy.
            model: Model to save if the metric improves.
        """
        if self.best_score is None:
            self.best_score = metric
            self._save_checkpoint(model)
        else:
            if self._is_improvement(metric):
                self.best_score = metric
                self.counter = 0
                self._save_checkpoint(model)
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True

    def _is_improvement(self, metric):
        """
        Check if the metric has improved based on mode and delta.
        """
        if self.mode == "min":
            return metric < self.best_score - self.delta
        else:  # mode == "max"
            return metric > self.best_score + self.delta

    def _save_checkpoint(self, model):
        """
        Save the model to the checkpoint path if provided.
        """
        if self.checkpoint_path:
            torch.save(model.state_dict(), self.checkpoint_path)
            # print(f"Checkpoint saved at {self.checkpoint_path}")
