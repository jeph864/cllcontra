import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def assump_free_loss(f, K, labels, ccp):
    """Assumption free loss (based on Thm 1) is equivalent to non_negative_loss if the max operator's threshold is
    negative inf."""
    return non_negative_loss(f=f, K=K, labels=labels, ccp=ccp, beta=np.inf)


def non_negative_loss(f, K, labels, ccp, beta):
    ccp = ccp.float().to(device)
    neglog = -F.log_softmax(f, dim=1)
    loss_vector = torch.zeros(K, requires_grad=True).to(device)
    temp_loss_vector = torch.zeros(K).to(device)
    for k in range(K):
        idx = labels == k
        if torch.sum(idx).item() > 0:
            idxs = idx.byte().view(-1, 1).repeat(1, K).to(torch.bool)
            neglog_k = torch.masked_select(neglog, idxs).view(-1, K)
            temp_loss_vector[k] = -(K - 1) * ccp[k] * torch.mean(neglog_k, dim=0)[
                k]  # average of k-th class loss for k-th comp class samples
            loss_vector = loss_vector + torch.mul(ccp[k], torch.mean(neglog_k,
                                                                     dim=0))  # only k-th in the summation of the second term inside max
    loss_vector = loss_vector + temp_loss_vector
    count = np.bincount(labels.data.cpu()).astype('float')
    while len(count) < K:
        count = np.append(count, 0)  # when largest label is below K, bincount will not take care of them
    loss_vector_with_zeros = torch.cat(
        (loss_vector.view(-1, 1), torch.zeros(K, requires_grad=True).view(-1, 1).to(device) - beta), 1)
    max_loss_vector, _ = torch.max(loss_vector_with_zeros, dim=1)
    final_loss = torch.sum(max_loss_vector)
    return final_loss, torch.mul(torch.from_numpy(count).float().to(device), loss_vector)


class AssumpFreeLoss(nn.Module):
    def __init__(self, K, ccp):
        """
        Assumption-free loss based on Theorem 1.
        Args:
            K: Number of classes.
            ccp: Class prior probabilities (complementary probabilities).
        """
        super().__init__()
        self.K = K
        self.ccp = torch.from_numpy(ccp).float().to(device)

    def forward(self, f, labels):
        return non_negative_loss(f=f, K=self.K, labels=labels, ccp=self.ccp, beta=np.inf)


class NonNegativeLoss(nn.Module):
    def __init__(self, K, ccp, beta=0):
        """
        Non-negative loss.
        Args:
            K: Number of classes.
            ccp: Class prior probabilities (complementary probabilities).
            beta: Threshold for the max operator.
        """
        super().__init__()
        self.K = K
        self.ccp = torch.from_numpy(ccp).float().to(device)
        self.beta = beta

    def forward(self, f, labels):
        K = self.K
        return non_negative_loss(f=f, K=self.K, labels=labels, ccp=self.ccp, beta=self.beta)[0]


class ForwardLoss(nn.Module):
    def __init__(self, K):
        """
        Forward loss for complementary labels.
        Args:
            K: Number of classes.
        """
        super().__init__()
        self.K = K

    def forward(self, f, labels):
        Q = torch.ones(self.K, self.K) * 1 / (self.K - 1)
        Q = Q.to(device)
        for k in range(self.K):
            Q[k, k] = 0
        q = torch.mm(F.softmax(f, 1), Q)
        return F.nll_loss(q.log(), labels.long())


class PCLoss(nn.Module):
    def __init__(self, K):
        """
        Pairwise comparison (PC) loss.
        Args:
            K: Number of classes.
        """
        super().__init__()
        self.K = K
        print("Number of classes: ", K)

    def forward(self, f, labels):
        K = self.K
        sigmoid = nn.Sigmoid()
        fbar = f.gather(1, labels.long().view(-1, 1)).repeat(1, K)
        loss_matrix = sigmoid(-1. * (f - fbar))  # multiply -1 for "complementary"
        M1, M2 = K * (K - 1) / 2, K - 1
        pc_loss = torch.sum(loss_matrix) * (K - 1) / len(labels) - M1 + M2
        return pc_loss, None


class ComplementaryLoss(nn.Module):
    def __init__(self, K, ccp, meta_method, device=None):
        """
        Dynamically choose a loss function.
        Args:
            K: Number of classes.
            ccp: Class prior probabilities (complementary probabilities).
            meta_method: Loss method ('free', 'nn', 'forward', 'pc').
        """
        super().__init__()
        self.K = K
        self.ccp = ccp
        self.meta_method = meta_method
        print("Method: ", self.meta_method)
        self.loss_fn = self._initialize_loss_fn()
        self.device = device

    def _initialize_loss_fn(self):
        if self.meta_method == 'free':
            return AssumpFreeLoss(K=self.K, ccp=self.ccp)
        elif self.meta_method == 'nn':
            return NonNegativeLoss(K=self.K, ccp=self.ccp, beta=0)
        elif self.meta_method == 'forward':
            return ForwardLoss(K=self.K)
        elif self.meta_method == 'pc':
            return PCLoss(K=self.K)
        elif self.meta_method == 'ga':
            return AssumpFreeLoss(K=self.K, ccp=self.ccp)
        elif self.meta_method == "scarce":
            return SCARCE(self.device)
        else:
            raise ValueError(f"Unsupported meta_method: {self.meta_method}")

    def forward(self, f, labels):
        return self.loss_fn(f, labels)


def logistic_loss(pred):
    negative_logistic = nn.LogSigmoid()
    logistic = -1. * negative_logistic(pred)
    return logistic


def scarce_loss(outputs, labels, device):
    comple_label_mat = torch.zeros_like(outputs)
    comple_label_mat[torch.arange(comple_label_mat.shape[0]), labels.long()] = 1
    comple_label_mat = comple_label_mat.to(device)
    pos_loss = logistic_loss(outputs)
    neg_loss = logistic_loss(-outputs)
    neg_data_mat = comple_label_mat.float()
    unlabel_data_mat = torch.ones_like(neg_data_mat)
    # calculate negative label loss of negative data
    neg_loss_neg_data_mat = neg_loss * neg_data_mat
    tmp1 = neg_data_mat.sum(dim=0)
    tmp1[tmp1 == 0.] = 1.
    neg_loss_neg_data_vec = neg_loss_neg_data_mat.sum(dim=0) / tmp1
    # calculate positive label loss of unlabeled data
    pos_loss_unlabel_data_mat = pos_loss * unlabel_data_mat
    tmp2 = unlabel_data_mat.sum(dim=0)
    tmp2[tmp2 == 0.] = 1.
    pos_loss_unlabel_data_vec = pos_loss_unlabel_data_mat.sum(dim=0) / tmp2
    # calculate positive label loss of negative data
    pos_loss_neg_data_mat = pos_loss * neg_data_mat
    pos_loss_neg_data_vec = pos_loss_neg_data_mat.sum(dim=0) / tmp1
    # calculate final loss
    prior_vec = 1. / outputs.shape[1] * torch.ones(outputs.shape[1])
    prior_vec = prior_vec.to(device)
    ccp = 1. - prior_vec
    loss1 = (ccp * neg_loss_neg_data_vec).sum()
    unmax_loss_vec = pos_loss_unlabel_data_vec - ccp * pos_loss_neg_data_vec
    max_loss_vec = torch.abs(unmax_loss_vec)
    loss2 = max_loss_vec.sum()
    loss = loss1 + loss2
    return loss


class SCARCE(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.device = device

    def forward(self, outputs, labels):
        return scarce_loss(outputs, labels, self.device), None


def complementary_forward_fn(model, inputs, labels, loss_fn, ccp, method="free", device="cuda"):
    """
    Compute the loss for complementary label learning.
    Args:
        model (nn.Module): The PyTorch model to train.
        inputs (Tensor): Input data batch.
        labels (Tensor): Complementary labels batch.
        loss_fn (Callable): Loss function for complementary learning.
        ccp (Tensor): Class prior probabilities (complementary probabilities).
        method (str): Complementary method ('free', 'nn', 'ga', etc.).
        device (str): Device to use ('cuda' or 'cpu').
    Returns:
        Tuple: (outputs, loss).
    """
    inputs, labels = inputs.to(device), labels.to(device)

    # Forward pass
    outputs = model(inputs)

    # Compute loss
    if method == "ga":
        loss, loss_vector = loss_fn(outputs, labels)
        if torch.min(loss_vector).item() < 0:
            loss_vector_with_zeros = torch.cat(
                (loss_vector.view(-1, 1), torch.zeros(len(ccp), requires_grad=True).to(device).view(-1, 1)), 1
            )
            min_loss_vector, _ = torch.min(loss_vector_with_zeros, dim=1)
            loss = torch.sum(min_loss_vector)
    else:
        loss, _ = loss_fn(outputs, labels)

    return outputs, loss


# Accuracy Forward Function
def ordinary_forward_fn(model, inputs, labels):
    """
    Forward function for accuracy checking.
    """
    outputs = model(inputs)
    accuracy = (outputs.argmax(dim=1) == labels).float().mean().item()
    return accuracy
