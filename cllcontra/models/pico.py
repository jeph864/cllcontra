import torch
import torch.nn as nn
import torch.nn.functional as F


class PiCO(nn.Module):
    def __init__(
            self,
            base_encoder,
            num_classes,
            feature_dim,
            architecture,
            queue_size,
            proto_momentum=0.9,
            moco_momentum=0.999,
            pretrained=False,
    ):
        """
        Initializes the PiCO base class.

        Args:
            base_encoder: The backbone encoder model.
            num_classes: Number of classes for classification.
            feature_dim: Dimensionality of the feature embeddings.
            architecture: Name of the architecture for the encoder.
            queue_size: Size of the feature queue for MoCo.
            proto_momentum: Momentum coefficient for prototype updates.
            moco_momentum: Momentum coefficient for key encoder updates.
            pretrained: Whether to use pretrained weights in the encoder.
        """
        super().__init__()
        self.prototypes = None
        self.encoder_q = base_encoder(
            num_class=num_classes, feat_dim=feature_dim, name=architecture, pretrained=pretrained
        )
        self.encoder_k = base_encoder(
            num_class=num_classes, feat_dim=feature_dim, name=architecture, pretrained=pretrained
        )

        self._initialize_momentum_encoder()
        self.queue_size = queue_size
        self.proto_momentum = proto_momentum
        self.moco_momentum = moco_momentum

        # Feature queue for MoCo
        self.register_buffer("queue", F.normalize(torch.randn(queue_size, feature_dim), dim=0))
        self.register_buffer("queue_labels", torch.zeros(queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        # Prototypes for contrastive learning
        self.register_buffer("prototypes", torch.zeros(num_classes, feature_dim))

    def _initialize_momentum_encoder(self):
        """
        Initialize the momentum encoder by copying the weights from the query encoder.
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    @torch.no_grad()
    def update_key_encoder(self):
        """
        Updates the momentum encoder using the MoCo momentum coefficient.
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.moco_momentum + param_q.data * (1. - self.moco_momentum)

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys, labels):
        """
        Updates the feature queue with new keys and labels.
        """
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        assert self.queue_size % batch_size == 0, "Queue size must be divisible by batch size."

        self.queue[ptr:ptr + batch_size, :] = keys
        self.queue_labels[ptr:ptr + batch_size] = labels
        self.queue_ptr[0] = (ptr + batch_size) % self.queue_size

    def forward(self, img_query, img_key=None, partial_labels=None, eval_only=False):
        """
        Forward pass for PiCO.

        Args:
            img_query: Query image batch.
            img_key: Key image batch.
            partial_labels: Partial labels for the query images.
            eval_only: If True, only returns the encoder output without training logic.
        """
        # Query encoder
        output, features_query = self.encoder_q(img_query)
        if eval_only:
            return output

        # Calculate pseudo-labels
        predicted_scores = torch.softmax(output, dim=1) * partial_labels
        _, pseudo_labels_query = torch.max(predicted_scores, dim=1)

        # Compute prototypical logits
        prototypes = self.prototypes.clone().detach()
        logits_proto = torch.mm(features_query, prototypes.t())
        scores_proto = torch.softmax(logits_proto, dim=1)

        # Update prototypes
        for feat, label in zip(features_query, pseudo_labels_query):
            self.prototypes[label] = self.proto_momentum * self.prototypes[label] + (1 - self.proto_momentum) * feat
        self.prototypes = F.normalize(self.prototypes, dim=1).detach()

        # Key encoder logic
        with torch.no_grad():
            self.update_key_encoder()
            _, features_key = self.encoder_k(img_key)

        # Combine features and pseudo-labels
        features = torch.cat([features_query, features_key, self.queue.clone().detach()], dim=0)
        pseudo_labels = torch.cat([pseudo_labels_query, pseudo_labels_query, self.queue_labels.clone().detach()], dim=0)

        # Update the queue
        self.dequeue_and_enqueue(features_key, pseudo_labels_query)

        return output, features, pseudo_labels, scores_proto


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Returns the tensor as is since no distributed data parallel (DDP) is being used.
    """
    return tensor
