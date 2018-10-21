from torch import nn
from modules.lovasz_losses import lovasz_hinge


class LovaszLoss(nn.Module):
    def __init__(self):
        super(LovaszLoss, self).__init__()

    def forward(self, input, target):
        return lovasz_hinge(input, target, per_image=True)


class LossAggregator(nn.Module):
    def __init__(self, losses, weights):
        super(LossAggregator, self).__init__()
        self.losses = losses
        self.weights = weights

    def forward(self, input, target):
        losses = [loss(input, target) * weight for loss, weight in zip(self.losses, self.weights)]
        return sum(losses)

    def set_weights(self, new_weights):
        self.weights = new_weights