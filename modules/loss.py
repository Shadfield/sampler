import logging
from collections import defaultdict

import torch
import torch.nn as nn


class WeightedLoss(nn.Module):
    """
    Combines losses and weightings into one loss
    Must be placed on same device as the model
    Will handle device transfer of any losses
    Provides useful information upon error in loss
    """

    def __init__(self, losses: dict, weights: list):
        """
        losses: {loss_name: module}
        weights: list of weights
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.weights = {}
        self.accumulator = nn.Parameter(torch.tensor(0.), requires_grad=False)
        self.last_loss = defaultdict(lambda: self.accumulator.clone().detach())

        # Register losses
        assert len(losses) == len(weights), "Must have same number of weights and losses"
        for loss, weight in zip(losses.keys(), weights):
            self.weights[loss] = weight
            self.add_module(loss, losses[loss])
            self.logger.info(f"Loss {loss}: {type(losses[loss]).__name__} ({weight})")

    def forward(self, lossargs: dict):
        """
        lossargs: {loss_name: tuple(arguments)}
        """
        total_loss = self.accumulator.clone().detach()

        for loss_name, args in lossargs.items():
            try:
                loss_fn = getattr(self, loss_name)
                loss = loss_fn(*args) * self.weights[loss_name]
                total_loss += loss
                self.last_loss[loss_name] += loss.detach()
                self.last_loss['total'] += loss.detach()

            except Exception as e:
                self.logger.error(f"Error in loss {loss_name}")
                raise e

        return total_loss

    def __getitem__(self, loss_name):
        return getattr(self, loss_name)

    def reset(self):
        self.last_loss = defaultdict(lambda: self.accumulator.clone().detach())
