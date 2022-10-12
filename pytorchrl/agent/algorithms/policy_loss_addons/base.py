from abc import ABC, abstractmethod
import torch


class PolicyLossAddOn(ABC):
    """Base class for all add ons to the policy loss."""

    @abstractmethod
    def setup(self, actor, device):
        """Initializes the class."""
        raise NotImplementedError

    @abstractmethod
    def compute_loss_term(self, batch, dist_entropy=None):
        """Calculates addon loss term."""
        raise NotImplementedError
