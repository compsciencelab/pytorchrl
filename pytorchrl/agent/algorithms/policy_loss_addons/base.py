from abc import ABC, abstractmethod
import torch


class PolicyLossAddOn(ABC):
    """Base class for all add ons to the policy loss."""

    @abstractmethod
    def setup(self, *args, **kwargs):
        """Initializes the class."""
        raise NotImplementedError

    @abstractmethod
    def compute_loss_term(self, actor, batch, dist_entropy=None, *args, **kwargs):
        """Calculates addon loss term."""
        raise NotImplementedError
