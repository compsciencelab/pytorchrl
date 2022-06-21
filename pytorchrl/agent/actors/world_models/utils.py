import torch


class StandardScaler(object):

    def __init__(self, device):
        self.input_mu = torch.zeros(1).to(device)
        self.input_std = torch.ones(1).to(device)
        self.target_mu = torch.zeros(1).to(device)
        self.target_std = torch.ones(1).to(device)
        self.device = device

    def fit(self, inputs, targets):
        """
        Runs two ops, one for assigning the mean of the data to the internal mean, and
        another for assigning the standard deviation of the data to the internal standard deviation.
        This function must be called within a 'with <session>.as_default()' block.

        Parameters
        ----------
        inputs : torch.Tensor
            A torch Tensor containing the input
        targets : torch.Tensor
            A torch Tensor containing the input

        """
        self.input_mu = torch.mean(inputs, dim=0, keepdims=True).to(self.device)
        self.input_std = torch.std(inputs, dim=0, keepdims=True).to(self.device)
        self.input_std[self.input_std < 1e-8] = 1.0
        self.target_mu = torch.mean(targets, dim=0, keepdims=True).to(self.device)
        self.target_std = torch.std(targets, dim=0, keepdims=True).to(self.device)
        self.target_std[self.target_std < 1e-8] = 1.0

    def transform(self, inputs, targets=None):
        """
        Transforms the input matrix data using the parameters of this scaler.

        Parameters
        ----------
        inputs : torch.Tensor
            A torch Tensor containing the points to be transformed.
        targets : torch.Tensor
            A torch Tensor containing the points to be transformed.

        Returns
        -------
        norm_inputs : torch.Tensor
            Normalized inputs
        norm_targets : torch.Tensor
            Normalized targets
        """
        norm_inputs = (inputs - self.input_mu) / self.input_std
        norm_targets = None
        if targets is not None:
            norm_targets = (targets - self.target_mu) / self.target_std
        return norm_inputs, norm_targets

    def inverse_transform(self, targets):
        """
        Undoes the transformation performed by this scaler.

        Parameters
        ----------
        targets : torch.Tensor
            A torch Tensor containing the points to be transformed.

        Returns
        -------
        output : torch.Tensor
            The transformed dataset.
        """
        output = self.target_std * targets + self.target_mu
        return output
