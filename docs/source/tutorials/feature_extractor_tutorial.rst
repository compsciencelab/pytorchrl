Create custom feature extractors
================================

In Deep Reinforcement Learning (DRL), Actors use Neural Networks to predict things such as the probability distribution over the next action, the value score or the Q-value.

In PyTorchRL, Neural Networks are implemented using Pytorch. More specifically, PyTorchRL Actors contain function approximators of ``pytorchrl.agent.actors.neural_networks.neural_network.NNBase`` `class <https://github.com/PyTorchRL/pytorchrl/blob/master/pytorchrl/agent/actors/neural_networks/neural_network.py>`_. ``NNBase`` accepts a Pytorch nn.Module subclass as a parameter and creates an instance of it to extract features from environment observations. On top of the feature extractor, ``NNBase`` optionally adds a recurrent layer. It finally ends it with a feed-forward layer to match a specified number of outputs. ``Actor`` components should take as input which feature extractor to use.

Therefore, experimenting with different network architectures is as easy as changing the feature extractor class provided to the Actor component. For example, here we have the default feature extractor used by PyTorchRL Actors, a multilayer perceptron network. The complete list of included feature extractors can be found `here <https://pytorchrl.readthedocs.io/en/latest/package/agent/actors/feature_extractors.html>`_.

.. code-block:: python

    import torch
    import torch.nn as nn
    from .utils import init


    class MLP(nn.Module):
        """
        Multilayer Perceptron network

        Parameters
        ----------
        input_shape : tuple
            Shape input tensors.
        hidden_sizes : list
            Hidden layers sizes.
        activation : func
            Non-linear activation function.

        Attributes
        ----------
        feature_extractor : nn.Module
            Neural network feature extractor block.
        """
        def __init__(self, input_shape, hidden_sizes=[256, 256], activation=nn.ReLU):
            super(MLP, self).__init__()

            # Define feature extractor
            layers = []
            sizes = [input_shape[0]] + hidden_sizes
            for j in range(len(sizes) - 1):
                layers += [(nn.Linear(sizes[j], sizes[j + 1]), activation()]
            self.feature_extractor = nn.Sequential(*layers)

            self.train()


        def forward(self, inputs):
            """
            Forward pass Neural Network

            Parameters
            ----------
            inputs : torch.tensor
                Input data.

            Returns
            -------
            out : torch.tensor
                Output feature map.
            """
            out = self.feature_extractor(inputs)
            return out

Defining new feature extractors is also simple. As only requirements, all features extractor classes should take as input the ``input_shape`` parameter, a tuple containing the shapes of the observations provided by the environment at every time step, and implement the ``nn.Module.forward`` method.

Use examples:

.. code-block:: python

    from pytorchrl.core.actors import OnPolicyActorCritic
    from pytorchrl.core.actors.neural_networks.feature_extractors.mlp import MLP

    actor_factory = OnPolicyActorCritic.create_factory(observation_space, action_space, feature_extractor_network=MLP)

and

.. code-block:: python

    from pytorchrl.core.actors import OnPolicyActorCritic, get_feature_extractor

    actor_factory = OnPolicyActorCritic.create_factory(observation_space, action_space, feature_extractor_network=get_feature_extractor("CNN"))

Create ``OnPolicyActorCritic`` Actors extracting features with a multilayer perceptron (MLP) and a Convolutional Neural Network (CNN) respectively.

.. note::
    To simplify the import of feature extractors classes already included in PyTorchRL, the ``get_feature_extractor`` method can be imported from ``pytorchrl.core.actors``. This methods returns a class from its name. See code `here <https://github.com/PyTorchRL/pytorchrl/blob/master/pytorchrl/core/actors/neural_networks/feature_extractors/__init__.py>`_.

