Feature Extractors
==================

Multilayer Perceptron (MLP)
---------------------------

.. autoclass:: pytorchrl.agent.actors.neural_networks.feature_extractors.mlp.MLP
   :members:
   :undoc-members:
   :show-inheritance:

Convolutional Neural Network (CNN)
---------------------------------

.. autoclass:: pytorchrl.agent.actors.neural_networks.feature_extractors.cnn.CNN
   :members:
   :undoc-members:
   :show-inheritance:

Residual CNN with Fixup initialization (Fixup)
----------------------------------------------

.. autoclass:: pytorchrl.agent.actors.neural_networks.feature_extractors.fixup_cnn.FixupCNN
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: FixupResidualModule

Multimodal Neural Network  (DictNet)
------------------------------------

.. autoclass:: pytorchrl.agent.actors.neural_networks.feature_extractors.dictnet.DictNet
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: FixupResidualModule