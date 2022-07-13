Motivation
==========

Deep Reinforcement Learning (RL) has been very successful in recent years but current methods still require vast amounts of data to solve non-trivial environments. Scaling to solve more complex tasks requires frameworks that are flexible enough to allow prototyping and testing of new ideas, yet avoiding the impractically slow experimental turnaround times associated to single-threaded implementations. PyTorchRL is a pytorch-based library for RL that allows to easily assemble RL agents using a set of core reusable and easily extendable sub-modules as building blocks. PyTorchRL permits the definition of distributed training architectures with flexibility and independence of the Agent components. In combination, these two features can accelerate the pace at which ideas are implemented and tested, simplifying research and enabling to tackle more challenging RL problems.

.. image:: ../images/logo.png
  :width: 300
  :align: center
