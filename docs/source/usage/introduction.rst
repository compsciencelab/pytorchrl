Motivation
==========

Deep Reinforcement learning (DRL) has been very successful in recent years but current methods still require vast amounts of data to solve non-trivial environments. Scaling to solve more complex tasks requires frameworks that are flexible enough to allow prototyping and testing of new ideas, yet avoiding the impractically slow experimental turnaround times associated to single-threaded implementations. NAPPO is a pytorch-based library for DRL that allows to easily assemble RL agents using a set of core reusable and easily extendable sub-modules as building blocks. To reduce training times, NAPPO allows scaling agents with a parametrizable component called Scheme, that permits to define distributed architectures with great flexibility by specifying which operations should be decoupled, which should be parallelized, and how parallel tasks should be synchronized.

