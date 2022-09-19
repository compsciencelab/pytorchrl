## PyTorchRL: A PyTorch library for reinforcement learning

Deep Reinforcement learning (DRL) has been very successful in recent years but current methods still require vast amounts of data to solve non-trivial environments.  Scaling to solve more complex tasks requires frameworks that are flexible enough to allow prototyping and testing of new ideas, yet avoiding the impractically slow experimental turnaround times associated to single-threaded implementations.  PyTorchRL is a pytorch-based library for DRL that allows to easily assemble RL agents using a set of core reusable and easily extendable sub-modules as building blocks.  To reduce training times, PyTorchRL allows scaling agents with a parameterizable component called Scheme, that permits to define distributed architectures with great flexibility by specifying which operations should be decoupled, which should be parallelized, and how parallel tasks should be synchronized.

### Installation

```
    conda create -y -n pytorchrl
    conda activate pytorchrl

    conda install pytorch torchvision cudatoolkit -c pytorch
    
    pip install pytorchrl
```

### Documentation

PyTorchRL documentation can be found [here](https://pytorchrl.readthedocs.io/en/latest/).

### Citing PyTorchRL
Here is the [paper](https://arxiv.org/abs/2007.02622)

```
@misc{bou2021pytorchrl,
      title={PyTorchRL: Modular and Distributed Reinforcement Learning in PyTorch}, 
      author={Albert Bou and Gianni De Fabritiis},
      year={2021},
      eprint={2007.02622},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
