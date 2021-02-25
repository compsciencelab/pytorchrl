Installation
============

Installing Anaconda or Miniconda
--------------------------------

If not already done, install conda (Miniconda is sufficient). To do so, see the `official documentation. <https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`_

Installing PyTorchRL library
----------------------------

1. Set up conda environment ::

    conda create -y -n pytorchrl # requires python3.7 or above
    conda activate pytorchrl

2. Install dependencies ::

    conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
    pip install git+https://github.com/PyTorchRL/baselines.git

3. Install package. It can be installed either via PyPI or from source ::

    # PyPI installation
    pip install pytorchrl

::

    # source installation
    git clone git@github.com:PyTorchRL/pytorchrl.git
    cd pytorchrl
    python -m pip install -e .

4. To quickly test if installation was successful you can execute the following script, which runs a few training steps of the ``AC2`` algorithm on the ``CartPole-v0`` environment.

.. code-block:: python

    #!/usr/bin/env python3

    import ray
    import gym
    from pytorchrl import Learner
    from pytorchrl.scheme import Scheme
    from pytorchrl.agent.algos import A2C
    from pytorchrl.agent.env import VecEnv
    from pytorchrl.agent.storages import VanillaOnPolicyBuffer
    from pytorchrl.agent.actors import OnPolicyActor

    def cartpole_train_env_factory(seed=0):
        env = gym.make("CartPole-v0")
        env.seed(seed)
        return env

    if __name__ == "__main__":

        ray.init()

        # 1. Define Train Vector of Envs
        train_envs_factory, action_space, obs_space = VecEnv.create_factory(
            vec_env_size=1, env_fn=cartpole_train_env_factory)

        # 3. Define RL training algorithm
        algo_factory = A2C.create_factory(lr_v=1e-3, lr_pi=1e-3, gamma=0.99)

        # 4. Define RL Policy
        actor_factory = OnPolicyActor.create_factory(obs_space, action_space)

        # 5. Define rollouts storage
        storage_factory = VanillaOnPolicyBuffer.create_factory(size=500)

        # 6. Define scheme
        scheme = Scheme(algo_factory, actor_factory, storage_factory, train_envs_factory)

        # 7. Define learner
        learner = Learner(scheme, target_steps=25000)

        # 8. Define train loop
        while not learner.done():
            learner.step()
            learner.print_info()

        print("Successful installation!")
