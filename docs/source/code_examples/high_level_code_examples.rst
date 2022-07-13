Simplified Code Examples 
========================

The simplified code examples have been created for users who are new to the field of Deep Reinforcement Learning. Due to the :ref:`flexible configuration`, it is possible for even newcomers to train a wide variety of algorithms in different environments without having to make any changes to the code. This allows quick testing and experimentation with the option to easily adjust important settings if necessary.

In the following, examples are given to explain how the settings can be adjusted.

- :ref:`Train Agents`
    - :ref:`Run Default Code Example`
    - :ref:`Train Different Agent`
    - :ref:`Train On Different Environment`
    - :ref:`Train On Your Custom Environment`
- :ref:`Advanced Training Config Changes`
    - :ref:`Change Agent Details`
    - :ref:`Change Agent Actor Architecture`
    - :ref:`Change Agent Storage`
    - :ref:`Change Training Scheme` 
- :ref:`Config`
    - :ref:`Overall Config Structure`
- :ref:`Available Algorithms`


Train Agents
------------

Run Default Code Example
~~~~~~~~~~~~~~~~~~~~~~~~
To run the code example execute: 

.. code-block:: console

    python code_examples/simplified_code_examples/run.py

This will execute the default code example, running PPO :footcite:`schulman2017proximal` on the OpenAI Gym :footcite:`DBLP:journals/corr/BrockmanCPSSTZ16` environment ``CartPole-v0``.


Train Different Agent
~~~~~~~~~~~~~~~~~~~~~
To change the default code example and train another agent there are two ways to adapt the code. Either you go into the overall config and change in ``cfg/conf.yaml`` the default agent parameter to another agent e.g. Soft Actor-Critic :footcite:`DBLP:journals/corr/abs-1801-01290` (SAC). Or you just override the default configuration by an additional terminal input that defines the agent new, e.g. training ``sac`` on the default ``CartPole-v0`` environment:

.. code-block:: console

    python code_examples/simplified_code_examples/run.py agent=sac

For the possible agents you can train visit the section :ref:`Available Algorithms` in the documentation.

Train On Different Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In the case you want to train on a different environment you can change that similar to the agent in two ways either in the default conf.yaml file or via the terminal input, e.g. training ``sac`` on the PyBullet :footcite:`coumans2021` Environments:

.. code-block:: console

    python code_examples/simplified_code_examples/run.py agent=sac environment=pybullet

Here the default task is set to ``AntBulletEnv-v0``. If you want to change that just add the depending environment ID to the input. For example if you want to train on the ``HalfCheetahBulletEnv-v0``:

.. code-block:: console

    python code_examples/simplified_code_examples/run.py agent=sac  environment=pybullet environment.task=HalfCheetahBulletEnv-v0

For the possible environments you can train the PyTorchRL agents visit the section :ref:`available_environments` in the documentation.

Train On Your Custom Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Will be updated soon!


Advanced Training Config Changes
--------------------------------
In this section we cover the options if you want to on top of agent and training environment also want to adapt the training scheme and agent details like architecture or storage.


Change Agent Details
~~~~~~~~~~~~~~~~~~~~
In case you want to change the default parameter of the selected agent you can have a look at your specific agent in the config what hyperparameters exist and how they are set as default. In the case of PPO check:

.. code-block:: console

    code_examples/simplified_code_examples/cfg/agent/ppo.yaml

If you decide you want to change for example the learning rate for PPO you can do it the following way:

.. code-block:: console

    python code_examples/simplified_code_examples/run.py agent=ppo agent.ppo_config.lr=1.0e-2

Similar you can change any other hyperparameter in PPO or of other agents in PyTorchRL. 

Change Agent Actor Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Similarly to the agent hyperparameter you can also change the overall architecture of the actors. Meaning, add additional layer to the policy network of PPO or change to a recurrent policy at all. You can see all possible parameters to change at: 

.. code-block:: console

    code_examples/simplified_code_examples/cfg/agent/actor

Inside here you have a yaml file for off-policy algorithms like DDPG, TD3, SAC and a on-policy file for algorithms like PPO. That said, if you decide to change the PPO policy to be a recurrent neural network you can do so with: 

.. code-block:: console

    python code_examples/simplified_code_examples/run.py agent=ppo agent.actor.recurrent_nets=True


Change Agent Storage
~~~~~~~~~~~~~~~~~~~~
Currently changes regarding the storage types need to be done directly in the config files. But this will be changed and updated in the future!

Change Training Scheme
~~~~~~~~~~~~~~~~~~~~~~
In this section we show you how you can change the training scheme so that you can scale up your experiments.
Will be updated soon!

Config 
------
This section visualizes the overal config structure in case you want to dont want to adapt your training run parameters via terminal inputs and specify new default parameters.

Overall Config Structure
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: console

    cfg
    │   README.md
    │   conf.yaml    
    │
    └───agent
    |   |   ppo.yaml
    │   │   ddpg.yaml
    │   │   td3.yaml
    │   │   sac.yaml 
    │   │   mpo.yaml
    │   │   
    │   └───actor
    │   |      off_policy.yaml
    │   |      on_policy.yaml
    │   |
    |   └───storage
    |          gae_buffer.yaml
    |          replay_buffer.yaml
    |          her_buffer.yaml
    |
    └───scheme
    |      a3c.yaml
    |      apex.yaml
    |      ddppo.yaml
    |      default.yaml
    |      impala.yaml
    |      r2d2.yaml
    |      rapid.yaml
    |
    └───environment
        atari.yaml
        causalworld.yaml
        crafter.yaml
        gym.yaml
        mujoco.yaml
        pybullet.yaml

Available Algorithms
--------------------
In this section you can see all possible algorithms that can be utilized with the simplified code examples. 

Off-Policy Algorithms
~~~~~~~~~~~~~~~~~~~~~
- Deep Deterministic Policy Gradient :footcite:`ddpg` (DDPG) in the config used as ``ddpg``
- Twin Delayed DDPG :footcite:`td3` (TD3) in the config used as ``td3``
- Soft Actor-Critic :footcite:`DBLP:journals/corr/abs-1801-01290` (SAC) in the config uses as ``sac``
- Maximum a Posteriori Policy Optimisation :footcite:`mpo` (MPO) in the config used as ``mpo``

On-Policy Algorithms
~~~~~~~~~~~~~~~~~~~~
- Proximal Policy Optimisation :footcite:`schulman2017proximal` (PPO) in the config used as ``ppo``


----------

.. footbibliography::