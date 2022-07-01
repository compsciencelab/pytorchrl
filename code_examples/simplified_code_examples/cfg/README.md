# Code example documentation 
The simplified code examples have been created for users who are new to the field of Deep Reinforcement Learning. Due to the [flexible configuration](#config), it is possible for even newcomers to train a wide variety of algorithms in different environments without having to make any changes to the code. This allows quick testing and experimentation with the option to easily adjust important settings if necessary.

In the following, examples are given to explain how the settings can be adjusted.

## Train Agents 
### Run Default Code Example 
To run the code example execute: 

`python code_examples/simplified_code_examples/run.py`

This will execute the default code example, running [PPO](https://arxiv.org/abs/1707.06347) on the [OpenAI gym](https://www.gymlibrary.ml/) environment `CartPole-v0`.


### Train Different Agent
To change the default code example and train another agent there are two ways to adapt the code. Either you go into the overall config and change in cfg/conf.yaml the default agent parameter to another agent e.g. Soft Actor-Critic ([SAC](https://arxiv.org/abs/1801.01290)). Or you just override the default configuration by an additional terminal input that defines the agent new, e.g. training `sac` on the default `CartPole-v0` environment:

`python code_examples/simplified_code_examples/run.py agent=sac`

For the possible agents you can train visit the section [Implemented Agents in PyTorchRL](#implemented-agents-in-pytorchrl).

### Train On Different Environment
In the case you want to train on a different environment you can change that similar to the agent in two ways either in the default conf.yaml file or via the terminal input, e.g. training `sac` on the [PyBulletEnvironments](https://pybullet.org/wordpress/):

`python code_examples/simplified_code_examples/run.py agent=sac environment=pybullet`

Here the default task is set to `AntBulletEnv-v0`. If you want to change that just add the depending environment ID to the input. For example if you want to train on the `HalfCheetahBulletEnv-v0`:

`python code_examples/simplified_code_examples/run.py agent=sac  environment=pybullet environment.task=HalfCheetahBulletEnv-v0`

For the possible environments you can train the PyTorchRL agents visit the section [Training Environments In PyTorchRL](#training-environments-in-pytorchrl).

### Train On Your Custom Environment
Will be updated soon!

### Advanced Training Config Changes
In this section we cover the options if you want to on top of agent and training environment also want to adapt the training scheme and agent details like architecture or storage.

#### Change Training Scheme
In this section we show you how you can change the training scheme so that you can scale up your experiments.

[Will be updated soon!](#training-schemes-implemented-in-pytorchrl)

#### Change Agent Details
In case you want to change the default parameter of the selected agent you can have a look at your specific agent in the config what hyperparameters exist and how they are set as default. In the case of PPO check:

`code_examples/simplified_code_examples/cfg/agent/ppo.yaml`

If you decide you want to change for example the learning rate for PPO you can do it the following way:

`python code_examples/simplified_code_examples/run.py agent=ppo agent.ppo_config.lr=1.0e-2`

Similar you can change any other hyperparameter in PPO or of other agents in PyTorchRL. 

##### Change Agent Actor Architecture
Similarly to the agent hyperparameter you can also change the overall architecture of the actors. Meaning, add additional layer to the policy network of PPO or change to a recurrent policy at all. You can see all possible parameters to change at: 

`code_examples/simplified_code_examples/cfg/agent/actor`

Inside here you have a yaml file for off-policy algorithms like DDPG, TD3, SAC and a on-policy file for algorithms like PPO. That said, if you decide to change the PPO policy to be a recurrent neural network you can do so with: 

`python code_examples/simplified_code_examples/run.py agent=ppo agent.actor.recurrent_nets=True`


##### Change Agent Storage
Currently changes regarding the storage types need to be done directly in the config files. But this will be changed and updated in the future!


## Config
This section visualizes the overal config structure in case you want to dont want to adapt your training run parameters via terminal inputs and specify new default parameters. 
### Overall Config Structure

```
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
```

## Implemented Agents in PyTorchRL
Below and overview of the algorithms that currently can be used within the simplified code examples. Will be updated with further algorithms in the future. 


**Off-Policy Algorithms:**
- [DDPG](https://arxiv.org/abs/1509.02971)
- [TD3](https://arxiv.org/pdf/1802.09477.pdf)
- [SAC](https://arxiv.org/abs/1801.01290)
- [MPO](https://arxiv.org/abs/1806.06920)

**On-Policy Algorithms:**
- [PPO](https://arxiv.org/abs/1707.06347) 



## Training Environments In PyTorchRL
Below and overview of the environments that can be used to train agents in the simplified code examples. In parentheses the name for the environment to change the configuration.


**Environments:**
- [**OpenAI gym**](https://www.gymlibrary.ml/) (gym)
- [**Atari**](https://www.gymlibrary.ml/) from OpenAI gym (atari)
- [**PyBullet**](https://pybullet.org/wordpress/) Environment (pybullet)
- [**MuJoCo**](https://mujoco.org/) (mujoco)
- [**Causal World**](https://sites.google.com/view/causal-world) (causalworld)
- [**Crafter**](https://github.com/danijar/crafter) (crafter) 

## Training Schemes Implemented In PytorchRL
To be added in the future!
