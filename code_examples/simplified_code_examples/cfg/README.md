# Code example documentation 

## Train Agents 
### Run Default Code Example 
To run the code example execute: 

`python code_examples/simplified_code_examples/run.py`

This will execute the default code example, running [PPO]() on the [OpenAI gym]() environment `CartPole-v0`.
(maybe add link to wandb project showing the results that are to expected)


### Train Different Agent
To change the default code example and train another agent there are two ways to adapt the code. Either you go into the overall config and change in cfg/conf.yaml the default agent parameter to another agent e.g. sac, td3 or ddpg. Or you just override the default configuration by an additional terminal input that defines the agent new, e.g. training `sac` on the default `CartPole-v0` environment:

`python code_examples/simplified_code_examples/run.py agent=sac`

For the possible agents you can train visit the section [Implemented Agents in PyTorchRL]().

### Train On Different Environment
In the case you want to train on a different environment you can change that similar to the agent in two ways either in the default conf.yaml file or via the terminal input, e.g. training `sac` on the [PyBulletEnvironments]():

`python code_examples/simplified_code_examples/run.py agent=sac environment=pybullet`

Here the default task is set to `AntBulletEnv-v0`. If you want to change that just add the depending environment ID to the input. For example if you want to train on the `HalfCheetahBulletEnv-v0`:

`python code_examples/simplified_code_examples/run.py agent=sac  environment=pybullet environment.task=HalfCheetahBulletEnv-v0`

For the possible environments you can train the PyTorchRL agents visit the section [Training Environments In PyTorchRL]().

### Train On Your Custom Environment
TODO: 

### Advanced Training Config Changes
In this section we cover the options if you want to on top of agent and training environment also want to adapt the training scheme and agent details like architecture or storage.

#### Change Training Scheme
In this section we show you how you can change the training scheme so that you can scale up your experiments.
[Will be updated soon!](TODO)

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
[Will be updated soon!](TODO)

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
(either in this doc or different page)

## Training Environments In PyTorchRL
(either in this doc or different page)

## Training Schemes Implemented In PytorchRL
(either in this doc or different page)
