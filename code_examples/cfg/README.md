# Code example documentation 

## Train Agents 
### Run Default Code Example 
To run the code example execute: 

`python code_examples/run.py`

This will execute the default code example, running [PPO]() on the [OpenAI gym]() environment `CartPole-v0`.
(maybe add link to wandb project showing the results that are to expected)


### Train Different Agent
To change the default code example and train another agent there are two ways to adapt the code. Either you go into the overall config and change in cfg/conf.yaml the default agent parameter to another agent e.g. sac, td3 or ddpg. Or you just override the default configuration by an additional terminal input that defines the agent new, e.g. training `sac` on the default `CartPole-v0` environment:

`python code_examples/run.py agent=sac`

For the possible agents you can train visit the section [Implemented Agents in PyTorchRL]().

### Train On Different Environment
In the case you want to train on a different environment you can change that similar to the agent in two ways either in the default conf.yaml file or via the terminal input, e.g. training `sac` on the [PyBulletEnvironments]():

`python code_examples/run.py agent=sac task=pybullet`

Here the default task is set to `AntBulletEnv-v0`. If you want to change that just add the depending environment ID to the input. For example if you want to train on the `HalfCheetahBulletEnv-v0`:

`python code_examples/run.py agent=sac  task=pybullet task.env_id=HalfCheetahBulletEnv-v0`

For the possible environments you can train the PyTorchRL agents visit the section [Training Environments In PyTorchRL]().

### Train On Your Custom Environment

### Advanced Training Config Changes
In this section we cover the options if you want to on top of agent and training environment also want to adapt the training scheme and agent details like architecture or storage.

#### Change Training Scheme
TODO:

#### Change Agent Details
TODO:
##### Change Agent Architecture
TODO:
##### Change Agent Storage
TODO:
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
│   └───architecture
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
└───task
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
