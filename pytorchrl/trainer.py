
from pytorchrl.agent.env import VecEnv
from pytorchrl.learner import Learner
from pytorchrl.scheme import Scheme

ON_POLICY_ALGOS = ["PPO", "PPOD"]
OFF_POLICY_ALGOS = ["DQN", "DDPG", "TD3", "SAC"]

class Trainer():
    def __init__(self, config, custom_environment_factory=None):
        self.config = config
        self.custom_environment_factory = custom_environment_factory
        
        self.train_envs_factory, self.test_envs_factory, self.action_space, self.obs_space = self.setup_environment()
        self.algo_factory, self.algo_name = get_algorithm(self.config)
        self.algo_factory = self.setup_algorithm()
        
        

    def setup_environment(self,):
        if self.custom_environment_factory is not None:
            environment_train_factory = self.custom_environment_factory
        else:
            environment_train_factory, environment_test_factory = get_enviroment_factory(self.config.task)
        # 1. Define Train Vector of Envs
        train_envs_factory, action_space, obs_space = VecEnv.create_factory(
            env_fn=self.environment_train_factory,
            env_kwargs={self.config.task},
            vec_env_size=self.config.num_env_processes, log_dir=self.config.log_dir)

        # 2. Define Test Vector of Envs (Optional)
        test_envs_factory, _, _ = VecEnv.create_factory(
            env_fn=self.environment_test_factory,
            env_kwargs={self.config.task},
            vec_env_size=self.config.num_env_processes, log_dir=self.config.log_dir)
        
        return train_envs_factory, test_envs_factory, action_space, obs_space
        
    
    def setup_algorithm(self, ):
        if self.config.agent.name in ON_POLICY_ALGOS:
            from pytorchrl.agent.actors import OnPolicyActor
            actor_factory = OnPolicyActor.create_factory(self.obs_space,
                                                         self.action_space,
                                                         self.algo_name,
                                                         restart_model=self.config.restart_model)
            return actor_factory
        elif self.config.agent.name in OFF_POLICY_ALGOS:
            from pytorchrl.agent.actors import OffPolicyActor
            actor_factory = OffPolicyActor.create_factory(self.obs_space,
                                                          self.action_space,
                                                          self.algo_name,
                                                          restart_model=self.config.restart_model,
                                                          noise=self.config.agent.noise)
            return actor_factory
        else:
            pass
    
    
    
    
# might get extracted to additional utils file 
def get_enviroment_factory(task):
    if task == "pybullet":
        from pytorchrl.envs.pybullet import pybullet_train_env_factory, pybullet_test_env_factory
        return pybullet_train_env_factory, pybullet_test_env_factory
    elif task == "atari":
        pass
    else:
        pass
        
    
def get_algorithm(config):


    if config.agent.name == "PPO":
        from pytorchrl.agent.algorithms import PPO
        algo_factory, algo_name = PPO.create_factory(lr=config.agent.ppo_config.lr,
                                                     num_epochs=config.agent.ppo_config.ppo_epoch,
                                                     clip_param=config.agent.ppo_config.clip_param,
                                                     entropy_coef=config.agent.ppo_config.entropy_coef,
                                                     value_loss_coef=config.agent.ppo_config.value_loss_coef,
                                                     max_grad_norm=config.agent.ppo_config.max_grad_norm,
                                                     num_mini_batch=config.agent.ppo_config.num_mini_batch,
                                                     use_clipped_value_loss=config.agent.ppo_config.use_clipped_value_loss,
                                                     gamma=config.agent.ppo_config.gamma)
        
        return algo_factory, algo_name
    elif config.agent.name == "DDPG":
        from pytorchrl.agent.algorithms import DDPG
        algo_factory, algo_name = DDPG.create_factory(lr_pi=config.agent.ddpg_config.lr,
                                                      lr_q=config.agent.ddpg_config.lr,
                                                      gamma=config.agent.ddpg_config.gamma,
                                                      polyak=config.agent.ddpg_config.polyak,
                                                      num_updates=config.agent.ddpg_config.num_updates,
                                                      update_every=config.agent.ddpg_config.update_every,
                                                      start_steps=config.agent.ddpg_config.start_steps,
                                                      mini_batch_size=config.agent.ddpg_config.mini_batch_size)
        return algo_factory, algo_name
    else:
        pass