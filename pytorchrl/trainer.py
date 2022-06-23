import os
import sys
import time

from pytorchrl.agent.env import VecEnv
from pytorchrl.learner import Learner
from pytorchrl.scheme import Scheme
from pytorchrl.utils import save_argparse, cleanup_log_dir

ON_POLICY_ALGOS = ["PPO", "PPOD"]
OFF_POLICY_ALGOS = ["DQN", "DDPG", "TD3", "SAC"]

class Trainer():
    def __init__(self, config, custom_environment_factory=None):
        print(config)
        cleanup_log_dir(config.log_dir)
        # not sure if this is needed anymore since we store it anyway on wandb
        # save_argparse(config, os.path.join(config.log_dir, "conf.yaml"),[])
        self.config = config
        self.custom_environment_factory = custom_environment_factory
        
        self.train_envs_factory, self.test_envs_factory, self.action_space, self.obs_space = self.setup_environment()
        self.algo_factory, self.algo_name = get_algorithm(self.config)
        self.actor_factory = self.setup_algorithm()
        self.storage_factory = self.setup_storage()
        self.scheme = self.setup_scheme()

        
    def train(self, ):
        iterations = 0
        start_time = time.time()
        while not self.learner.done():

            self.learner.step()

            if iterations % self.config.log_interval == 0:
                self.learner.print_info()

            if iterations % self.config.save_interval == 0:
                save_name = self.learner.save_model()

            if self.config.max_time != -1 and (time.time() - start_time) > self.config.max_time:
                break

            iterations += 1

        print("Finished!")
        sys.exit()

    def setup_environment(self,):
        if self.custom_environment_factory is not None:
            environment_train_factory = self.custom_environment_factory
        else:
            environment_train_factory, environment_test_factory = self.get_enviroment_factory(self.config.environment)
        # 1. Define Train Vector of Envs

        train_envs_factory, action_space, obs_space = VecEnv.create_factory(
            env_fn=environment_train_factory,
            env_kwargs=self.config.task,
            vec_env_size=self.config.num_env_processes, log_dir=self.config.log_dir)

        # 2. Define Test Vector of Envs (Optional)
        test_envs_factory, _, _ = VecEnv.create_factory(
            env_fn=environment_test_factory,
            env_kwargs=self.config.task,
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
                                                          noise=self.config.agent.ddpg_config.noise)
            return actor_factory
        else:
            pass
        
    def setup_storage(self,):
        print(self.config.agent)
        if self.config.agent.storage.name == "replay_buffer":
            from pytorchrl.agent.storages import ReplayBuffer
            return  ReplayBuffer.create_factory(size=self.config.agent.storage.size)
        elif self.config.agent.storage.name == "her_buffer":
            from pytorchrl.agent.storages import HERBuffer
            return HERBuffer.create_factory(size=self.config.agent.storage.name)
        elif self.config.agent.stroage.name == "gae_buffer":
            from pytorchrl.agent.storages import GAEBuffer
            return GAEBuffer.create_factory(size=self.config.agent.storage.size,
                                            gae_lambda=self.config.agent.storage.gae_lambda)
        else:
            pass
    
    
    def setup_scheme(self,):
        params = {"algo_factory": self.algo_factory,
                  "actor_factory": self.actor_factory,
                  "storage_factory": self.storage_factory,
                  "train_envs_factory": self.train_envs_factory,
                  "test_envs_factory": self.test_envs_factory,
                  "num_col_workers": self.config.scheme.num_col_workers,
                  "col_workers_communication": self.config.scheme.com_col_workers,
                  "col_workers_resources": {"num_cpus": self.config.scheme.col_workers_resources.num_cpus,
                                            "num_gpus": self.config.scheme.col_workers_resources.num_gpus},
                  "num_grad_workers": self.config.scheme.num_grad_workers,
                  "grad_workers_communication": self.config.scheme.com_grad_workers,
                  "grad_workers_resources": {"num_cpus": self.config.scheme.grad_workers_resources.num_cpus,
                                             "num_gpus": self.config.scheme.grad_workers_resources.num_gpus},
                  }
        return Scheme(**params)
    
    def setup_learner(self, ):
        return  Learner(self.scheme,
                        target_steps=self.config.num_env_steps, log_dir=self.config.log_dir)
    
    # might get extracted to additional utils file 
    def get_enviroment_factory(self, task):
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
        algo_factory, algo_name = DDPG.create_factory(lr_pi=config.agent.ddpg_config.lr_pi,
                                                      lr_q=config.agent.ddpg_config.lr_q,
                                                      gamma=config.agent.ddpg_config.gamma,
                                                      max_grad_norm=config.agent.ddpg_config.max_grad_norm,
                                                      polyak=config.agent.ddpg_config.polyak,
                                                      num_updates=config.agent.ddpg_config.num_updates,
                                                      update_every=config.agent.ddpg_config.update_every,
                                                      test_every=config.agent.ddpg_config.test_every,
                                                      num_test_episodes=config.agent.ddpg_config.num_test_episodes,
                                                      start_steps=config.agent.ddpg_config.start_steps,
                                                      mini_batch_size=config.agent.ddpg_config.mini_batch_size)
        return algo_factory, algo_name
    else:
        pass