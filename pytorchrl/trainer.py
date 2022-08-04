import sys
import time

from pytorchrl.agent.env import VecEnv
from pytorchrl.learner import Learner
from pytorchrl.scheme import Scheme
from pytorchrl.utils import cleanup_log_dir

# backup checks for possible algos and envs
ENVS_LIST = ["gym", "pybullet", "mujoco", "causalworld", "crafter", "atari"]
ON_POLICY_ALGOS = ["PPO", "PPOD"]
OFF_POLICY_ALGOS = ["DQN", "DDPG", "TD3", "SAC", "MPO"]


class Trainer:
    def __init__(self, config, custom_environment_factory=None):
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
        self.learner = self.setup_learner()

    def train(self, wandb):
        iterations = 0
        start_time = time.time()
        while not self.learner.done():

            self.learner.step()

            if iterations % self.config.log_interval == 0:
                log_data = self.learner.get_metrics(add_episodes_metrics=True)
                log_data = {k.split("/")[-1]: v for k, v in log_data.items()}
                wandb.log(log_data, step=self.learner.num_samples_collected)
                self.learner.print_info()

            if iterations % self.config.save_interval == 0:
                save_name = self.learner.save_model()

            if self.config.max_time != -1 and (time.time() - start_time) > self.config.max_time:
                break

            iterations += 1

        print("Finished!")
        sys.exit()

    def setup_environment(self, ):
        if self.custom_environment_factory is not None:
            environment_train_factory = self.custom_environment_factory
        else:
            environment_train_factory, environment_test_factory = self.get_enviroment_factory(
                self.config.environment_name)
        # 1. Define Train Vector of Envs
        train_envs_factory, action_space, obs_space = VecEnv.create_factory(
            env_fn=environment_train_factory,
            env_kwargs=self.config.environment.train_env_config,
            vec_env_size=self.config.num_env_processes, log_dir=self.config.log_dir)

        # 2. Define Test Vector of Envs (Optional)
        if environment_test_factory is not None:
            test_envs_factory, _, _ = VecEnv.create_factory(
                env_fn=environment_test_factory,
                env_kwargs=self.config.environment.test_env_config,
                vec_env_size=self.config.num_env_processes, log_dir=self.config.log_dir)
        else:
            test_envs_factory = lambda v, x, y, z: None
        return train_envs_factory, test_envs_factory, action_space, obs_space

    def setup_algorithm(self, ):
        if self.config.agent.name in ON_POLICY_ALGOS:
            from pytorchrl.agent.actors import OnPolicyActor, get_feature_extractor, get_memory_network
            actor_factory = OnPolicyActor.create_factory(input_space=self.obs_space,
                                                         action_space=self.action_space,
                                                         algorithm_name=self.algo_name,
                                                         restart_model=self.config.restart_model,
                                                         recurrent_net=get_memory_network(self.config.agent.actor.recurrent_net),
                                                         recurrent_net_kwargs=self.config.agent.actor.recurrent_net_kwargs,
                                                         feature_extractor_kwargs=self.config.agent.actor.feature_extractor_kwargs,
                                                         feature_extractor_network=get_feature_extractor(self.config.agent.actor.feature_extractor_network),
                                                         shared_policy_value_network=self.config.agent.actor.shared_policy_value_network)
            return actor_factory
        elif self.config.agent.name in OFF_POLICY_ALGOS:
            from pytorchrl.agent.actors import OffPolicyActor, get_feature_extractor, get_memory_network
            actor_factory = OffPolicyActor.create_factory(input_space=self.obs_space,
                                                          action_space=self.action_space,
                                                          algorithm_name=self.algo_name,
                                                          noise=self.config.agent.noise,
                                                          restart_model=self.config.restart_model,
                                                          sequence_overlap=self.config.agent.sequence_overlap,
                                                          recurrent_net_kwargs=self.config.agent.actor.recurrent_net_kwargs,
                                                          recurrent_net=get_memory_network(self.config.agent.actor.recurrent_net),
                                                          obs_feature_extractor=get_feature_extractor(
                                                              self.config.agent.actor.obs_feature_extractor),
                                                          obs_feature_extractor_kwargs=self.config.agent.actor.obs_feature_extractor_kwargs,
                                                          act_feature_extractor=get_feature_extractor(
                                                              self.config.agent.actor.act_feature_extractor),
                                                          common_feature_extractor=get_feature_extractor(
                                                              self.config.agent.actor.common_feature_extractor),
                                                          common_feature_extractor_kwargs=self.config.agent.actor.common_feature_extractor_kwargs,
                                                          num_critics=self.config.agent.actor.num_critics)
            return actor_factory
        else:
            pass

    def setup_storage(self, ):
        if self.config.agent.storage.name == "replay_buffer":
            from pytorchrl.agent.storages import ReplayBuffer
            return ReplayBuffer.create_factory(size=self.config.agent.storage.size)
        elif self.config.agent.storage.name == "her_buffer":
            from pytorchrl.agent.storages import HERBuffer
            return HERBuffer.create_factory(size=self.config.agent.storage.name)
        elif self.config.agent.storage.name == "vanilla_on_policy":
            from pytorchrl.agent.storages import VanillaOnPolicyBuffer
            return VanillaOnPolicyBuffer.create_factory(size=self.config.agent.storage.size)
        elif self.config.agent.storage.name == "gae_buffer":
            from pytorchrl.agent.storages import GAEBuffer
            return GAEBuffer.create_factory(size=self.config.agent.storage.size,
                                            gae_lambda=self.config.agent.storage.gae_lambda)
        elif self.config.agent.storage.name == "vtrace":
            from pytorchrl.agent.storages import VTraceBuffer
            return VTraceBuffer.create_factory(size=self.config.agent.storage.size)
        else:
            pass

    def setup_scheme(self, ):
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
        return Learner(self.scheme,
                       target_steps=self.config.num_env_steps, log_dir=self.config.log_dir)

    # might get extracted to additional utils file 
    def get_enviroment_factory(self, task):
        if task == "pybullet":
            from pytorchrl.envs.pybullet import pybullet_train_env_factory, pybullet_test_env_factory
            return pybullet_train_env_factory, pybullet_test_env_factory
        elif task == "mujoco":
            from pytorchrl.envs.mujoco import mujoco_train_env_factory, mujoco_test_env_factory
            return mujoco_train_env_factory, mujoco_test_env_factory
        elif task == "gym":
            # TODO: maybe create extra factory for gym.
            from pytorchrl.envs.pybullet import pybullet_train_env_factory, pybullet_test_env_factory
            return pybullet_train_env_factory, pybullet_test_env_factory
        elif task == "causalworld":
            from pytorchrl.envs.causal_world import causal_world_train_env_factory
            return causal_world_train_env_factory, causal_world_train_env_factory
        elif task == "atari":
            from pytorchrl.envs.atari import atari_train_env_factory, atari_test_env_factory
            return atari_train_env_factory, atari_test_env_factory
        elif task == "crafter":
            from pytorchrl.envs.crafter import crafter_train_env_factory, crafter_test_env_factory
            return crafter_train_env_factory, crafter_test_env_factory
        elif task == "gen_chem":
            # TODO: 
            # from pytorchrl.envs.generative_chemistry.generative_chemistry_env_factory import generative_chemistry_train_env_factory
            # return generative_chemistry_train_env_factory, None
            raise NotImplementedError
        else:
            assert task in ENVS_LIST, "Selected environment is not in list of possible training environments: {} ".format(
                ENVS_LIST)


def get_algorithm(config):
    if config.agent.name == "PPO":
        from pytorchrl.agent.algorithms import PPO
        algo_factory, algo_name = PPO.create_factory(lr=config.agent.ppo_config.lr,
                                                     eps=config.agent.ppo_config.eps,
                                                     gamma=config.agent.ppo_config.gamma,
                                                     num_epochs=config.agent.ppo_config.num_epochs,
                                                     clip_param=config.agent.ppo_config.clip_param,
                                                     num_mini_batch=config.agent.ppo_config.num_mini_batch,
                                                     test_every=config.agent.ppo_config.test_every,
                                                     max_grad_norm=config.agent.ppo_config.max_grad_norm,
                                                     entropy_coef=config.agent.ppo_config.entropy_coeff,
                                                     value_loss_coef=config.agent.ppo_config.value_loss_coef,
                                                     num_test_episodes=config.agent.ppo_config.num_test_episodes,
                                                     use_clipped_value_loss=config.agent.ppo_config.use_clipped_value_loss,
                                                     )

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
                                                      mini_batch_size=config.agent.ddpg_config.mini_batch_size,
                                                      target_update_interval=config.agent.ddpg_config.target_update_interval)
        # TODO: check how to set empty list in hydra
        # policy_loss_addons=config.agent.sac_config.policy_loss_addons)
        return algo_factory, algo_name
    elif config.agent.name == "TD3":
        from pytorchrl.agent.algorithms import TD3
        algo_factory, algo_name = TD3.create_factory(lr_pi=config.agent.td3_config.lr_pi,
                                                     lr_q=config.agent.td3_config.lr_q,
                                                     gamma=config.agent.td3_config.gamma,
                                                     max_grad_norm=config.agent.td3_config.max_grad_norm,
                                                     polyak=config.agent.td3_config.polyak,
                                                     num_updates=config.agent.td3_config.num_updates,
                                                     update_every=config.agent.td3_config.update_every,
                                                     test_every=config.agent.td3_config.test_every,
                                                     num_test_episodes=config.agent.td3_config.num_test_episodes,
                                                     start_steps=config.agent.td3_config.start_steps,
                                                     mini_batch_size=config.agent.td3_config.mini_batch_size,
                                                     target_update_interval=config.agent.td3_config.target_update_interval)
        # TODO: check how to set empty list in hydra
        # policy_loss_addons=config.agent.sac_config.policy_loss_addons)
        return algo_factory, algo_name
    elif config.agent.name == "SAC":
        from pytorchrl.agent.algorithms import SAC
        algo_factory, algo_name = SAC.create_factory(lr_pi=config.agent.sac_config.lr_pi,
                                                     lr_q=config.agent.sac_config.lr_q,
                                                     lr_alpha=config.agent.sac_config.lr_alpha,
                                                     gamma=config.agent.sac_config.gamma,
                                                     polyak=config.agent.sac_config.polyak,
                                                     num_updates=config.agent.sac_config.num_updates,
                                                     test_every=config.agent.sac_config.test_every,
                                                     update_every=config.agent.sac_config.update_every,
                                                     start_steps=config.agent.sac_config.start_steps,
                                                     max_grad_norm=config.agent.sac_config.max_grad_norm,
                                                     initial_alpha=config.agent.sac_config.initial_alpha,
                                                     mini_batch_size=config.agent.sac_config.mini_batch_size,
                                                     num_test_episodes=config.agent.sac_config.num_test_episodes,
                                                     target_update_interval=config.agent.sac_config.target_update_interval
                                                     # TODO: check how to set empty list in hydra
                                                     # policy_loss_addons=config.agent.sac_config.policy_loss_addons 
                                                     )
        return algo_factory, algo_name
    elif config.agent.name == "MPO":
        from pytorchrl.agent.algorithms import MPO
        algo_factory, algo_name = MPO.create_factory(lr_pi=config.agent.mpo_config.lr_pi,
                                                     lr_q=config.agent.mpo_config.lr_q,
                                                     gamma=config.agent.mpo_config.gamma,
                                                     polyak=config.agent.mpo_config.polyak,
                                                     num_updates=config.agent.mpo_config.num_updates,
                                                     test_every=config.agent.mpo_config.test_every,
                                                     update_every=config.agent.mpo_config.update_every,
                                                     start_steps=config.agent.mpo_config.start_steps,
                                                     mini_batch_size=config.agent.mpo_config.mini_batch_size,
                                                     num_test_episodes=config.agent.mpo_config.num_test_episodes,
                                                     target_update_interval=config.agent.mpo_config.target_update_interval,
                                                     dual_constraint=config.agent.mpo_config.dual_constraint,
                                                     kl_mean_constraint=config.agent.mpo_config.kl_mean_constraint,
                                                     kl_var_constraint=config.agent.mpo_config.kl_var_constraint,
                                                     kl_constraint=config.agent.mpo_config.kl_constraint,
                                                     alpha_scale=config.agent.mpo_config.alpha_scale,
                                                     alpha_mean_scale=config.agent.mpo_config.alpha_mean_scale,
                                                     alpha_var_scale=config.agent.mpo_config.alpha_var_scale,
                                                     alpha_mean_max=config.agent.mpo_config.alpha_mean_max,
                                                     alpha_var_max=config.agent.mpo_config.alpha_var_max,
                                                     alpha_max=config.agent.mpo_config.alpha_max,
                                                     mstep_iterations=config.agent.mpo_config.mstep_iterations,
                                                     sample_action_num=config.agent.mpo_config.sample_action_num,
                                                     max_grad_norm=config.agent.mpo_config.max_grad_norm,
                                                     # TODO: check how to set empty list in hydra
                                                     # policy_loss_addons=config.agent.sac_config.policy_loss_addons 
                                                     )
        return algo_factory, algo_name
    else:
        pass
