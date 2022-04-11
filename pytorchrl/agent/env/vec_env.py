import torch
from pytorchrl.agent.env.make_env import make_env
from pytorchrl.agent.env.vector_wrappers import VecPyTorch
from pytorchrl.agent.env.openai_baselines_dependencies.vec_env.dummy_vec_env import DummyVecEnv
from pytorchrl.agent.env.openai_baselines_dependencies.vec_env.subproc_vec_env import SubprocVecEnv


class VecEnv:
    """Class to handle creation of environment vectors"""

    @classmethod
    def create_factory(cls, env_fn, env_kwargs={}, vec_env_size=1, log_dir=None, info_keywords=()):
        """
        Returns a function to create a vector of environments of size
        num_processes, so it can be executed by any worker, remote or not.

        Parameters
        ----------
        env_fn : func
            Function to create the environment.
        env_kwargs : dict
            keyword arguments of env_fn.
        vec_env_size : int
            size of the vector of environments.
        log_dir : str
            Target path for envs to log information through bench.Monitor class.
        info_keywords : tuple
            Information keywords to be logged stored by bench.Monitor class.

        Returns
        -------
        make_vec_env : func
            Function to create a vector of environments.
        dummy_env.action_space : gym.Space
            Environments action space.
        dummy_env.observation_space: gym.Space
            Environments observation space.
        """

        def make_vec_env(device=torch.device("cpu"), index_col_worker=1, index_grad_worker=1, mode="train"):
            """Create and return a vector environment"""

            if mode == "train":
                env_indexes = range(1, vec_env_size + 1)
            else:
                env_indexes = range(1 + vec_env_size, 2 * vec_env_size + 1)

            envs = [make_env(
                env_fn=env_fn, env_kwargs=env_kwargs,
                index_col_worker=index_col_worker,
                index_grad_worker=index_grad_worker,
                index_env=i, log_dir=log_dir, mode=mode,
                info_keywords=info_keywords
            ) for i in env_indexes]

            if len(envs) > 1:
                envs = SubprocVecEnv(envs)
            else:
                envs = DummyVecEnv(envs)

            envs = VecPyTorch(envs, device)

            envs.env_kwargs = env_kwargs

            return envs

        dummy_env = [make_env(
            env_fn=env_fn, env_kwargs=env_kwargs, index_col_worker=0,
            index_grad_worker=0, index_env=0)]
        dummy_env = DummyVecEnv(dummy_env)

        cls.action_space = dummy_env.action_space
        cls.observation_space = dummy_env.observation_space
        dummy_env.envs[0].close()

        return make_vec_env, dummy_env.action_space, dummy_env.observation_space
