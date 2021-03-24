import gym
from gym.spaces.box import Box


class TransposeImagesIfRequired(gym.ObservationWrapper):
    """
    When environment observations are images, this wrapper transposes
    the axis. It is useful when the images have shape (W,H,C), as they can be
    transposed "on the fly" to (C,W,H) for PyTorch convolutions to be applied.

    Parameters
    ----------
    env : gym.Env
        Original Gym environment, previous to applying the wrapper.
    op : list
        New axis ordering.
    """

    def __init__(self, env=None, op=[2, 0, 1]):
        """Transpose observation space for images"""
        super(TransposeImagesIfRequired, self).__init__(env)

        self.op = op

        if isinstance(self.observation_space, gym.spaces.Box) and \
                len(self.observation_space.shape) == 3:
            obs_shape = self.observation_space.shape
            self.observation_space = Box(
                self.observation_space.low[0, 0, 0],
                self.observation_space.high[0, 0, 0],
                [obs_shape[self.op[0]], obs_shape[self.op[1]], obs_shape[self.op[2]]],
                dtype=self.observation_space.dtype)

        elif isinstance(self.observation_space, gym.spaces.Dict):
            for k in self.observation_space.spaces:
                if isinstance(self.observation_space[k], gym.spaces.Box) and \
                        len(self.observation_space[k].shape) == 3:
                    obs_shape = self.observation_space[k].shape
                    self.observation_space[k] = Box(
                        self.observation_space[k].low[0, 0, 0],
                        self.observation_space[k].high[0, 0, 0],
                        [obs_shape[self.op[0]], obs_shape[self.op[1]], obs_shape[self.op[2]]],
                        dtype=self.observation_space.dtype)

    def observation(self, ob):
        """Transpose observation"""

        if isinstance(ob, dict):
            for k in ob:
                if len(ob[k].shape) == 3:
                    ob[k] = ob[k].transpose(self.op[0], self.op[1], self.op[2])
        else:
            if len(ob.shape) == 3:
                ob = ob.transpose(self.op[0], self.op[1], self.op[2])

        return ob