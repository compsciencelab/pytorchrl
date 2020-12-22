import gym
from gym.spaces.box import Box


class TransposeImage(gym.ObservationWrapper):
    """
    When environment observations are images, this wrapper allows to transpose
    the axis. It is useful when the images have shape (W,H,C), as they can be
    transposed "on the fly" to (C,W,H) for PyTorch convolutions to be applied.

    Parameters
    ----------
    env : gym.Env
        Original Gym environment, previous to applying the wrapper.
    op : list
        New axis ordering.

    Attributes
    ----------
    op : list
        New axis ordering.
    observation_space : gym.Space
        New observation space
    """

    def __init__(self, env=None, op=[2, 0, 1]):
        """Transpose observation space for images"""
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, "Error: Operation, {}, must be dim3".format(str(op))
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[self.op[0]], obs_shape[self.op[1]],
                obs_shape[self.op[2]]],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        """Transpose observation"""
        return ob.transpose(self.op[0], self.op[1], self.op[2])