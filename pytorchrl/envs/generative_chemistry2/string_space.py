import numpy as np
import gym.spaces


class Char(gym.spaces.Discrete):
    """ Character observation/action space

    This space consists of a series of `gym.spaces.Discrete` objects all with
    the same parameters. Each `gym.spaces.Discrete` can take integer values
    between 0 and len(self.vocab).
    """

    def __init__(self, vocab):
        """
        Parameters
        ----------
        vocab : list of char, optional
            Vocabulary defining this space. It shouldn't contain any
            duplicate characters.
        """

        if len(vocab) != len(set(vocab)):
            raise ValueError("Vocabulary has duplicated tokens")

        self.vocab = vocab
        super().__init__(len(self.vocab) - 1)
        self.dtype = np.int64  # Overwrite Gym's dtype=int8.

    def filter_unknown(self, text):
        """ Strip out all characters not in the vocabulary. """
        return "".join(c for c in text if c in self.vocab)

