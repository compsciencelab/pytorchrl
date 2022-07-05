import numpy as np
import gym.spaces


class Char(gym.spaces.MultiDiscrete):
    """ Character observation/action space

    This space consists of a series of `gym.spaces.Discrete` objects all with
    the same parameters. Each `gym.spaces.Discrete` can take integer values
    between 0 and len(self.vocab).
    """

    def __init__(self, vocab, max_length=100):
        """
        Parameters
        ----------
        vocab : list of char, optional
            Vocabulary defining this space. It shouldn't contain any
            duplicate characters.
        max_length : int
            Maximum number of characters in a text.
        """

        if len(vocab) != len(set(vocab)):
            raise ValueError("Vocabulary has duplicated tokens")

        self.vocab = vocab
        self.max_length = max_length
        self.individual_shape = (len(vocab),)

        super().__init__([len(self.vocab) - 1] * self.max_length)
        self.dtype = np.int64  # Overwrite Gym's dtype=int8.

    def filter_unknown(self, text):
        """ Strip out all characters not in the vocabulary. """
        return "".join(c for c in text if c in self.vocab)

    def __repr__(self):
        return "Character({})".format(self.max_length)
