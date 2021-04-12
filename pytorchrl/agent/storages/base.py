from abc import ABC, abstractmethod


class Storage(ABC):
    """
    Base class for all storage components. It should serve as a template to
    create new Storage classes with new or extended features.
    """

    def __init__(self, size, device, actor, algorithm, *args):
        """
        Initialize Storage class.

        Parameters
        ----------
        size : int
            Storage capacity along time axis.
        device: torch.device
            CPU or specific GPU where data tensors will be placed and class
            computations will take place. Should be the same device where the
            actor model is located.
        actor : Actor
            Actor class instance.
        algorithm : Algorithm
            Algorithm class instance
        """
        raise NotImplementedError


    @classmethod
    @abstractmethod
    def create_factory(cls, size, *args):
        """
        Returns a function to create new Storage instances.

        Parameters
        ----------
        size : int
            Storage capacity along time axis.
        """
        raise NotImplementedError

    @abstractmethod
    def init_tensors(self, sample, *args):
        """
        Lazy initialization of data tensors from a sample.

        Parameters
        ----------
        sample : dict
            Data sample (containing all tensors of an environment transition)
        """
        raise NotImplementedError

    @abstractmethod
    def get_all_buffer_data(self, data_to_cpu=False, *args):
        """
        Return all currently stored data. If data_to_cpu, moves
        data tensors to cpu memory.
        """
        raise NotImplementedError

    @abstractmethod
    def insert_data_slice(self, new_data, *args):
        """
        Add new_data to the buffer stored data.

        Parameters
        ----------
        new_data : dict
            Dictionary of env transition samples to replace self.data with.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self, *args):
        """Set class counters to zero and remove stored data"""
        raise NotImplementedError

    @abstractmethod
    def insert_transition(self, sample, *args):
        """
        Store new transition sample.

        Parameters
        ----------
        sample : dict
            Data sample (containing all tensors of an environment transition)
        """
        raise NotImplementedError

    @abstractmethod
    def before_gradients(self, actor, algo, *args):
        """
        Steps required before updating actor policy model.

        Parameters
        ----------
        actor : Actor class
            An actor class instance.
        algo : Algo class
            An algorithm class instance.
        """
        raise NotImplementedError

    @abstractmethod
    def after_gradients(self, actor, algo, info, *args):
        """
        Steps required after updating actor policy model

        Parameters
        ----------
        actor : Actor class
            An actor class instance.
        algo : Algo class
            An algorithm class instance.
        info : dict
            Additional relevant info from gradient computation.

        Returns
        -------
        info : dict
            info dict updated with relevant info from Storage.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_batches(self, num_mini_batch, mini_batch_size, num_epochs=1, *args):
        """
        Returns a batch iterator to update actor critic.

        Parameters
        ----------
        num_mini_batch : int
           Number mini batches per epoch.
        mini_batch_size : int
            Number of samples contained in each mini batch.
        num_epochs : int
            Number of epochs.
        shuffle : bool
            Whether to shuffle collected data or generate sequential

        Yields
        ------
        batch : dict
            Generated data batches.
        """
        raise NotImplementedError

    @abstractmethod
    def update_storage_parameter(self, parameter_name, new_parameter_value, *args):
        """
        If `parameter_name` is an attribute of the algorithm, change its value
        to `new_parameter_value value`.

        Parameters
        ----------
        parameter_name : str
            Attribute name
        new_parameter_value : int or float
            New value for `parameter_name`.
        """
        raise NotImplementedError
