import abc
import pandas as pd
from reinvent_chemistry.conversions import Conversions
from pytorchrl.envs.generative_chemistry.diversity_filter.diversity_filter_memory import DiversityFilterMemory


class BaseDiversityFilter(abc.ABC):

    @abc.abstractmethod
    def __init__(self, minscore=0.4, bucket_size=25, minsimilarity=0.0):
        self.minscore = minscore
        self.bucket_size = bucket_size
        self.minsimilarity = minsimilarity
        self._chemistry = Conversions()
        self._diversity_filter_memory = DiversityFilterMemory()

    @abc.abstractmethod
    def update_score(self, score, smile):
        raise NotImplementedError("The method 'evaluate' is not implemented!")

    def get_memory_as_dataframe(self) -> pd.DataFrame:
        return self._diversity_filter_memory.get_memory()

    def set_memory_from_dataframe(self, memory: pd.DataFrame):
        self._diversity_filter_memory.set_memory(memory)

    def number_of_smiles_in_memory(self) -> int:
        return self._diversity_filter_memory.number_of_smiles()

    def _smiles_exists(self, smile):
        return self._diversity_filter_memory.smiles_exists(smile)

    def _add_to_memory(self, smile):
        self._diversity_filter_memory.update(smile)

