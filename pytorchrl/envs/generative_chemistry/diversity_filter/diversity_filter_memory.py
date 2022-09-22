import pandas as pd


class DiversityFilterMemory:

    def __init__(self):
        df_dict = {"Step": [], "SMILES": []}
        self._memory_dataframe = pd.DataFrame(df_dict)

    def update(self, smile: str):
        if not self.smiles_exists(smile):
            self._add_to_memory_dataframe(smile)

    def _add_to_memory_dataframe(self, smile: str):
        data, headers = [], []
        headers.append("SMILES")
        data.append(smile)
        new_data = pd.DataFrame([data], columns=headers)
        self._memory_dataframe = pd.concat([self._memory_dataframe, new_data], ignore_index=True, sort=False)

    def get_memory(self) -> pd.DataFrame:
        return self._memory_dataframe

    def set_memory(self, memory: pd.DataFrame):
        self._memory_dataframe = memory

    def smiles_exists(self, smiles: str):
        if len(self._memory_dataframe) == 0:
            return False
        return smiles in self._memory_dataframe['SMILES'].values

    def number_of_smiles(self):
        return len(set(self._memory_dataframe["SMILES"].values))
