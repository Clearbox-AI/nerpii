from typing import Dict
import pandas as pd

class NerAnalyzer:
    dataframe: pd.DataFrame
    dict_global_entities: Dict


    def __init__(self, df_input_path: pd.DataFrame) -> "NerAnalyzer":
        self.dataframe = pd.read_csv(df_input_path)
        self.dict_global_entities = self.get_global_entities()


    def get_global_entities(self, data_sample: int = 500) -> Dict:
        df_input = df_input.sample(n=min(data_sample, self.dataframe.shape[0]))
        

