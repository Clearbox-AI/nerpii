from typing import Any, Dict, List, Optional

from presidio_analyzer import AnalyzerEngine, BatchAnalyzerEngine, PatternRecognizer
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

class NerAnalyzer:
    dataframe: pd.DataFrame
    presidio_analyzer: BatchAnalyzerEngine
    nlp_model: Any
    #dict_global_entities: Dict


    def __init__(self, df_input_path: pd.DataFrame) -> "NerAnalyzer":
        self.dataframe = pd.read_csv(df_input_path)
        self.presidio_analyzer = self.set_analyzer()
        # self.nlp_model = self.set_nlp_model(9)
        

    def set_analyzer(self, addresses_recognizer: Optional[PatternRecognizer] = None) -> BatchAnalyzerEngine:
        """ 
        Return a batch analyzer used to assign entities.
        User can add a customized recognizer as parameter.
        

        Parameters
        ----------
        addresses_recognizer : Optional[PatternRecognizer], optional
        Address customized recognizer created with add_address_entity function , by default None

        Returns
        -------
        BatchAnalyzerEngine
            Batch analyzer used to assign entities
        """
        analyzer = AnalyzerEngine()
        if addresses_recognizer:
            analyzer.registry.add_recognizer(addresses_recognizer)
        batch_analyzer = BatchAnalyzerEngine(analyzer_engine=analyzer)

        return batch_analyzer 

    
    def set_nlp_model(self, model = "dslim/bert-base-NER") -> Any:
        """
        Return a pretrained nlp model downloaded from Hugging Face (https://huggingface.co/dslim/bert-base-NER)
        used to recognize ORGANIZATION entities.

        Parameters
        ----------
        model : str, optional
            String containing the name of the nlp model, by default "dslim/bert-base-NER"

        Returns
        -------
            Pipeline function from Hugging Face transformers library
        """

        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelForTokenClassification.from_pretrained(model)
        nlp_model = pipeline("ner", model=model, tokenizer=tokenizer)

        return nlp_model

    def scheda_analyzer(self):
        scheda = f"""
        Dataset: {self.dataframe}
        Presidio Analyzer: {self.presidio_analyzer}
        """

        return scheda



analyzer = NerAnalyzer('dataset/contact_list_Austin.csv')
analyzer.scheda_analyzer()


