from typing import Any, Dict, List, Optional, Union

import pandas as pd

from presidio_analyzer import AnalyzerEngine, BatchAnalyzerEngine, PatternRecognizer
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline


def frequency(values: List, element: Any) -> float:
    """
    Calculate the frequency of an element in a list of values.
    
    Parameters
    ----------
    values : List
        List of values.
    element : Any
        Element to calculate the frequency of.
    
    Returns
    -------
    float
        Frequency of the element in the list.
    """
    return values.count(element) / len(values) if len(values) else 0

def add_address_entity(additional_addresses: Optional[List] = []) -> PatternRecognizer:
    """
    Return a customized presidio recognizer that can recognize ADDRESS entity.
    Some address-related words are already set, but user can add others. 

    Parameters
    ----------
    additional_addresses : Optional[List], optional
        A list in which user can add new address-related words, by default []

    Returns
    -------
    PatternRecognizer
        A customized presidio recognizer
    """
    
    addresses = ['Street', 'Rue', 'Via', 'Square', 'Avenue', 'Place', 'Strada', 'St', 'Lane', 
    'Road', 'Boulevard', 'Ln', "Rd", "Highway" "Drive", "Av", "Hwy", "Blvd", "Corso", "Piazza", 
    "Calle", "Plaza", "Avenida", "Rambla", "C/"]
    addresses = addresses + additional_addresses
    addresses_recognizer = PatternRecognizer(supported_entity="ADDRESS", deny_list=addresses)

    return addresses_recognizer


class NamedEntityRecognizer:
    """
    A class used to recognize named entities in a dataset.
    
    Attributes
    ----------
    dataset : pd.DataFrame
        A pandas dataframe containing a sample of the dataset.
    object_columns : List
        A list of the object columns of the dataset.
    presidio_analyzer : BatchAnalyzerEngine
        A Presidio BatchAnalyzerEngine instance.
    assigned_entities_cols : List
        A list of the object columns of the dataset for which entities have been assigned.
    model : Any
        A pretrained nlp model downloaded from Hugging Face
    model_entities : Dict
        A dictionary whose keys are object column names and whose values are a list 
        containing all the entities assigned to each value by the model
    dict_global_entities : Dict
        A dictionary whose keys have the same names of the dataframe columns and values 
        are dictionaries in which the entity associated to the column and its confidence 
        score are reported.


    Returns
    -------
    _type_
        _description_
    """
    
    dataset: pd.DataFrame
    object_columns: List
    presidio_analyzer: BatchAnalyzerEngine
    assigned_entities_cols: List
    model: Any
    model_entities: Dict
    dict_global_entities: Dict
    
    
    def __init__(self, df_input: Union[str, pd.DataFrame], data_sample: Optional[int] = 500, nan_filler: str = "?") -> "NamedEntityRecognizer":
        """
        Create a NamedEntityRecognizer instance.
        
        Parameters
        ----------
        df_input : Union[str, pd.DataFrame]
            A pandas dataframe or a path to a csv file.
        data_sample : Optional[int], optional
            Number of rows to sample from the dataframe, by default 500
        nan_filler : str, optional
            A string to fill the NaN values for object columns, by default "?"
        
        Returns
        -------
        NamedEntityRecognizer
            A NamedEntityRecognizer instance.
        """
        
        if not isinstance(df_input, pd.DataFrame):
            df_input = pd.read_csv(df_input)
            
        self.dataset = df_input.sample(n=min(data_sample, df_input.shape[0]))
        self.object_columns = list(self.dataset.select_dtypes(['object']).columns)
        # fill NaN values for object columns
        self.dataset.loc[:, self.object_columns] = self.dataset.loc[:, self.object_columns].fillna(nan_filler)


        self.presidio_analyzer = None
        self.model = None
        
        self.dict_global_entities = dict.fromkeys(list(self.dataset.columns))
        self.model_entities = {}
        self.assigned_entities_cols = []
        
    def set_presidio_analyzer(self, add_addresses_recognizer: Optional[bool] = True, additional_addresses: Optional[List] = []) -> None:
        """
        Set a Presidio BatchAnalyzer for the instance.
        
        Parameters
        ----------
        add_addresses_recognizer : Optional[bool], optional
            Whether to add a customized address recognizer, by default True
        additional_addresses : Optional[List], optional
            A list in which user can add new address-related words, by default []
        
        """
        analyzer = AnalyzerEngine()
        
        if add_addresses_recognizer:
            addresses_recognizer = add_address_entity(additional_addresses)
            analyzer.registry.add_recognizer(addresses_recognizer)
            
        self.presidio_analyzer = BatchAnalyzerEngine(analyzer_engine=analyzer)
        
    def set_model(self, nlp_model: str = "dslim/bert-base-NER") -> None:
        """
        Set a pretrained nlp model downloaded from Hugging Face 
        (https://huggingface.co/dslim/bert-base-NER) used to recognize ORGANIZATION entities.

        Parameters
        ----------
        nlp_model : str, optional
            A NLP model name, by default "dslim/bert-base-NER"
        """
        tokenizer = AutoTokenizer.from_pretrained(nlp_model)
        model = AutoModelForTokenClassification.from_pretrained(nlp_model)
        self.model = pipeline("ner", model=model, tokenizer=tokenizer)
        
    def get_presidio_analyzer_results(self) -> List:
        """
        Get the results of the Presidio BatchAnalyzer: assign entities to each record
        in each columns of the dataset.
        
        Returns
        -------
        List
            A list containing the results of the analyzer.
        """
        analyzer_results = list(self.presidio_analyzer.analyze_dict(self.dataset.to_dict(orient="list"), language="en"))
        return analyzer_results
    
    def assign_presidio_entities_list(self) -> None:
        """
        Get Presidio Analyzer results and assign entities to each object column of the dataset.
        """
        analyzer_results = self.get_presidio_analyzer_results()
        for col in analyzer_results:
            col_name = col.key
            if col_name in self.object_columns:
                # Get the list of entities for each record in the column
                entities_list = [single_value_type[0].entity_type for single_value_type in col.recognizer_results if len(single_value_type) > 0]
                # If the number of entities is more than 30% of the number of records, assign the list to the column
                if len(entities_list) > 0.3 * self.dataset.shape[0]:
                    self.dict_global_entities[col_name] = entities_list
                    self.assigned_entities_cols.append(col_name)
    
    def assign_location_entity(self) -> None:
        """
        Check whether the LOCATION entity is present among the entities assigned to the values in 
        each column with assigned entities.
        If the LOCATION entity has been assigned to at least 10 percent of the values in that column, 
        then the function assigns the LOCATION entity and the confidence score to that specific column
        """
        for col in self.assigned_entities_cols:
            entities_list = self.dict_global_entities[col]
            col_lower = col.lower() 
            location_freq = frequency(entities_list, "LOCATION")
            if ('LOCATION' in entities_list) and ('name' not in col_lower) and location_freq > 0.1:
                self.dict_global_entities[col] = {'entity': 'LOCATION', 'confidence_score': location_freq}
                
    def assign_entities_and_score(self) -> None:
        """
        Assign the most frequent entity and the confidence score to each object column 
        with assigned entities.
        """
        for col in self.assigned_entities_cols:
            entities_list = self.dict_global_entities[col]
            if isinstance(entities_list, list):
                # Get the most frequent entity
                most_freq = max(set(entities_list), key=entities_list.count)
                self.dict_global_entities[col] = {'entity': most_freq, 'confidence_score': frequency(entities_list, most_freq)}
            
    def assign_model_entities_list(self) -> None:
        """
        Assign entities to each object column which didn't get an entity from the Presidio Analyzer 
        using the NLP model.
        """
        for col in self.object_columns:
            if self.dict_global_entities[col] is None:
                self.model_entities[col] = self.model(self.dataset[col].tolist())
                self.model_entities[col] = [item["entity"] for sublist in self.model_entities[col] for item in sublist]
                
    def assign_organization_entity(self) -> None:
        """
        Check whether the B-ORG entity is present among the entities assigned to the values in each 
        column by the NLP model.
        
        If the B-ORG entity has been assigned to at least 10 percent of the values in that column, 
        then the function assigns the LOCATION entity and the confidence score to that specific column.
        """
        for col in self.model_entities:
            entities_list = self.model_entities[col]
            organization_freq = frequency(entities_list, "B-ORG")
            if ('B-ORG' in entities_list) and organization_freq > 0.1:
                self.dict_global_entities[col] = {'entity': 'ORGANIZATION', 'confidence_score': organization_freq}
                
    def assign_entities_manually(self, zipcode: Optional[bool] = True, credit_card: Optional[bool] = True) -> None:
        """
        Assign ZIPCODE and CREDIT_CARD_NUMBER entities to each column of the dataset.

        Parameters
        ----------
        zipcode : Optional[bool], optional
            Whether to look for zipcodes in the column name, by default True
        credit_card : Optional[bool], optional
            Whether to look for credit card numbers in the column name, by default True
        """
        for col in self.dict_global_entities:
            col_lower = col.lower()
            if zipcode and ((('postal' in col_lower) and ('code' in col_lower)) or ('zip' in col_lower)):
                self.dict_global_entities[col] = ['ZIPCODE', 1.0]
            if credit_card and ((('credit' in col_lower) or ('card' in col_lower)) and ('number' in col_lower)):
                self.dict_global_entities[col] = ['CREDIT_CARD_NUMBER', 1.0]
                
    def assign_entities_with_presidio(self) -> None:
        """
        Set Presidio Analyzer and assign entities with a confidence score
        to each object column of the dataset.
        """
        self.set_presidio_analyzer()
        self.assign_presidio_entities_list()
        self.assign_location_entity()
        self.assign_entities_and_score()
        
    def assign_organization_entity_with_model(self) -> None:
        """
        Set NLP model and assign entities with a confidence score to each 
        object column of the dataset.
        """
        self.set_model()
        self.assign_model_entities_list()
        self.assign_organization_entity()
            
        