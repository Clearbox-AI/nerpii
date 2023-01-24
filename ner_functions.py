from typing import Dict, List, Optional

from presidio_analyzer import AnalyzerEngine, BatchAnalyzerEngine, PatternRecognizer
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline


def fill_na_on_object_columns_with_zero(df_input: pd.DataFrame) -> pd.DataFrame :
    """
    This function fills all the NA and/or Nan values on an object column with zero.
    It returns a copy of the original dataframe where NA and/or Nan values in object columns are filled with zero.

    Parameters
    ----------
    df_input : pd.DataFrame
        String containing path of csv file

    Returns
    -------
    pd.DataFrame
        Copy of the original dataframe where NA or Nan values in object columns are filled with zero.
    """
    df_fill_na_zero = df_input

    for col in df_fill_na_zero.columns:
        if df_fill_na_zero[col].dtype == 'object':
            df_fill_na_zero[col] = df_fill_na_zero[col].fillna(0)

    return df_fill_na_zero


def add_address_entity(additional_addresses: Optional[List] = []) -> PatternRecognizer:
    """
    This function allows to an user to create a customized presidio recognizer that 
    is able to recognize address entity.

    Some address-related words are already set, but an user can add ither new words. 

    Parameters
    ----------
    additional_addresses : Optional[List], optional
        It is a list in which an user can add new address-related words, by default []

    Returns
    -------
    PatternRecognizer
        It is the customized presidio recognizer
    """
    
    addresses = ['Street', 'Rue', 'Via', 'Square', 'Avenue', 'Place', 'Strada', 'St', 'Lane', 
    'Road', 'Boulevard', 'Ln', "Rd", "Highway" "Drive", "Av", "Hwy", "Blvd", "Corso", "Piazza", 
    "Calle", "Plaza", "Avenida", "Rambla", "C/"]
    addresses = addresses + additional_addresses
    addresses_recognizer = PatternRecognizer(supported_entity="ADDRESS", deny_list=addresses)

    return addresses_recognizer


def set_analyzer(addresses_recognizer: Optional[PatternRecognizer] = None) -> BatchAnalyzerEngine:

    analyzer = AnalyzerEngine()
    if addresses_recognizer:
        analyzer.registry.add_recognizer(addresses_recognizer)
    batch_analyzer = BatchAnalyzerEngine(analyzer_engine=analyzer)

    return batch_analyzer



def get_analyzer_results(df_input: pd.DataFrame, analyzer: BatchAnalyzerEngine) -> List:
    # Presidio was created as a tool to recognize entities in text, the following line is used to analyze a pandas dataframe
    # by first converting it to json

    analyzer_results = list(analyzer.analyze_dict(df_input.to_dict(orient="list"), language="en"))

    return analyzer_results

def assign_none_value(list_of_df_cols: List, dict_global_entities: Dict):
    for col in list_of_df_cols:
        dict_global_entities[col] = None
    
    return dict_global_entities


def assign_zipcode_entity(list_of_df_cols: List, dict_global_entities: Dict) -> Dict:

    for col in list_of_df_cols:
    #assigning ZIPCODE entity
        col_lower = col.lower()
        if (('postal' in col_lower) and ('code' in col_lower)) or ('zip' in col_lower):
            dict_global_entities[col] = ['ZIPCODE', 1.0]
    
    return dict_global_entities


def assign_credit_card_entity(list_of_df_cols: List, dict_global_entities: Dict) -> Dict:
    for col in list_of_df_cols:
        col_lower = col.lower()
    #assigning CREDIT_CARD_NUMBER entity
        if (('credit' in col_lower) or ('card' in col_lower)) and ('number' in col_lower):
            dict_global_entities[col] = ['CREDIT_CARD_NUMBER', 1.0]
    
    return dict_global_entities



def assign_entity_to_object_columns(df_input: pd.DataFrame, analyzer_results: List, dict_type_object_entities: Dict) -> Dict:
    for col, res in zip(df_input.columns, analyzer_results):
        if df_input[col].dtype == 'object':
            dict_type_object_entities[col] = [single_value_type[0].entity_type for single_value_type in res.recognizer_results if len(single_value_type) > 0]
    
    return dict_type_object_entities



def get_columns_with_assigned_entity(df_input: pd.DataFrame, dict_type_object_entities: Dict):
    cols = [col for col in dict_type_object_entities.keys() if len(dict_type_object_entities[col]) > 0.3 * df_input.shape[0]]

    return cols



def assign_location_entity(df_input: pd.DataFrame, dict_global_entities: Dict, dict_type_object_entities: Dict, cols: List) -> Dict:
    for col in cols:
        list_object_entities = dict_type_object_entities[col]
        col_lower = col.lower()
        #assigning LOCATION entity
        if ('LOCATION' in list_object_entities) and ('name' not in col_lower) and (
                len([i for i in dict_type_object_entities[col] if i == 'LOCATION']) > 0.1 * df_input.shape[0]):
            dict_global_entities[col] = ['LOCATION', len([i for i in dict_type_object_entities[col] 
            if i == 'LOCATION'])/df_input.shape[0]]
        
    
    return dict_global_entities




def assign_entities(df_input: pd.DataFrame, dict_global_entities: Dict, dict_type_object_entities: Dict, cols: List) -> Dict:
    for col in cols:
        list_object_entities = dict_type_object_entities[col]
        most_freq = max(set(list_object_entities), key=list_object_entities.count)
        dict_global_entities[col] = [most_freq, len([i for i in list_object_entities if i == most_freq])/df_input.shape[0]]

    return dict_global_entities


def fill_na_on_object_columns_with_slash(df_input: pd.DataFrame) -> pd.DataFrame :
    df_fill_na_slash = df_input

    for col in df_fill_na_slash.columns:
        if df_fill_na_slash[col].dtype == 'object':
            df_fill_na_slash[col] = df_fill_na_slash[col].fillna('/')

    return df_fill_na_slash

def set_nlp_model(model = "dslim/bert-base-NER") -> pipeline:

    tokenizer = AutoTokenizer.from_pretrained(model)
    model = AutoModelForTokenClassification.from_pretrained(model)
    nlp_model = pipeline("ner", model=model, tokenizer=tokenizer)

    return nlp_model

def apply_nlp_model(df_input: pd.DataFrame, keyColumns_valueNone: Dict, dict_global_entities: Dict, nlp_model: pipeline) -> Dict:
    for col in df_input.columns:
        if df_input[col].dtype == 'object' and dict_global_entities[col] == None:
            keyColumns_valueNone[col]= nlp_model(df_input[col].to_list())
    
    return keyColumns_valueNone

def get_entities_from_nlp_model_results(keyColumns_valueNone: Dict, keyColumns_valueEntities: Dict) -> Dict:
    for col, dict_value in keyColumns_valueNone.items():
        flat_entity_list = [entity_dict for sublist in dict_value for entity_dict in sublist]
        keyColumns_valueEntities[col] = [entity['entity'] for entity in flat_entity_list]
    
    return keyColumns_valueEntities

def assign_organization_entity(df_input: pd.DataFrame, keyColumns_valueEntities: Dict, dict_global_entities: Dict) -> Dict:
    key_columns = [key for key in keyColumns_valueEntities]
    for col in key_columns:
        list_entities = keyColumns_valueEntities[col]
        if (('B-ORG' in list_entities)) and (
            len([i for i in keyColumns_valueEntities[col] if ((i == 'B-ORG'))]) > 0.1 * df_input.shape[0]):
            dict_global_entities[col] = ['ORGANIZATION', len([i for i in keyColumns_valueEntities[col] 
                    if (i == 'B-ORG')])/df_input.shape[0]]
    
    return dict_global_entities


def ner_presidio(df_input_path: pd.DataFrame, data_sample: int = 500) -> Dict:
    """
    This function takes as input a dataframe and try to assign to each object column an entity.
    It returns a dictionary with columns' names as keys and a list as values.

    The list contains the entity associated to the column and a confidence score.
    The confidence score is the probability that a column is associated with the correct entity. 

    Parameters
    ----------
    df_input_path : pd.DataFrame
        String containing path of csv file
    data_sample : int, optional
        It is a sample of the previous dataset, by default 500. 
        The fuction takes a sample of the original datset to reduce computational costs.

    Returns
    -------
    Dict
        It returns a dictionary with columns' names as keys and a list as values.

    The list contains the entity associated to the column and a confidence score.
    The confidence score is the probability that a column is associated with the correct entity. 
    """

    df_input = pd.read_csv(df_input_path)
    df_input = df_input.sample(n=min(data_sample, df_input.shape[0]))

    df_input = fill_na_on_object_columns_with_zero (df_input) 

    addresses_recognizer = add_address_entity()
    analyzer = set_analyzer(addresses_recognizer)
    analyzer_results = get_analyzer_results(df_input, analyzer)

    dict_global_entities = {}
    list_of_df_cols = list(df_input.columns)

    dict_global_entities = assign_none_value(list_of_df_cols, dict_global_entities)
    dict_global_entities = assign_zipcode_entity(list_of_df_cols, dict_global_entities)
    dict_global_entities = assign_credit_card_entity(list_of_df_cols, dict_global_entities)

    dict_type_object_entities = {}

    dict_type_object_entities = assign_entity_to_object_columns(df_input, analyzer_results, dict_type_object_entities)

    cols = get_columns_with_assigned_entity(df_input, dict_type_object_entities)
    dict_global_entities = assign_location_entity(df_input, dict_global_entities, dict_type_object_entities, cols)
    dict_global_entities = assign_entities(df_input, dict_global_entities, dict_type_object_entities, cols)

    
    return dict_global_entities
    



def ner_organization_entity(df_input_path: pd.DataFrame, dict_global_entities: Dict, data_sample: int = 500) -> Dict:
    """
    This function associates organization entity to dataframe columns. 
    It uses a pretrained nlp model downloaded from Hugging Face (https://huggingface.co/dslim/bert-base-NER)
    to recognize organization entities.

    This function has to be applied after ner_presidio function.

    Parameters
    ----------
    df_input_path : pd.DataFrame
        String containing path of csv file
    dict_global_entities : Dict
        It is a dictionary where keys are the dataframe's columns and values are a list 
    containing the type of entity associated to a column and a confidence score.
    data_sample : int, optional
        This parameter sets the number of row to take to sample the original dataset, by default 500.
        it is necessary to reduce computational costs.

    Returns
    -------
    Dict
        It is a dictionary where keys are the dataframe's columns and values are a list 
    containing the type of entity associated to a column and a confidence score.
    """
    
    df_input = pd.read_csv(df_input_path)
    df_input = df_input.sample(n=min(data_sample, df_input.shape[0]))

    df_input = fill_na_on_object_columns_with_slash(df_input) 
    nlp_model = set_nlp_model()

    keyColumns_valueNone = {}

    keyColumns_valueNone = apply_nlp_model(df_input, keyColumns_valueNone, dict_global_entities, nlp_model)

    keyColumns_valueEntities = {}

    keyColumns_valueEntities = get_entities_from_nlp_model_results(keyColumns_valueNone, keyColumns_valueEntities)
    dict_global_entities = assign_organization_entity(df_input, keyColumns_valueEntities, dict_global_entities)

    return dict_global_entities


def get_dict_entities_confidence_score(df_input_path: pd.DataFrame) -> Dict:
    """
    This function takes as input a dataframe and 
    applies ner_presidio function and the ner_organization_entity function on the dataframe.

    It returns a dictionary where keys are the dataframe's columns and values are a dictionary 
    containing the type of entity associated to a column and a confidence score.

    The confidence score is the probability that a column is associated with the correct entity.

    Parameters
    ----------
    df_input_path : pd.DataFrame
        String containing path of csv file

    Returns
    -------
    Dict
        In this dictionary, keys have the same names of the dataframe columns and values are dictionaries in 
        which the entity associated to the column and its confidence score are reported.
    """
    dict_global_entities = ner_presidio(df_input_path)
    dict_global_entities = ner_organization_entity(df_input_path, dict_global_entities)

    for col in dict_global_entities:
        if dict_global_entities[col] is not None:
            dict_global_entities[col] = {'entity': dict_global_entities[col][0],'confidence_score' : dict_global_entities[col][1]}


    return dict_global_entities


    