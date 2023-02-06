from typing import Any, Dict, List, Optional

from presidio_analyzer import AnalyzerEngine, BatchAnalyzerEngine, PatternRecognizer
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline


def fill_na_on_object_columns_with_zero(df_input: pd.DataFrame) -> pd.DataFrame :
    """
    Return a copy of the original dataframe where NA and/or Nan values in object columns are filled with zero.

    Parameters
    ----------
    df_input : pd.DataFrame
        A pandas dataframe that needs to be analyzed

    Returns
    -------
    pd.DataFrame
        Copy of the original dataframe where NA or Nan values in object columns are filled with zero.
    """
    df_fill_na_zero = df_input.copy()

    for col in df_fill_na_zero.columns:
        if df_fill_na_zero[col].dtype == 'object':
            df_fill_na_zero[col] = df_fill_na_zero[col].fillna(0)

    return df_fill_na_zero


def add_address_entity(additional_addresses: Optional[List] = []) -> PatternRecognizer:
    """
    Return a customized presidio recognizer that 
    can recognize ADDRESS entity.

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


def set_analyzer(addresses_recognizer: Optional[PatternRecognizer] = None) -> BatchAnalyzerEngine:
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



def get_analyzer_results(df_input: pd.DataFrame, analyzer: BatchAnalyzerEngine) -> List:
    """
    Assign entities to each record in each columns in a pandas dataframe.
    An analyzer analyzes a pandas dataframe by first converting it to json.
    Return analyzer results
    
    Parameters
    ----------
    df_input : pd.DataFrame
        A pandas dataframe that needs to be analyzed
    analyzer : BatchAnalyzerEngine
        Analyzer defined previously by set-analyzer function

    Returns
    -------
    List
        A list that contains the results of the analyzer. 
    """
    analyzer_results = list(analyzer.analyze_dict(df_input.to_dict(orient="list"), language="en"))

    return analyzer_results

def assign_none_value(list_of_df_cols: List, dict_global_entities: Dict) -> Dict:
    """
    Return a dictionary whose keys are the columns of the dataframe and assigns
    each key the value None.

    Parameters
    ----------
    list_of_df_cols : List
        A list containing all column names
    dict_global_entities : Dict
        An empty dictionary 

    Returns
    -------
    Dict
        A dictionary whose keys are the columns of the dataframe and whose values are None
    """
    for col in list_of_df_cols:
        dict_global_entities[col] = None
    
    return dict_global_entities


def assign_zipcode_entity(list_of_df_cols: List, dict_global_entities: Dict) -> Dict:
    """
    Return a dictionary with ZIPCODE entity assigned.

    Look for some zipcode-related words in the column names and if it finds
    them it assigns the ZIPCODE entity to that column.

    Parameters
    ----------
     list_of_df_cols : List
        A list containing all column names
    dict_global_entities : Dict
        A dictionary whose keys are the column names and whose values are a list containing an assigned entity and the confidence score
    Returns
    -------
    Dict
        A dictionary in which a list containing ZIPCODE entity and the confidence score 1.0 is assigned as value to zipcode column(s).
    """

    for col in list_of_df_cols:
        col_lower = col.lower()
        if (('postal' in col_lower) and ('code' in col_lower)) or ('zip' in col_lower):
            dict_global_entities[col] = ['ZIPCODE', 1.0]
    
    return dict_global_entities


def assign_credit_card_entity(list_of_df_cols: List, dict_global_entities: Dict) -> Dict:
    """
    Return a dictionary with CREDIT_CARD_NUMBER entity assigned.

    Look for some creditcard-related words in the column names and if it finds
    them it assign the CREDIT_CARD_NUMBER entity to that column.

    Parameters
    ----------
    list_of_df_cols : List
        A list containing all column names
    dict_global_entities : Dict
        A dictionary whose keys are the column names and whose values are a list containing an assigned entity and the confidence score
    Returns
    -------
    Dict
        A dictionary in which a list containing CREDIT_CARD_NUMBER entity and the confidence score 1.0 is assigned as value to credit card number column(s).
    """
    for col in list_of_df_cols:
        col_lower = col.lower()
        if (('credit' in col_lower) or ('card' in col_lower)) and ('number' in col_lower):
            dict_global_entities[col] = ['CREDIT_CARD_NUMBER', 1.0]
    
    return dict_global_entities



def assign_entity_to_object_columns(df_input: pd.DataFrame, analyzer_results: List, dict_type_object_entities: Dict) -> Dict:
    """
    Assign each object column in the pandas dataframe the respective entity assigned by the analyzer.

    Parameters
    ----------
    df_input : pd.DataFrame
        A pandas dataframe that needs to be analyzed
    analyzer_results : List
        A list that contains the analyzer results
    dict_type_object_entities : Dict
        A dictionary whose keys are the column names and whose values are a list containing an assigned entity and the confidence score
    Returns
    -------
    Dict
        A dictionary whose keys are the object column names and whose values are a list of all the entities assigned to each value in that specific column
    """
    for col, res in zip(df_input.columns, analyzer_results):
        if df_input[col].dtype == 'object':
            dict_type_object_entities[col] = [single_value_type[0].entity_type for single_value_type in res.recognizer_results if len(single_value_type) > 0]
    
    return dict_type_object_entities



def get_columns_with_assigned_entity(df_input: pd.DataFrame, dict_type_object_entities: Dict) -> List:
    """
    Return a list of those columns for which the analyzer was able to 
    assign an entity to at least 30 percent of the values of that specific column.

    Parameters
    ----------
    df_input : pd.DataFrame
        A pandas dataframe that needs to be analyzed
    dict_type_object_entities : Dict
        A dictionary whose keys are the column names and whose values are an assigned entity
        

    Returns
    -------
    List
        A list of those columns for which the analyzer was able to 
    assign an entity to at least 30 percent of the values of that specific column.
    """
    cols = [col for col in dict_type_object_entities.keys() if len(dict_type_object_entities[col]) > 0.3 * df_input.shape[0]]

    return cols



def assign_location_entity(df_input: pd.DataFrame, dict_global_entities: Dict, dict_type_object_entities: Dict, cols: List) -> Dict:
    """ 
    Check whether the LOCATION entity is present among 
    the entities assigned to the values in each column.

    If the LOCATION entity has been assigned to at least 10 percent of the values in that column, 
    then the function assigns the LOCATION entity and the confidence score to that specific column. 

    Parameters
    ----------
    df_input : pd.DataFrame
        A pandas dataframe that needs to be analyzed
    dict_global_entities : Dict
        A dictionary whose keys are the column names and whose values are an assigned entity
    dict_type_object_entities : Dict
        A dictionary whose keys are the object column names and whose values are a list of all the entities assigned to each value in that specific column
    cols : List
        A list of those columns for which the analyzer was able to assign an entity to at least 30 percent of the values of that specific column.
        

    Returns
    -------
    Dict
        A dictionary whose keys are the column names and whose values are a list containing an assigned entity and the confidence score
    """
    for col in cols:
        list_object_entities = dict_type_object_entities[col]
        col_lower = col.lower()
        if ('LOCATION' in list_object_entities) and ('name' not in col_lower) and (
                len([i for i in dict_type_object_entities[col] if i == 'LOCATION']) > 0.1 * df_input.shape[0]):
            dict_global_entities[col] = ['LOCATION', len([i for i in dict_type_object_entities[col] 
            if i == 'LOCATION'])/df_input.shape[0]]
        
    
    return dict_global_entities




def assign_entities(df_input: pd.DataFrame, dict_global_entities: Dict, dict_type_object_entities: Dict, cols: List) -> Dict:
    """
    Assign the most frequent entity assigned by the analyzer and 
    the confidence score to the object columns.

    Parameters
    ----------
    df_input : pd.DataFrame
        A pandas dataframe that needs to be analyzed
    dict_global_entities : Dict
        A dictionary whose keys are the column names and whose values are an assigned entity
    dict_type_object_entities : Dict
        A dictionary whose keys are the object column names and whose values are a list of all the entities assigned to each value in that specific column
    cols : List
        A list of those columns for which the analyzer was able to assign an entity to at least 30 percent of the values of that specific column.

    Returns
    -------
    Dict
        A dictionary whose keys are the column names and whose values are a list containing an assigned entity and the confidence score
    """

    for col in cols:
        list_object_entities = dict_type_object_entities[col]
        most_freq = max(set(list_object_entities), key=list_object_entities.count)
        dict_global_entities[col] = [most_freq, len([i for i in list_object_entities if i == most_freq])/df_input.shape[0]]

    return dict_global_entities


def fill_na_on_object_columns_with_slash(df_input: pd.DataFrame) -> pd.DataFrame :
    """
    Return a copy of the original dataframe where NA and/or Nan values in object columns are filled with a string containing a slash ('/').

    Parameters
    ----------
    df_input : pd.DataFrame
        A pandas dataframe that needs to be analyzed

    Returns
    -------
    pd.DataFrame
        A copy of the original sampled dataframe where NA or Nan values in object columns are filled with a string containing a slash ('/').
    """
    df_fill_na_slash = df_input.copy()

    for col in df_fill_na_slash.columns:
        if df_fill_na_slash[col].dtype == 'object':
            df_fill_na_slash[col] = df_fill_na_slash[col].fillna('/')

    return df_fill_na_slash

def set_nlp_model(model = "dslim/bert-base-NER") -> Any:
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

def apply_nlp_model(df_input: pd.DataFrame, keyColumns_valueNone: Dict, dict_global_entities: Dict, nlp_model: pipeline) -> Dict:
    """
    Apply the nlp_model on those object columns in the pandas dataframe which didn't get an entity from the ner_presidio function.

    Parameters
    ----------
    df_input : pd.DataFrame
        A pandas dataframe that needs to be analyzed
    keyColumns_valueNone : Dict
        An empty dictionary
    nlp_model : pipeline
        IA pretrained nlp model

    Returns
    -------
    Dict
        A dictionary whose keys are those object columns without an assigned entity and
        whose values are a list of lists containing dictionaries with the entities assigned to each value in that specific column
    """
    for col in df_input.columns:
        if df_input[col].dtype == 'object' and dict_global_entities[col] == None:
            keyColumns_valueNone[col]= nlp_model(df_input[col].to_list())
    
    return keyColumns_valueNone

def get_entities_from_nlp_model_results(keyColumns_valueNone: Dict, keyColumns_valueEntities: Dict) -> Dict:
    """
    Return a dictionary whose keys are object column names and
    whose values are a list containing all the entities assigned to each value in that specific column

    Parameters
    ----------
    keyColumns_valueNone : Dict
        A dictionary whose keys are those object columns without an assigned entity and
        whose values are a list of lists containing dictionaries with the entities assigned to each value in that specific column 
    keyColumns_valueEntities : Dict
        An empty dictionary

    Returns
    -------
    Dict
        A dictionary whose keys are object column names and
        whose values are a list containing all the entities assigned to each value in that specific column
    """
    for col, dict_value in keyColumns_valueNone.items():
        flat_entity_list = [entity_dict for sublist in dict_value for entity_dict in sublist]
        keyColumns_valueEntities[col] = [entity['entity'] for entity in flat_entity_list]
    
    return keyColumns_valueEntities

def assign_organization_entity(df_input: pd.DataFrame, keyColumns_valueEntities: Dict, dict_global_entities: Dict) -> Dict:
    """
    Check whether the B-ORG entity is present among 
    the entities assigned to the values in each column.

    If the B-ORG entity has been assigned to at least 10 percent of the values in that column, 
    then the function assigns the LOCATION entity and the confidence score to that specific column. 

    Parameters
    ----------
    df_input : pd.DataFrame
        A pandas dataframe that needs to be analyzed
    keyColumns_valueEntities : Dict
        A dictionary whose keys are object column names and
        whose values are a list containing all the entities assigned to each value in that specific column
    dict_global_entities : Dict
        A dictionary whose keys are the column names and whose values are a list containing an assigned entity and the confidence score
        
    Returns
    -------
    Dict
        A dictionary whose keys are the column names and whose values are a list containing an assigned entity and the confidence score
    """
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
    Take as input a dataframe and try to assign to each object column an entity.
    Return a dictionary whose keys are column names and whose values are a list.

    The list contains the entity assigned to the column and a confidence score.
    The confidence score is the probability that a column is associated with the correct entity. 

    Parameters
    ----------
    df_input_path : pd.DataFrame
        String containing path of csv file
    data_sample : int, optional
        It is a sample of the previous dataset, by default 500. 
        The fuction takes a sample of the original dataset to reduce computational costs.

    Returns
    -------
    Dict
        A dictionary whose keys are the column names and whose values are a list containing an assigned entity and the confidence score
        
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
    Associate organization entity to dataframe columns. 
    Use a pretrained nlp model downloaded from Hugging Face (https://huggingface.co/dslim/bert-base-NER)
    to recognize organization entities.

    This function has to be applied after ner_presidio function.

    Parameters
    ----------
    df_input_path : pd.DataFrame
        String containing path of csv file
    dict_global_entities : Dict
        A dictionary where keys are the dataframe's columns and values are a list 
    containing the type of entity associated to a column and a confidence score.
    data_sample : int, optional
        Number of row to take to sample the original dataset, by default 500.
        

    Returns
    -------
    Dict
        A dictionary whose keys are the column names and 
        whose values are a list containing an assigned entity and the confidence score
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


    
    
def get_entities_confidence_score(df_input_path: pd.DataFrame) -> Dict:
    """
    Take as input a dataframe and 
    applies ner_presidio function and the ner_organization_entity function on the dataframe.

    Return a dictionary whose keys are the columns of the pandas dataframe and whose values are a dictionary 
    containing the type of entity associated to a column and a confidence score.

    The confidence score is the probability that a column is associated with the correct entity.

    Parameters
    ----------
    df_input_path : pd.DataFrame
        String containing path of csv file

    Returns
    -------
    Dict
        A dictionary whose keys have the same names of the dataframe columns and values are dictionaries in 
        which the entity associated to the column and its confidence score are reported.
    """
    dict_global_entities = ner_presidio(df_input_path)
    dict_global_entities = ner_organization_entity(df_input_path, dict_global_entities)

    for col in dict_global_entities:
        if dict_global_entities[col] is not None:
            dict_global_entities[col] = {'entity': dict_global_entities[col][0],'confidence_score' : dict_global_entities[col][1]}


    return dict_global_entities


    