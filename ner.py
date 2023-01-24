from typing import Dict, List, Optional

from presidio_analyzer import AnalyzerEngine, BatchAnalyzerEngine, RecognizerRegistry, PatternRecognizer
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline


def ner(df_input_path: pd.DataFrame, data_sample: int = 500) -> Dict:
    """This function takes as input a dataframe and try to assign to each object column an entity.
    It returns a dictionary with columns' names as keys and a list as values.

    The list contains the entity associated to the column and a confidence score.
    The confidence score is the probability that a column is associated with the correct entity. 

    Parameters
    ----------
    df_input_path : pd.DataFrame
        It is a dataset
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
    #df_input_path: string containing path of csv file
    print('Starting...')

    df_input = pd.read_csv(df_input_path)
    df_input = df_input.sample(n=min(data_sample, df_input.shape[0]))

    # this for loop fillna in object column with zero

    for col in df_input.columns:
        if df_input[col].dtype == 'object':
            df_input[col] = df_input[col].fillna(0)

    #add some lists with rules to identify customed entity
    addresses = ['Street', 'Rue', 'Via', 'Square', 'Avenue', 'Place', 'Strada', 'St', 'Lane', 
    'Road', 'Boulevard', 'Ln', "Rd", "Highway" "Drive", "Av", "Hwy", "Blvd", "Corso", "Piazza", 
    "Calle", "Plaza", "Avenida", "Rambla", "C/"]
    addresses_recognizer = PatternRecognizer(supported_entity="ADDRESS", deny_list=addresses)


    analyzer = AnalyzerEngine(supported_languages=["en", "fr", "it", "es"])
    analyzer.registry.add_recognizer(addresses_recognizer)
    batch_analyzer = BatchAnalyzerEngine(analyzer_engine=analyzer)

    # Presidio was created as a tool to recognize entities in text, the following line is used to analyze a pandas dataframe
    # by first converting it to json
    analyzer_results = list(batch_analyzer.analyze_dict(df_input.to_dict(orient="list"), language="en"))

    # dict_entities is a dictionary where the columns' names are the keys and the entity and the confidence are the values
    dict_global_entities = {}
    list_of_df_cols = list(df_input.columns)

    for col in list_of_df_cols:
    #assigning ZIPCODE entity
        col_lower = col.lower()
        if (('postal' in col_lower) and ('code' in col_lower)) or ('zip' in col_lower):
            dict_global_entities[col] = ['ZIPCODE', 1.0]
    #assigning CREDIT_CARD_NUMBER entity
        elif (('credit' in col_lower) or ('card' in col_lower)) and ('number' in col_lower):
            dict_global_entities[col] = ['CREDIT_CARD_NUMBER', 1.0]
        else:
            dict_global_entities[col] = None
    
    # the following part creates a python dictionary (keys-> column names, values-> entity type) for columns of type object
    dict_type_object_entities = {}
    for col, res in zip(df_input.columns, analyzer_results):
        if df_input[col].dtype == 'object':
            dict_type_object_entities[col] = [single_value_type[0].entity_type for single_value_type in res.recognizer_results if len(single_value_type) > 0]
        
    cols = [col for col in dict_type_object_entities.keys() if len(dict_type_object_entities[col]) > 0.3 * df_input.shape[0]]
    
    # this is a very heuristic loop that assigns each object column with an entity
  
    for col in cols:
        list_object_entities = dict_type_object_entities[col]
        col_lower = col.lower()
        #assigning LOCATION entity
        if ('LOCATION' in list_object_entities) and ('name' not in col_lower) and (
                len([i for i in dict_type_object_entities[col] if i == 'LOCATION']) > 0.1 * df_input.shape[0]):
            dict_global_entities[col] = ['LOCATION', len([i for i in dict_type_object_entities[col] 
            if i == 'LOCATION'])/df_input.shape[0]]
        #assigning ZIPCODE entity if zipcode in dataset is 'object' type
        elif (('postal' in col_lower) and ('code' in col_lower)) or ('zip' in col_lower): 
            dict_global_entities[col] = ['ZIPCODE', 1.0]
        #assigning CREDIT_CARD_NUMBER entity if credit card number in dataset is 'object' type
        elif ((('credit' in col_lower) or ('card' in col_lower)) and ('number' in col_lower)):
            dict_global_entities[col] = ['CREDIT_CARD_NUMBER', 1.0]
        else:
            most_freq = max(set(list_object_entities), key=list_object_entities.count)
            dict_global_entities[col] = [most_freq, len([i for i in list_object_entities if i == most_freq])/df_input.shape[0]]

    # trying to assign the ORGANIZATION entity to keys/columns 
    # in dict_global_entities with None value
 

    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

    nlp_model = pipeline("ner", model=model, tokenizer=tokenizer)

    # this for loop change 0 value in object column in '/' value to be processed by the nlp model

    for col in df_input.columns:
        if df_input[col].dtype == 'object':
            df_input[col].replace(to_replace = 0, value = '/', inplace =True)

    # the following for loop returns a dict where keys are columns 
    # with no assigned entity and values are a list of the records of 
    # that column in dataset

    keyColumns_valueNone = {}

    for col in df_input.columns:
        if df_input[col].dtype == 'object' and dict_global_entities[col] == None:
            keyColumns_valueNone[col]= nlp_model(df_input[col].to_list())
    
    keyColumns_valueEntities = {}
    
    for col,big_list in keyColumns_valueNone.items():
        flat_entity_list = [entity_dict for sublist in big_list for entity_dict in sublist]
        keyColumns_valueEntities[col] = [entity['entity'] for entity in flat_entity_list]

    key_columns = [key for key in keyColumns_valueEntities]
    for col in key_columns:
        list_entities = keyColumns_valueEntities[col]
        if (('B-ORG' in list_entities)) and (
            len([i for i in keyColumns_valueEntities[col] if ((i == 'B-ORG'))]) > 0.1 * df_input.shape[0]):
            dict_global_entities[col] = ['ORGANIZATION', len([i for i in keyColumns_valueEntities[col] 
                    if (i == 'B-ORG')])/df_input.shape[0]]
                
    
    return dict_global_entities
