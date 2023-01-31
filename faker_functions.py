from typing import Dict, List, Optional

import pandas as pd
from ner_functions import get_dict_entities_confidence_score
from faker import Faker


def get_columns_with_assigned_entity (dict_of_global_entities: Dict):

    if len(dict_of_global_entities) > 0:
        columns_with_assigned_entity = [[i, dict_of_global_entities[i]['entity']] for i in dict_of_global_entities if dict_of_global_entities[i] is not None and dict_of_global_entities[i]['confidence_score'] > 0.4]

        

    return columns_with_assigned_entity


def get_synthetic_address (df_input: pd.DataFrame, addresses: List):

    faker = Faker()
    for i in addresses:
        df_input[i] = [faker.street_address() for i in range(df_input.shape[0])]
    
    return df_input

def get_synthetic_phone_number (df_input: pd.DataFrame, phone_number: List):

    faker = Faker()
    for i in phone_number:
        df_input[i] = [faker.phone_number() for i in range(df_input.shape[0])]
    
    return df_input

def get_email_address (df_input: pd.DataFrame, email_address: List):
    faker = Faker()
    for i in email_address:
            df_input[i] = [faker.free_email() for i in range(df_input.shape[0])]
    
    return df_input

def get_first_name (df_input: pd.DataFrame, first_name_person: List):

    faker = Faker()
    for i in first_name_person:
        df_input[i] = [faker.first_name() for i in range(df_input.shape[0])]
    
    return df_input

def get_last_name (df_input: pd.DataFrame, last_name_person: List):

    faker = Faker()
    for i in last_name_person:
        df_input[i] = [faker.last_name() for i in range(df_input.shape[0])]
    
    return df_input

def get_person (df_input: pd.DataFrame, person: List):

    faker = Faker()
    for i in person:
            df_input[i] = [faker.name() for i in range(df_input.shape[0])]
    
    return df_input

def get_city (df_input: pd.DataFrame, city: List):

    faker = Faker()
    for i in city:
        df_input[i] = [faker.city() for i in range(df_input.shape[0])]
    
    return df_input

def get_city (df_input: pd.DataFrame, city: List):

    faker = Faker()
    for i in city:
        df_input[i] = [faker.city() for i in range(df_input.shape[0])]
    
    return df_input

def get_state(df_input: pd.DataFrame, state: List):
    faker = Faker()
    for i in state:
        if len(df_input[i].iloc[0]) == 2:
            df_input[i] = [faker.state_abbr() for i in range(df_input.shape[0])]
        else:
            df_input[i] = [faker.state_abbr() for i in range(df_input.shape[0])]
    
    return df_input

def get_state_full_name(df_input: pd.DataFrame, state: List):
    faker = Faker()
    for i in state:
        if len(df_input[i].iloc[0]) == 2:
            df_input[i] = [faker.state() for i in range(df_input.shape[0])]
    
    return df_input




def get_synthetic_dataset (df_input: pd.DataFrame, dict_of_global_entities: Dict):

    columns_with_assigned_entity = get_columns_with_assigned_entity(dict_of_global_entities)

    addresses = [i[0] for i in columns_with_assigned_entity if i[1] == 'ADDRESS']
    phone_number  = [i[0] for i in columns_with_assigned_entity if i[1] == 'PHONE_NUMBER']
    email_address = [i[0] for i in columns_with_assigned_entity if i[1] == 'EMAIL_ADDRESS']
    first_name_person = [i[0] for i in columns_with_assigned_entity if i[1] == 'PERSON' and (('first' in i[0].lower()) and ('name' in i[0].lower()))]
    last_name_person = [i[0] for i in columns_with_assigned_entity if i[1] == 'PERSON' and (('last' in i[0].lower()) and ('name' in i[0].lower()))]
    person = [i[0] for i in columns_with_assigned_entity if i[1] == 'PERSON']
    city = [i[0] for i in columns_with_assigned_entity if i[1] == 'LOCATION' and (('city' in i[0].lower()) or ('cities' in i[0].lower()))]
    state = [i[0] for i in columns_with_assigned_entity if i[1] == 'LOCATION' and ('state' in i[0].lower())]

    

    df_input = get_synthetic_address (df_input, addresses)
    df_input = get_synthetic_phone_number (df_input, phone_number)
    df_input = get_email_address (df_input, email_address)
    df_input = get_person(df_input, person)
    df_input = get_first_name(df_input, first_name_person)
    df_input = get_last_name(df_input, last_name_person)
    df_input = get_city(df_input, city)
    df_input = get_state(df_input, state)
    

    
    print ('Columns with faker data: \n')
    for i in columns_with_assigned_entity:
        print (i)

    return df_input