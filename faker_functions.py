from typing import Dict, List, Optional

import pandas as pd
from ner_functions import get_entities_confidence_score
from faker import Faker
import gender_guesser.detector as gender
from simple_colors import *




def get_columns_with_assigned_entity (dict_of_global_entities: Dict) -> List:

    if len(dict_of_global_entities) > 0:
        columns_with_assigned_entity = [[i, dict_of_global_entities[i]['entity']] for i in dict_of_global_entities if dict_of_global_entities[i] is not None and dict_of_global_entities[i]['confidence_score'] > 0.4]

        

    return columns_with_assigned_entity


def synthesis_message (list: List) -> str:
    for col in list:
        message = 'Column ' + red(col, 'bold') + ' synthesized.'
        print(message)

    return 




def get_address (df_input: pd.DataFrame, addresses: List) -> pd.DataFrame:

    faker = Faker()
    for i in addresses:
        df_input[i] =  df_input.apply(lambda row: faker.street_address(), axis = 1)
    
    if len(addresses) > 0:
        synthesis_message(addresses)

    return df_input

def get_phone_number (df_input: pd.DataFrame, phone_number: List) -> pd.DataFrame:

    faker = Faker()
    for i in phone_number:
        df_input[i] =  df_input.apply(lambda row: faker.phone_number(), axis = 1)
    
    if len(phone_number) > 0:
        synthesis_message(phone_number)
    
    return df_input

def get_email_address (df_input: pd.DataFrame, email_address: List) -> pd.DataFrame:
    faker = Faker()
    for i in email_address:
        df_input[i] =  df_input.apply(lambda row: faker.free_email(), axis = 1)
    
    if len(email_address) > 0:
        synthesis_message(email_address)
    
    return df_input


def get_gender(df_input: pd.DataFrame, first_name_person: List) -> pd.DataFrame:

    detector = gender.Detector(case_sensitive=False)
    first_name_gender = []

    for col in first_name_person:
        for name in df_input[col]:
            first_name_gender.append(detector.get_gender(name))
    
    df_input['first_name_gender'] = pd.Series(first_name_gender)

    
    return df_input


def get_first_name (df_input: pd.DataFrame, first_name_person: List) -> pd.DataFrame:

    faker = Faker()

    for col in first_name_person:
        df_input[col] = df_input.apply(lambda row: faker.first_name_female() if (row['first_name_gender'] == 'female' or row['first_name_gender'] == 'mostly_female') else row[col], axis=1)
        df_input[col] = df_input.apply(lambda row: faker.first_name_male() if (row['first_name_gender'] == 'male' or row['first_name_gender'] == 'mostly_male') else row[col], axis=1)
        df_input[col] = df_input.apply(lambda row: faker.first_name() if (row['first_name_gender'] == 'unknown' or row['first_name_gender'] == 'andy') else row[col], axis=1)

    if len(first_name_person) > 0:
        synthesis_message(first_name_person)

    return df_input

def get_last_name (df_input: pd.DataFrame, last_name_person: List) -> pd.DataFrame:

    faker = Faker()

    if len(last_name_person) > 0:
        for i in last_name_person:
            df_input[i] =  df_input.apply(lambda row: faker.last_name(), axis = 1)
    
    else:
        last_name_person = [i for i in df_input.columns if (('last' in i.lower()) and ('name' in i.lower()))]
        for i in last_name_person:
            df_input[i] =  df_input.apply(lambda row: faker.last_name(), axis = 1)


    if len(last_name_person) > 0:
        synthesis_message(last_name_person)
    
    return df_input

def get_person (df_input: pd.DataFrame, person: List) -> pd.DataFrame:

    faker = Faker()
    for i in person:
        df_input[i] =  df_input.apply(lambda row: faker.name(), axis = 1)
        
    if len(person) > 0:
        synthesis_message(person)

    return df_input

def get_city (df_input: pd.DataFrame, city: List) -> pd.DataFrame:

    faker = Faker()
    for i in city:
        df_input[i] =  df_input.apply(lambda row: faker.city(), axis = 1)
    
    if len(city) > 0:
        synthesis_message(city)
    
    return df_input


def get_state(df_input: pd.DataFrame, state: List) -> pd.DataFrame:
    faker = Faker()
    for i in state:
        if len(df_input[i].iloc[0]) == 2:
            df_input[i] =  df_input.apply(lambda row: faker.state_abbr(), axis = 1)
        else:
            df_input[i] =  df_input.apply(lambda row: faker.state(), axis = 1)
    
    if len(state) > 0:
        synthesis_message(state)
    
    return df_input

def get_url (df_input: pd.DataFrame, url: List) -> pd.DataFrame:
    faker = Faker()
    for i in url:
        df_input[i] =  df_input.apply(lambda row: faker.url(), axis = 1)
    
    if len(url) > 0:
        synthesis_message(url)
    
    return df_input


    




def get_synthetic_dataset (df_input: pd.DataFrame, dict_of_global_entities: Dict)  -> pd.DataFrame:

    columns_with_assigned_entity = get_columns_with_assigned_entity(dict_of_global_entities)
    

    addresses = [i[0] for i in columns_with_assigned_entity if i[1] == 'ADDRESS']
    phone_number  = [i[0] for i in columns_with_assigned_entity if i[1] == 'PHONE_NUMBER']
    email_address = [i[0] for i in columns_with_assigned_entity if i[1] == 'EMAIL_ADDRESS']
    first_name_person = [i[0] for i in columns_with_assigned_entity if i[1] == 'PERSON' and (('first' in i[0].lower()) and ('name' in i[0].lower()))]
    last_name_person = [i[0] for i in columns_with_assigned_entity if i[1] == 'PERSON' and (('last' in i[0].lower()) and ('name' in i[0].lower()))]
    #person = [i[0] for i in columns_with_assigned_entity if i[1] == 'PERSON']
    city = [i[0] for i in columns_with_assigned_entity if i[1] == 'LOCATION' and (('city' in i[0].lower()) or ('cities' in i[0].lower()))]
    state = [i[0] for i in columns_with_assigned_entity if i[1] == 'LOCATION' and ('state' in i[0].lower())]
    url = [i[0] for i in columns_with_assigned_entity if i[1] == 'URL']

    
    

    df_input = get_address (df_input, addresses)
    df_input = get_phone_number (df_input, phone_number)
    df_input = get_email_address (df_input, email_address)
    df_input = get_gender(df_input, first_name_person)
    #df_input = get_person(df_input, person)
    df_input = get_first_name(df_input, first_name_person)
    df_input = get_last_name(df_input, last_name_person)
    df_input = get_city(df_input, city)
    df_input = get_state(df_input, state)
    df_input = get_url(df_input, url)
    



    return df_input