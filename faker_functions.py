from typing import Dict, List, Optional

import pandas as pd
from ner_functions import get_entities_confidence_score
from faker import Faker
import gender_guesser.detector as gender
from simple_colors import *
import re
import numpy as np




def get_columns_with_assigned_entity (dict_of_global_entities: Dict) -> List:
    """
    Create a list containing those columns with an assigned entity and confidence score > 0.3.
    If this list is not empty, return list, else return a message

    Parameters
    ----------
    dict_of_global_entities : Dict
        A dictionary whose keys are the column names and whose values are an assigned entity

    Returns
    -------
    List
        A list containing those columns with an assigned entity and confidence score > 0.3
    """

    if len(dict_of_global_entities) > 0:
        columns_with_assigned_entity = [[i, dict_of_global_entities[i]['entity']] for i in dict_of_global_entities if dict_of_global_entities[i] is not None and dict_of_global_entities[i]['confidence_score'] > 0.3]

    if len(columns_with_assigned_entity) > 0:
        return columns_with_assigned_entity
    else:
        return print ('Impossible to generate Faker data: no assigned entities.')



def get_columns_with_low_cs (dict_of_global_entities: Dict) -> List:
    """
    Return a list containing those columns with confidence score <= 0.3.
    These columns won't be synthesized.

    Parameters
    ----------
    dict_of_global_entities : Dict
        A dictionary whose keys are the column names and whose values are an assigned entity

    Returns
    -------
    List
        A list containing those columns with an assigned entity, but confidence score <= 3
    """

    if len(dict_of_global_entities) > 0:
        columns_not_synthesized= [[i, dict_of_global_entities[i]['entity']] for i in dict_of_global_entities if dict_of_global_entities[i] is not None and dict_of_global_entities[i]['confidence_score'] <= 0.3 and not re.match('.*?last.*?name.*?', i.lower())]

    
    return columns_not_synthesized


def get_columns_not_synthesized (columns_with_assigned_entity: List, list_faker: List, columns_not_synthesized: List) -> List:
    """
    Return a list of all non-synthesized columns.

    Parameters
    ----------
    columns_with_assigned_entity : List
        A list containing those columns with an assigned entity and confidence score > 0.3
    list_faker : List
        A list containing the synthesized columns
    columns_not_synthesized : List
        A list containing those columns with an assigned entity, but confidence score <= 3

    Returns
    -------
    List
        A list containing non-synthesized columns
    """

    for i in columns_with_assigned_entity:
            if i[0] not in list_faker:
                columns_not_synthesized.append(i)
    

    return columns_not_synthesized
     
    
def synthesis_message (list_faker: List, list_not_faker:List) -> str:
    """
    Return a message with synthesized and unsynthesized columns.

    Parameters
    ----------
    list_faker : List
        A list containing the synthesized columns
    list_not_faker : List
        A list containing the unsynthesized columns

    Returns
    -------
    str
        message
    """
    for col in list_faker:
        message = 'Column ' + red(col, 'bold') + ' synthesized.'
        print(message)
    
    for col in list_not_faker:
        message = 'Column ' + green(col[0], 'bold') + ' not synthesized.'
        print(message)

    return 




def get_address (df_input: pd.DataFrame, addresses: List, list_faker: List) -> pd.DataFrame:
    """
    Return pandas dataframe where address columns are synthesized.

    Parameters
    ----------
    df_input : pd.DataFrame
        A pandas dataframe that needs to be analyzed
    addresses : List
        A list of the columns to which the entity 'ADDRESS' is assigned
    list_faker : List
        A list containing the synthesized columns

    Returns
    -------
    pd.DataFrame
        A pandas dataframe with synthesized address columns 
    list_faker
        A list containing the synthesized columns
    """

    faker = Faker()
    for i in addresses:
        df_input[i] =  df_input.apply(lambda row: faker.street_address(), axis = 1)
        
        list_faker.append(i)

    return df_input, list_faker

def get_phone_number (df_input: pd.DataFrame, phone_number: List, list_faker: List) -> pd.DataFrame:
    """
    Return pandas dataframe where phone_number columns are synthesized.

    Parameters
    ----------
    df_input : pd.DataFrame
        A pandas dataframe that needs to be analyzed
    phone_number : List
        A list of the columns to which the entity 'PHONE NUMBER' is assigned
    list_faker : List
        A list containing the synthesized columns

    Returns
    -------
    pd.DataFrame
        A pandas dataframe with synthesized phone_number columns 
    list_faker
        A list containing the synthesized columns
    """

    faker = Faker()
    for i in phone_number:
        df_input[i] =  df_input.apply(lambda row: faker.phone_number(), axis = 1)
        
        list_faker.append(i)
    
    return df_input, list_faker



def get_email_address (df_input: pd.DataFrame, email_address: List, list_faker: List) -> pd.DataFrame:
    """
    Return pandas dataframe where email_address columns are synthesized.

    Parameters
    ----------
    df_input : pd.DataFrame
        A pandas dataframe that needs to be analyzed
    email_address : List
        A list of the columns to which the entity 'EMAIL ADDRESS' is assigned
    list_faker : List
        A list containing the synthesized columns

    Returns
    -------
    pd.DataFrame
        A pandas dataframe with synthesized email_address columns 
    list_faker
        A list containing the synthesized columns
    """    
    faker = Faker()
    for i in email_address:
        df_input[i] =  df_input.apply(lambda row: faker.free_email(), axis = 1)
    
        list_faker.append(i)
    
    return df_input, list_faker


def get_gender(df_input: pd.DataFrame, first_name_person: List) -> pd.DataFrame:

    detector = gender.Detector(case_sensitive=False)
    first_name_gender = []

    if len(first_name_person) > 0:

        for col in first_name_person:
            for name in df_input[col]:
                if name is not np.NaN:
                    first_name_gender.append(detector.get_gender(name))
                else:
                    first_name_gender.append('Nan value')
        
        df_input['first_name_gender'] = pd.Series(first_name_gender)

    
    return df_input


def get_first_name (df_input: pd.DataFrame, first_name_person: List, list_faker: List) -> pd.DataFrame:
    """
    Return pandas dataframe where first_name_person columns are synthesized.

    Parameters
    ----------
    df_input : pd.DataFrame
        A pandas dataframe that needs to be analyzed
    first_name_person : List
        A list of columns to which the entity "PERSON" is assigned and whose column name contains "first name"
    list_faker : List
        A list containing the synthesized columns

    Returns
    -------
    pd.DataFrame
        A pandas dataframe with synthesized first_name_person columns 
    list_faker
        A list containing the synthesized columns
    """ 

    faker = Faker()

    for i in first_name_person:
        df_input[i] = df_input.apply(lambda row: faker.first_name_female() if (row['first_name_gender'] == 'female' or row['first_name_gender'] == 'mostly_female') else row[i], axis=1)
        df_input[i] = df_input.apply(lambda row: faker.first_name_male() if (row['first_name_gender'] == 'male' or row['first_name_gender'] == 'mostly_male') else row[i], axis=1)
        df_input[i] = df_input.apply(lambda row: faker.first_name() if (row['first_name_gender'] == 'unknown' or row['first_name_gender'] == 'andy') else row[i], axis=1)

        list_faker.append(i)

    return df_input, list_faker

def get_last_name (df_input: pd.DataFrame, last_name_person: List, list_faker: List) -> pd.DataFrame:
    """
    Return pandas dataframe where last_name_person columns are synthesized.

    Parameters
    ----------
    df_input : pd.DataFrame
        A pandas dataframe that needs to be analyzed
    last_name_person : List
        A list of columns to which the entity "PERSON" is assigned and whose column name contains "last name"
    list_faker : List
        A list containing the synthesized columns

    Returns
    -------
    pd.DataFrame
        A pandas dataframe with synthesized last_name_person columns 
    list_faker
        A list containing the synthesized columns
    """

    faker = Faker()

    if len(last_name_person) > 0:
        for i in last_name_person:
            df_input[i] =  df_input.apply(lambda row: faker.last_name(), axis = 1)

            list_faker.append(i)
    
    else:
        last_name_person = [i for i in df_input.columns if (('last' in i.lower()) and ('name' in i.lower()))]
        for i in last_name_person:
            df_input[i] =  df_input.apply(lambda row: faker.last_name(), axis = 1)

            list_faker.append(i)

    return df_input, list_faker


def get_city (df_input: pd.DataFrame, city: List, list_faker: List) -> pd.DataFrame:
    """
    Return pandas dataframe where city columns are synthesized.

    Parameters
    ----------
    df_input : pd.DataFrame
        A pandas dataframe that needs to be analyzed
    city : List
        A list of columns to which the entity "LOCATION" is assigned and whose column name contains "city"
    list_faker : List
        A list containing the synthesized columns

    Returns
    -------
    pd.DataFrame
        A pandas dataframe with synthesized city columns 
    list_faker
        A list containing the synthesized columns
    """


    faker = Faker()
    for i in city:
        df_input[i] =  df_input.apply(lambda row: faker.city(), axis = 1)
    
        list_faker.append(i)

    return df_input, list_faker


def get_state(df_input: pd.DataFrame, state: List, list_faker: List) -> pd.DataFrame:
    """
    Return pandas dataframe where state columns are synthesized.

    Parameters
    ----------
    df_input : pd.DataFrame
        A pandas dataframe that needs to be analyzed
    state : List
        A list of columns to which the entity "LOCATION" is assigned and whose column name contains "state"
    list_faker : List
        A list containing the synthesized columns

    Returns
    -------
    pd.DataFrame
        A pandas dataframe with synthesized state columns 
    list_faker
        A list containing the synthesized columns
    """
    faker = Faker()
    for i in state:
        if len(df_input[i].iloc[0]) == 2:
            df_input[i] =  df_input.apply(lambda row: faker.state_abbr(), axis = 1)
        else:
            df_input[i] =  df_input.apply(lambda row: faker.state(), axis = 1)
    
        list_faker.append(i)

    return df_input, list_faker

def get_url (df_input: pd.DataFrame, url: List, list_faker: List) -> pd.DataFrame:
    """
    Return pandas dataframe where url columns are synthesized.

    Parameters
    ----------
    df_input : pd.DataFrame
        A pandas dataframe that needs to be analyzed
    url : List
        A list of columns to which the entity "URL" is assigned
    list_faker : List
        A list containing the synthesized columns

    Returns
    -------
    pd.DataFrame
        A pandas dataframe with synthesized url columns 
    list_faker
        A list containing the synthesized columns
    """
    faker = Faker()
    for i in url:
        df_input[i] =  df_input.apply(lambda row: faker.url(), axis = 1)
    
        list_faker.append(i)

    return df_input, list_faker

def get_zipcode (df_input: pd.DataFrame, zipcode: List, list_faker: List) -> pd.DataFrame:
    """
    Return pandas dataframe where zipcode columns are synthesized.

    Parameters
    ----------
    df_input : pd.DataFrame
        A pandas dataframe that needs to be analyzed
    zipcode : List
        A list of columns to which the entity "ZIPCODE" is assigned
    list_faker : List
        A list containing the synthesized columns

    Returns
    -------
    pd.DataFrame
        A pandas dataframe with synthesized zipcode columns 
    list_faker
        A list containing the synthesized columns
    """

    faker = Faker()
    for i in zipcode:
        df_input[i] =  df_input.apply(lambda row: faker.zipcode() , axis = 1)
        
        list_faker.append(i)

    return df_input, list_faker

def get_credit_card (df_input: pd.DataFrame, credit_card: List, list_faker: List) -> pd.DataFrame:
    """
    Return pandas dataframe where credit_card columns are synthesized.

    Parameters
    ----------
    df_input : pd.DataFrame
        A pandas dataframe that needs to be analyzed
    credit_card : List
        A list of columns to which the entity "CREDIT_CARD_NUMBER" is assigned
    list_faker : List
        A list containing the synthesized columns

    Returns
    -------
    pd.DataFrame
        A pandas dataframe with synthesized credit_card columns 
    list_faker
        A list containing the synthesized columns
    """

    faker = Faker()
    for i in credit_card:
        df_input[i] =  df_input.apply(lambda row: faker.credit_card_number() , axis = 1)
        
        list_faker.append(i)

    return df_input, list_faker

def get_ssn (df_input: pd.DataFrame, ssn: List, list_faker: List) -> pd.DataFrame:
    """
    Return pandas dataframe where ssn columns are synthesized.

    Parameters
    ----------
    df_input : pd.DataFrame
        A pandas dataframe that needs to be analyzed
    ssn : List
        A list of columns to which the entity "US_SSN" is assigned
    list_faker : List
        A list containing the synthesized columns

    Returns
    -------
    pd.DataFrame
        A pandas dataframe with synthesized ssn columns 
    list_faker
        A list containing the synthesized columns
    """

    faker = Faker()
    for i in ssn:
        df_input[i] =  df_input.apply(lambda row: faker.ssn() , axis = 1)
        
        list_faker.append(i)

    return df_input, list_faker


def get_country (df_input: pd.DataFrame, country: List, list_faker: List) -> pd.DataFrame:
    """
    Return pandas dataframe where country columns are synthesized.

    Parameters
    ----------
    df_input : pd.DataFrame
        A pandas dataframe that needs to be analyzed
    country : List
        A list of columns to which the entity "LOCATION" is assigned and whose column name contains "country"
    list_faker : List
        A list containing the synthesized columns

    Returns
    -------
    pd.DataFrame
        A pandas dataframe with synthesized country columns 
    list_faker
        A list containing the synthesized columns
    """

    faker = Faker()
    for i in country:
        df_input[i] =  df_input.apply(lambda row: faker.country() , axis = 1)
        
        list_faker.append(i)

    return df_input, list_faker



    




def get_synthetic_dataset (df_input: pd.DataFrame, dict_of_global_entities: Dict)  -> pd.DataFrame:
    """
    Return a synthesized dataframe

    Parameters
    ----------
    df_input : pd.DataFrame
        A pandas dataframe
    dict_of_global_entities : Dict
        A dictionary whose keys are the column names and whose values are an assigned entity
        

    Returns
    -------
    pd.DataFrame
        A synthesized pandas dataframe
    """

    columns_with_assigned_entity = get_columns_with_assigned_entity(dict_of_global_entities)
    columns_with_low_cs = get_columns_with_low_cs(dict_of_global_entities)
    

    addresses = [i[0] for i in columns_with_assigned_entity if i[1] == 'ADDRESS']
    phone_number  = [i[0] for i in columns_with_assigned_entity if i[1] == 'PHONE_NUMBER']
    email_address = [i[0] for i in columns_with_assigned_entity if i[1] == 'EMAIL_ADDRESS']
    first_name_person = [i[0] for i in columns_with_assigned_entity if i[1] == 'PERSON' and (('first' in i[0].lower()) and ('name' in i[0].lower()))]
    last_name_person = [i[0] for i in columns_with_assigned_entity if i[1] == 'PERSON' and (('last' in i[0].lower()) and ('name' in i[0].lower()))]
    city = [i[0] for i in columns_with_assigned_entity if i[1] == 'LOCATION' and (('city' in i[0].lower()) or ('cities' in i[0].lower()))]
    state = [i[0] for i in columns_with_assigned_entity if i[1] == 'LOCATION' and ('state' in i[0].lower())]
    url = [i[0] for i in columns_with_assigned_entity if i[1] == 'URL']
    zipcode = [i[0] for i in columns_with_assigned_entity if i[1] == 'ZIPCODE']
    credit_card = [i[0] for i in columns_with_assigned_entity if i[1] == 'CREDIT_CARD_NUMBER']
    ssn = [i[0] for i in columns_with_assigned_entity if i[1] == 'US_SSN']
    country = [i[0] for i in columns_with_assigned_entity if i[1] == 'LOCATION' and ('country' in i[0].lower())]

    
    list_faker = []

    df_input, list_faker = get_address (df_input, addresses, list_faker)
    df_input, list_faker = get_phone_number (df_input, phone_number, list_faker)
    df_input, list_faker = get_email_address (df_input, email_address, list_faker)
    df_input = get_gender(df_input, first_name_person)
    df_input, list_faker = get_first_name(df_input, first_name_person, list_faker)
    df_input, list_faker = get_last_name(df_input, last_name_person, list_faker)
    df_input, list_faker = get_city(df_input, city, list_faker)
    df_input, list_faker = get_state(df_input, state, list_faker)
    df_input, list_faker = get_url(df_input, url, list_faker)
    df_input, list_faker = get_zipcode(df_input, zipcode, list_faker)
    df_input, list_faker = get_credit_card (df_input, credit_card, list_faker)
    df_input, list_faker = get_ssn (df_input, ssn, list_faker)
    df_input, list_faker = get_country (df_input, country, list_faker)
    

    columns_not_synthesized = get_columns_not_synthesized(columns_with_assigned_entity, list_faker, columns_with_low_cs)

    
    synthesis_message(list_faker, columns_not_synthesized)

    return df_input