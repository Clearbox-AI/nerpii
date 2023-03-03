from typing import Dict, List, Any, Union

import pandas as pd
from faker import Faker
import gender_guesser.detector as gender
from simple_colors import *
import re
import numpy as np



class FakerGenerator:
    """
    A class used to generate faker objects in a dataframe

    Attributes
    -------
    dataset : pd.DataFrame
        A pandas dataframe 
    dict_global_entities : Dict
        A dictionary whose keys have the same names of the dataframe columns and values 
        are dictionaries in which the entity associated to the column and its confidence 
        score are reported.
    faker : Any
        A generator to obtain synthetisized objects
    columns_with_assigned_entities : List
        A list of columns with an assigned entity 
    columns_not_synthesized : List
        A list of those columns which are not synthesized by faker
    list_faker : List
        A list of those columns which are synthesized by faker

    Returns
    -------
    _type_
        _description_
    """

    dataset: pd.DataFrame
    dict_global_entities: Dict
    faker: Any
    columns_with_assigned_entity: List 
    columns_not_synthesized: List 
    list_faker: List

    def __init__(self, df_input: Union[str, pd.DataFrame], dict_global_entities: Dict) -> "FakerGenerator":
        """
        Create a FakerGenerator instance

        Parameters
        ----------
        df_input : Union[str, pd.DataFrame]
            A pandas dataframe or a path to a csv file.
        dict_global_entities : Dict
            A dictionary whose keys have the same names of the dataframe columns and values 
            are dictionaries in which the entity associated to the column and its confidence 
            score are reported.

        Returns
        -------
        FakerGenerator
            A Fakergenerator instance.
        """

        if not isinstance(df_input, pd.DataFrame):
            df_input = pd.read_csv(df_input)
        

        self.dataset = df_input
        self.dict_global_entities = dict_global_entities
        self.faker = Faker()
        self.columns_with_assigned_entity = []
        self.columns_not_synthesized = []
        self.list_faker = []

    
    def get_columns_with_assigned_entity (self) -> None:
        """
        Get a list containing those columns with an assigned entity and confidence score > 0.3.

        """

        if len(self.dict_global_entities) > 0:
            columns_with_assigned_entity = [[i, self.dict_global_entities[i]['entity']] for i in self.dict_global_entities if self.dict_global_entities[i] is not None and self.dict_global_entities[i]['confidence_score'] > 0.3]
            self.columns_not_synthesized = [[i, self.dict_global_entities[i]['entity']] for i in self.dict_global_entities if self.dict_global_entities[i] is not None and self.dict_global_entities[i]['confidence_score'] <= 0.3 and not re.match('.*?last.*?name.*?', i.lower())]

        if len(columns_with_assigned_entity) > 0:
            self.columns_with_assigned_entity = columns_with_assigned_entity
        else:
            return print ('Impossible to generate Faker data: no assigned entities.')
    
    
    def get_address (self) -> None:
        """
        Synthesize address columns in a pandas dataframe

        """

        addresses = [i[0] for i in self.columns_with_assigned_entity if i[1] == 'ADDRESS']

        for i in addresses:
            self.dataset[i] =  self.dataset.apply(lambda row: self.faker.street_address(), axis = 1)
            
            self.list_faker.append(i)


    def get_phone_number (self) -> None:
        """
        Synthesize phone number columns in a pandas dataframe

        """

        phone_number  = [i[0] for i in self.columns_with_assigned_entity if i[1] == 'PHONE_NUMBER']

        for i in phone_number:
            self.dataset[i] =  self.dataset.apply(lambda row: self.faker.phone_number(), axis = 1)
            
            self.list_faker.append(i)
    
    def get_email_address (self) -> None:
        """
        Synthesize email address columns in a pandas dataframe

        """

        email_address = [i[0] for i in self.columns_with_assigned_entity if i[1] == 'EMAIL_ADDRESS']

        for i in email_address:
            self.dataset[i] =  self.dataset.apply(lambda row: self.faker.free_email(), axis = 1)
        
            self.list_faker.append(i)
    


    def  get_first_name (self) -> None:
        """
        Synthesize first name columns in a pandas dataframe

        """

        detector = gender.Detector(case_sensitive=False)
        first_name_gender = []

        first_name_person = [i[0] for i in self.columns_with_assigned_entity if i[1] == 'PERSON' and (('first' in i[0].lower()) and ('name' in i[0].lower()))]

        if len(first_name_person) > 0:

            for col in first_name_person:
                for name in self.dataset[col]:
                    if name is not np.NaN:
                        first_name_gender.append(detector.get_gender(name))
                    else:
                        first_name_gender.append('Nan value')
            
            self.dataset['first_name_gender'] = pd.Series(first_name_gender)


        for i in first_name_person:
            self.dataset[i] =  self.dataset.apply(lambda row: self.faker.first_name_female() if (row['first_name_gender'] == 'female' or row['first_name_gender'] == 'mostly_female') else row[i], axis=1)
            self.dataset[i] =  self.dataset.apply(lambda row: self.faker.first_name_male() if (row['first_name_gender'] == 'male' or row['first_name_gender'] == 'mostly_male') else row[i], axis=1)
            self.dataset[i] =  self.dataset.apply(lambda row: self.faker.first_name() if (row['first_name_gender'] == 'unknown' or row['first_name_gender'] == 'andy') else row[i], axis=1)

            self.list_faker.append(i)

        
    
    def get_last_name (self) -> None:
        """
        Synthesize last name columns in a pandas dataframe

        """

        last_name_person = [i[0] for i in self.columns_with_assigned_entity if i[1] == 'PERSON' and (('last' in i[0].lower()) and ('name' in i[0].lower()))]

        if len(last_name_person) > 0:
            for i in last_name_person:
                self.dataset[i] =  self.dataset.apply(lambda row: self.faker.last_name(), axis = 1)

                self.list_faker.append(i)
        
        else:
            last_name_person = [i for i in self.dataset.columns if (('last' in i.lower()) and ('name' in i.lower()))]
            for i in last_name_person:
                self.dataset[i] =  self.dataset.apply(lambda row: self.faker.last_name(), axis = 1)

                self.list_faker.append(i)
    

    def get_city (self) -> None:
        """
        Synthesize city columns in a pandas dataframe

        """

        city = [i[0] for i in self.columns_with_assigned_entity if i[1] == 'LOCATION' and (('city' in i[0].lower()) or ('cities' in i[0].lower()))]

        for i in city:
            self.dataset[i] =  self.dataset.apply(lambda row: self.faker.city(), axis = 1)
        
            self.list_faker.append(i)


    def get_state(self) -> None:
        """
        Synthesize state columns in a pandas dataframe

        """

        state = [i[0] for i in self.columns_with_assigned_entity if i[1] == 'LOCATION' and ('state' in i[0].lower())]
    
        for i in state:
            if len(self.dataset[i].iloc[0]) == 2:
                self.dataset[i] =  self.dataset.apply(lambda row: self.faker.state_abbr(), axis = 1)
            else:
                self.dataset[i] =  self.dataset.apply(lambda row: self.faker.state(), axis = 1)
        
            self.list_faker.append(i)

        

    def get_url (self) -> None:
        """
        Synthesize url columns in a pandas dataframe

        """

        url = [i[0] for i in self.columns_with_assigned_entity if i[1] == 'URL']
        
        for i in url:
            self.dataset[i] =  self.dataset.apply(lambda row: self.faker.url(), axis = 1)
        
            self.list_faker.append(i)

    

    def get_zipcode (self) -> None:
        """
        Synthesize zipcode columns in a pandas dataframe

        """

        zipcode = [i[0] for i in self.columns_with_assigned_entity if i[1] == 'ZIPCODE']
        
        for i in zipcode:
            self.dataset[i] =  self.dataset.apply(lambda row: self.faker.zipcode() , axis = 1)
            
            self.list_faker.append(i)


    def get_credit_card (self) -> None:
        """
        Synthesize credit card columns in a pandas dataframe

        """

        credit_card = [i[0] for i in self.columns_with_assigned_entity if i[1] == 'CREDIT_CARD_NUMBER']

        for i in credit_card:
            self.dataset[i] =  self.dataset.apply(lambda row: self.faker.credit_card_number() , axis = 1)
            
            self.list_faker.append(i)


    def get_ssn (self) -> None:
        """
        Synthesize ssn columns in a pandas dataframe

        """

        ssn = [i[0] for i in self.columns_with_assigned_entity if i[1] == 'US_SSN']

        for i in ssn:
            self.dataset[i] =  self.dataset.apply(lambda row: self.faker.ssn() , axis = 1)
            
            self.list_faker.append(i)



    def get_country (self) -> None:
        """
        Synthesize country columns in a pandas dataframe

        """

        country = [i[0] for i in self.columns_with_assigned_entity if i[1] == 'LOCATION' and ('country' in i[0].lower())]
    
        for i in country:
            self.dataset[i] =  self.dataset.apply(lambda row: self.faker.country() , axis = 1)
            
            self.list_faker.append(i)


        

    
    def get_columns_not_synthesized (self) -> None:
        """
        Get a list of all non-synthesized columns.

        """
     
        for i in self.columns_with_assigned_entity:
                if i[0] not in self.list_faker:
                    self.columns_not_synthesized.append(i)
        

    def synthesis_message (self) -> None:
        """
        Get a message with synthesized and unsynthesized columns.

        """

        for col in self.list_faker:
            message = 'Column ' + green(col, 'bold') + ' synthesized.'
            print(message)
                
        for col in self.columns_not_synthesized:
            message = 'Column ' + red(col[0], 'bold') + ' not synthesized.'
            print(message)

    

    def get_faker_generation(self) -> None:
        """
        Get faker objects for columns in a pandas dataframe
        
        """
        self.get_columns_with_assigned_entity()
        self.get_address()
        self.get_phone_number()
        self.get_email_address()
        self.get_first_name()
        self.get_last_name()
        self.get_city()
        self.get_state()
        self.get_url()
        self.get_zipcode()
        self.get_credit_card()
        self.get_ssn()
        self.get_country()


        self.get_columns_not_synthesized()

        self.synthesis_message()

