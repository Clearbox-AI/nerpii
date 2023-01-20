from presidio_analyzer import AnalyzerEngine, BatchAnalyzerEngine, RecognizerRegistry, PatternRecognizer
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline


def ner(df_input_path): #df_input_path: string containing path of csv file
    print('Starting...')

    df_input = pd.read_csv(df_input_path)
    df_input = df_input.sample(n=min(500, df_input.shape[0]))

    # this for loop fillna in object column with zero

    for i in df_input.columns:
        if df_input[i].dtype == 'object':
            df_input[i] = df_input[i].fillna(0)

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
    listOf_df_cols = [col for col in df_input.columns]

    for col in listOf_df_cols:
        dict_global_entities[col] = None
    
    for col in listOf_df_cols:
    #assigning ZIPCODE entity
        if (('postal' in col.lower()) and ('code' in col.lower())) or (('zip' in col.lower()) and (
            'code' in col.lower())) or (('zip' in col.lower())):
            dict_global_entities[col] = ['ZIPCODE', 1.0]
    #assigning CREDIT_CARD_NUMBER entity
        elif (('credit' in col.lower()) or ('card' in col.lower())) and ('number' in col.lower()):
            dict_global_entities[col] = ['CREDIT_CARD_NUMBER', 1.0]
    
    # the following part creates a python dictionary (keys-> column names, values-> entity type) for columns of type object
    dict_typeObject_entities = {}
    for w, j in zip(df_input.columns, analyzer_results):
        if df_input[w].dtype == 'object':
            dict_typeObject_entities[w] = [i[0].entity_type for i in j.recognizer_results if len(i) > 0]
        
    cols = [col for col in dict_typeObject_entities.keys() if len(dict_typeObject_entities[col]) > 0.3 * df_input.shape[0]]
    
    # this is a very heuristic loop that assigns each object column with an entity
  
    for col in cols:
        lst = [i for i in dict_typeObject_entities[col]]
        #assigning LOCATION entity
        if ('LOCATION' in lst) and ('name' not in col.lower()) and (
                len([i for i in dict_typeObject_entities[col] if i == 'LOCATION']) > 0.1 * df_input.shape[0]):
            dict_global_entities[col] = ['LOCATION', len([i for i in dict_typeObject_entities[col] 
            if i == 'LOCATION'])/df_input.shape[0]]
        #assigning ZIPCODE entity if zipcode in dataset is 'object' type
        elif (('postal' in col.lower()) and ('code' in col.lower())) or (('zip' in col.lower()) and (
                'code' in col.lower())) or (('zip' in col.lower())): 
            dict_global_entities[col] = ['ZIPCODE', 1.0]
        #assigning CREDIT_CARD_NUMBER entity if credit card number in dataset is 'object' type
        elif ((('credit' in col.lower()) or ('card' in col.lower())) and ('number' in col.lower())):
            dict_global_entities[col] = ['CREDIT_CARD_NUMBER', 1.0]
        else:
            most_freq = max(set(lst), key=lst.count)
            dict_global_entities[col] = [most_freq, len([i for i in lst if i == most_freq])/df_input.shape[0]]

    # tryin to assign the ORGANIZATION entity to keys/columns 
    # in dict_global_entities with None value
 

    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

    nlp_model = pipeline("ner", model=model, tokenizer=tokenizer)

    # this for loop change 0 value in object column in '/' value to be processed by the nlp model

    for i in df_input.columns:
        if df_input[i].dtype == 'object':
            df_input[i].replace(to_replace = 0, value = '/', inplace =True)

    # the following for loop returns a dict where keys are columns 
    # with no assigned entity and values are a list of the records of 
    # that column in dataset

    keyColumns_valueNone = {}

    for i in df_input.columns:
        if df_input[i].dtype == 'object' and dict_global_entities[i] == None:
            for j in range (0, df_input.shape[0]):
                keyColumns_valueNone[i]= [w for w in df_input[i]] 
    
    # applying nlp_model to recognize entities

    for i in keyColumns_valueNone.keys():
        keyColumns_valueNone[i] = nlp_model(keyColumns_valueNone[i])
    
    keyColumns_valueEntities = {}

    for i in keyColumns_valueNone.keys():
        #print(i, dict_of_entities[i])
        lst_of_entities_for_keys = []
        for j in range(0, len(keyColumns_valueNone[i])):
            if len(keyColumns_valueNone[i][j]) > 0:
                #print(i, dict_of_entities[i][j])
                for w in range(0, len(keyColumns_valueNone[i][j])):
                    lst_of_entities_for_keys.append(keyColumns_valueNone[i][j][w]['entity'])
                    keyColumns_valueEntities[i] = lst_of_entities_for_keys
    
    key_columns = [key for key in keyColumns_valueEntities.keys()]
    for col in key_columns:
        lista = [i for i in keyColumns_valueEntities[col]]
        if (('B-ORG' in lista)) and (
            len([i for i in keyColumns_valueEntities[col] if ((i == 'B-ORG'))]) > 0.1 * df_input.shape[0]):
            dict_global_entities[col] = ['ORGANIZATION', len([i for i in keyColumns_valueEntities[col] 
                    if (i == 'B-ORG')])/df_input.shape[0]]
                
    
    return dict_global_entities
