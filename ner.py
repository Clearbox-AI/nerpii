from presidio_analyzer import AnalyzerEngine, BatchAnalyzerEngine, RecognizerRegistry
import pandas as pd


def ner(df_input_path): #df_input_path: string containing path of csv file

    df_input = pd.read_csv(df_input_path)
    df_input = df_input.sample(n=min(1000, df_input.shape[0]))
    analyzer = AnalyzerEngine(supported_languages=["en", "fr"])
    batch_analyzer = BatchAnalyzerEngine(analyzer_engine=analyzer)

    # Presidio was created as a tool to recognize entities in text, the following line is used to analyze a pandas dataframe
    # by first converting it to json
    analyzer_results = list(batch_analyzer.analyze_dict(df_input.to_dict(orient="list"), language="en"))
    
    # the following part creates a python dictionary (keys-> column names, values-> entity type)
    l = {}
    for w, j in zip(df_input.columns, analyzer_results):
        if df_input[w].dtype == 'object':
            l[w] = [i[0].entity_type for i in j.recognizer_results if len(i) > 0]
  
    list_entities = {}
    cols = [col for col in df_input.columns]

    for col in cols:
        list_entities[col] = None
        
    cols = [col for col in l.keys() if len(l[col]) > 0.1 * df_input.shape[0]]
    
    # this is a very heuristic loop that assigns each object column with an entity
  
    for col in cols:
        lst = [i for i in l[col]]
        if ('LOCATION' in lst) and ('name' not in col.lower()) and (
                len([i for i in l[col] if i == 'LOCATION']) > 0.1 * df_input.shape[0]):
            list_entities[col] = ['LOCATION', len([i for i in l[col] if i == 'LOCATION'])/df_input.shape[0]]
        else:
            most_freq = max(set(lst), key=lst.count)
            list_entities[col] = [most_freq, len([i for i in lst if i == most_freq])/df_input.shape[0]]
    
    # This heuristic is designed to the understand whether a location entity is actually an address
    # (presidio only recognise a single entity for addresses, cities, countries: LOCATION)
    loc_cols = [[i, list_entities[i][0]] for i in list_entities.keys() if list_entities[i] is not None]
    loc_cols = [i[0] for i in loc_cols if i[1] == 'LOCATION']

        
    addresses = ['street', 'rue', 'via', 'square', 'avenue', 'place', 'strada', 'st', 'lane', 'road', 'boulevard', 'nln']
    for col in loc_cols:
        list_addresses = [i for i in df_input[col].fillna('none') if any(ele in i.lower() for ele in addresses)]
        if len(list_addresses) > 0.1 * int(df_input.shape[0]):
            list_entities[col] = ['ADDRESS', len(list_addresses)/int(df_input.shape[0])]

    return list_entities
