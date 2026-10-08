from typing import Any, Dict, List, Optional, Union
import warnings

import gender_guesser.detector as gender
import pandas as pd
from presidio_analyzer import (
    AnalyzerEngine,
    BatchAnalyzerEngine,
    PatternRecognizer,
)
from presidio_analyzer.nlp_engine import NlpEngineProvider
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    logging,
    pipeline,
)

from nerpii.faker_generator import column_words, GENDER_COLUMN, is_first_name_column

logging.set_verbosity_error()


# è importante eseguire prima assign_entities_with_presidio, poi
# assign_entities_manually, infine assisgn_entities_with_model


def split_name(df_input: Union[str, pd.DataFrame], name_of_column: str) -> pd.DataFrame:
    """
    Return a copy of a dataframe where a column with people's full names is replaced
    by first_name and last_name columns.

    The first word of each name becomes the first name and the rest becomes the last
    name, so "Alexis R. Graves" gives "Alexis" and "R. Graves". A single word gives a
    first name and a missing last name, and missing names stay missing.

    Parameters
    ----------
    df_input : Union[str, pd.DataFrame]
        A pandas dataframe or a path to a csv file. A dataframe is not modified.
    name_of_column : str
        name of the column which contains names that have to be splitted

    Returns
    -------
    pd.DataFrame
        A pandas dataframe with first_name and last_name columns
    """

    if isinstance(df_input, pd.DataFrame):
        df_input = df_input.copy()
    else:
        df_input = pd.read_csv(df_input)

    names = df_input[name_of_column].astype("string").str.strip().str.split(n=1)
    df_input["first_name"] = names.str[0]
    df_input["last_name"] = names.str[1]

    return df_input.drop(columns=name_of_column)


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


ADDRESS_WORDS = [
    "Street",
    "Rue",
    "Via",
    "Square",
    "Avenue",
    "Place",
    "Strada",
    "St",
    "Lane",
    "Road",
    "Boulevard",
    "Ln",
    "Rd",
    "Highway",
    "Drive",
    "Av",
    "Hwy",
    "Blvd",
    "Corso",
    "Piazza",
    "Calle",
    "Plaza",
    "Avenida",
    "Rambla",
    "Vico",
    "C/",
]

# Words in a column name that mark it as a zipcode column. "cap" is the Italian
# codice di avviamento postale.
ZIPCODE_WORDS = {"zip", "zipcode", "postcode", "postalcode", "cap"}


def add_address_entity(
    lang: str, additional_addresses: Optional[List] = []
) -> PatternRecognizer:
    """
    Return a customized presidio recognizer that can recognize ADDRESS entity.
    Some address-related words are already set, but user can add others.

    Parameters
    ----------
    lang : str
        Language of the recognizer, "en" or "it"
    additional_addresses : Optional[List], optional
        A list in which user can add new address-related words, by default []

    Returns
    -------
    PatternRecognizer
        A customized presidio recognizer
    """
    return PatternRecognizer(
        supported_language=lang,
        supported_entity="ADDRESS",
        deny_list=ADDRESS_WORDS + additional_addresses,
    )


def en_add_address_entity(
    additional_addresses: Optional[List] = [],
) -> PatternRecognizer:
    """English version of add_address_entity."""
    return add_address_entity("en", additional_addresses)


def it_add_address_entity(
    additional_addresses: Optional[List] = [],
) -> PatternRecognizer:
    """Italian version of add_address_entity."""
    return add_address_entity("it", additional_addresses)


def is_zipcode_column(column: str) -> bool:
    words = column_words(column)
    return bool(ZIPCODE_WORDS & set(words)) or (
        ("postal" in words or "postale" in words)
        and ("code" in words or "codice" in words)
    )


def has_organization(labels: List[str]) -> bool:
    """
    Whether the NLP model found an organization in a value, given the labels of its
    tokens. The English model uses B-ORG/I-ORG labels and the Italian one ORG.
    """
    return any(label.split("-")[-1] == "ORG" for label in labels)


def get_gender(df_input: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of a dataframe with a first_name_gender column, guessed from its
    first first-name column.

    Pass the result to FakerGenerator to generate first names of the same gender as
    the original ones. The generator drops the column from its output.

    Parameters
    ----------
    df_input : pd.DataFrame
        A pandas dataframe. It is not modified.

    Returns
    -------
    pd.DataFrame
        A pandas dataframe with first_name_gender column, if it has a first-name
        column
    """
    df_input = df_input.copy()
    first_names = [
        column for column in df_input.columns if is_first_name_column(column)
    ]
    if not first_names:
        return df_input

    detector = gender.Detector(case_sensitive=False)
    df_input[GENDER_COLUMN] = df_input[first_names[0]].map(
        lambda name: detector.get_gender(str(name)) if pd.notna(name) else "Nan value"
    )

    return df_input


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
        A Presidio BatchAnalyzerEngine instance, created on first use.
    assigned_entities_cols : List
        A list of the object columns of the dataset for which entities have been
         assigned.
    model : Any
        A pretrained nlp model downloaded from Hugging Face, loaded on first use
    model_entities : Dict
        A dictionary whose keys are object column names and whose values are lists
        with the labels the model assigned to the tokens of each value
    dict_global_entities : Dict
        A dictionary whose keys have the same names of the dataframe columns and values
        are dictionaries in which the entity associated to the column and its confidence
        score are reported.
    """

    original_dataset: pd.DataFrame
    dataset: pd.DataFrame
    object_columns: List
    presidio_analyzer: BatchAnalyzerEngine
    assigned_entities_cols: List
    model: Any
    model_entities: Dict
    dict_global_entities: Dict
    lang: str

    def __init__(
        self,
        df_input: Union[str, pd.DataFrame],
        data_sample: Optional[int] = 500,
        nan_filler: str = "?",
        lang: Optional[str] = "en",
        get_gender_option: Optional[bool] = False,
        random_state: Optional[int] = None,
    ) -> "NamedEntityRecognizer":
        """
        Create a NamedEntityRecognizer instance.

        The Presidio analyzer and the Hugging Face model are loaded the first time
        they are needed. Presidio downloads its spaCy model if it is not installed.

        Parameters
        ----------
        df_input : Union[str, pd.DataFrame]
            A pandas dataframe or a path to a csv file.
        data_sample : Optional[int], optional
            Number of rows to sample from the dataframe, by default 500
        nan_filler : str, optional
            A string to fill the NaN values for object columns, by default "?"
        lang : str, optional
            Input language by default "en". Set this parameter to "it" to return
            better performance on Italian data.
        get_gender_option : Optional[bool], optional
            Deprecated and ignored, use get_gender() instead. By default False
        random_state : Optional[int], optional
            Seed for sampling the rows, by default None

        Returns
        -------
        NamedEntityRecognizer
            A NamedEntityRecognizer instance.
        """

        if not isinstance(df_input, pd.DataFrame):
            df_input = pd.read_csv(df_input)

        if get_gender_option:
            warnings.warn(
                "get_gender_option no longer adds a first_name_gender column to your "
                "dataframe and will be removed. Call get_gender() on the dataframe and "
                "pass the result to FakerGenerator instead.",
                FutureWarning,
                stacklevel=2,
            )

        self.dataset = df_input.sample(
            n=min(data_sample, df_input.shape[0]), random_state=random_state
        )
        # "string" picks up pandas 3's default str dtype for text columns
        self.object_columns = list(
            self.dataset.select_dtypes(["object", "string"]).columns
        )
        # fill NaN values for object columns
        self.dataset.loc[:, self.object_columns] = self.dataset.loc[
            :, self.object_columns
        ].fillna(nan_filler)
        self.lang = lang

        self.presidio_analyzer = None
        self.model = None

        self.dict_global_entities = dict.fromkeys(list(self.dataset.columns))
        self.model_entities = {}
        self.assigned_entities_cols = []

    def set_presidio_analyzer(
        self,
        add_addresses_recognizer: Optional[bool] = True,
        additional_addresses: Optional[List] = [],
    ) -> None:
        """
        Set a Presidio BatchAnalyzer for the instance.

        Parameters
        ----------
        add_addresses_recognizer : Optional[bool], optional
            Whether to add a customized address recognizer, by default True
        additional_addresses : Optional[List], optional
            A list in which user can add new address-related words, by default []
        """

        if self.lang == "it":
            configuration = {
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "it", "model_name": "it_core_news_lg"}],
            }
            provider = NlpEngineProvider(nlp_configuration=configuration)
            nlp_engine_with_italian = provider.create_engine()

            analyzer = AnalyzerEngine(
                nlp_engine=nlp_engine_with_italian,
                supported_languages=["it"],
            )

            if add_addresses_recognizer:
                addresses_recognizer = it_add_address_entity(additional_addresses)
                analyzer.registry.add_recognizer(addresses_recognizer)

            self.presidio_analyzer = BatchAnalyzerEngine(analyzer_engine=analyzer)

        else:
            analyzer = AnalyzerEngine()

            if add_addresses_recognizer:
                addresses_recognizer = en_add_address_entity(additional_addresses)
                analyzer.registry.add_recognizer(addresses_recognizer)

            self.presidio_analyzer = BatchAnalyzerEngine(analyzer_engine=analyzer)

    def set_model(self) -> None:
        """
        Set a pretrained nlp model downloaded from Hugging Face
        (https://huggingface.co/dslim/bert-base-NER) used to recognize ORGANIZATION
        entities.

        Parameters
        ----------
        nlp_model : str, optional
            A NLP model name
        """
        if self.lang == "it":
            nlp_model = "osiria/bert-italian-uncased-ner"
        else:
            nlp_model = "dslim/bert-base-NER"

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
        if self.presidio_analyzer is None:
            self.set_presidio_analyzer()

        if self.lang == "it":
            analyzer_results = list(
                self.presidio_analyzer.analyze_dict(
                    self.dataset.to_dict(orient="list"), language="it"
                )
            )
            return analyzer_results
        else:
            analyzer_results = list(
                self.presidio_analyzer.analyze_dict(
                    self.dataset.to_dict(orient="list"), language="en"
                )
            )
            return analyzer_results

    def assign_presidio_entities_list(self) -> None:
        """
        Get Presidio Analyzer results and assign entities to each object column of the
        dataset.
        """
        analyzer_results = self.get_presidio_analyzer_results()
        for col in analyzer_results:
            col_name = col.key
            if col_name in self.object_columns:
                # Get the list of entities for each record in the column
                # Keep the highest-scoring entity found in each value
                entities_list = [
                    max(value_results, key=lambda result: result.score).entity_type
                    for value_results in col.recognizer_results
                    if len(value_results) > 0
                ]
                # If the number of entities is more than 30% of the number of records,
                # assign the list to the column
                if len(entities_list) > 0.3 * self.dataset.shape[0]:
                    self.dict_global_entities[col_name] = entities_list
                    if col_name not in self.assigned_entities_cols:
                        self.assigned_entities_cols.append(col_name)

    def assign_location_entity(self) -> None:
        """
        Check whether the LOCATION entity is present among the entities assigned to the
        values in each column with assigned entities.
        If the LOCATION entity has been assigned to at least 10 percent of the values
        in that column,then the function assigns the LOCATION entity and the confidence
        score to that specific column
        """
        for col in self.assigned_entities_cols:
            entities_list = self.dict_global_entities[col]
            col_lower = col.lower()
            location_freq = frequency(entities_list, "LOCATION")
            if (
                ("LOCATION" in entities_list)
                and ("name" not in col_lower)
                and location_freq > 0.1
            ):
                self.dict_global_entities[col] = {
                    "entity": "LOCATION",
                    "confidence_score": location_freq,
                }

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
                self.dict_global_entities[col] = {
                    "entity": most_freq,
                    "confidence_score": frequency(entities_list, most_freq),
                }

    def assign_model_entities_list(self) -> None:
        """
        Assign entities to each object column which didn't get an entity from the
        Presidio Analyzer using the NLP model.
        """
        if self.model is None:
            self.set_model()

        for col in self.object_columns:
            if self.dict_global_entities[col] is None:
                self.model_entities[col] = [
                    [token["entity"] for token in tokens]
                    for tokens in self.model(self.dataset[col].tolist())
                ]

    def assign_organization_entity(self) -> None:
        """
        Check in how many values of each column the NLP model found an organization.

        If it found one in more than 10 percent of the values in that column, then the
        function assigns the ORGANIZATION entity to that column, with that share as
        the confidence score.
        """
        for col in self.model_entities:
            organization_freq = frequency(
                [has_organization(labels) for labels in self.model_entities[col]], True
            )
            if organization_freq > 0.1:
                self.dict_global_entities[col] = {
                    "entity": "ORGANIZATION",
                    "confidence_score": organization_freq,
                }

    def assign_entities_manually(
        self, zipcode: Optional[bool] = True, credit_card: Optional[bool] = True
    ) -> None:
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
            if zipcode and is_zipcode_column(col):
                self.dict_global_entities[col] = {
                    "entity": "ZIPCODE",
                    "confidence_score": 1.0,
                }
            if credit_card and (
                (("credit" in col_lower) or ("card" in col_lower))
                and ("number" in col_lower)
                or (("carta" in col_lower) and ("credito" in col_lower))
            ):
                self.dict_global_entities[col] = {
                    "entity": "CREDIT_CARD_NUMBER",
                    "confidence_score": 1.0,
                }

    def assign_entities_with_presidio(self) -> None:
        """
        Assign entities with a confidence score to each object column of the dataset
        using the Presidio Analyzer.
        """
        self.assign_presidio_entities_list()
        self.assign_location_entity()
        self.assign_entities_and_score()

    def assign_organization_entity_with_model(self) -> None:
        """
        Assign the ORGANIZATION entity with a confidence score to each object column
        of the dataset using the NLP model.
        """
        self.assign_model_entities_list()
        self.assign_organization_entity()
