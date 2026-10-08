import re
from typing import Any, Dict, List, Optional, Sequence, Union
import warnings

import gender_guesser.detector as gender
import pandas as pd
from presidio_analyzer import (
    AnalyzerEngine,
    BatchAnalyzerEngine,
    PatternRecognizer,
)
from presidio_analyzer.nlp_engine import NlpEngineProvider
import spacy
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

# Address words are matched with their case: lowercase "via" and "place" are
# common words, and in free text they would beat the names found by spaCy.
ADDRESS_REGEX_FLAGS = re.DOTALL | re.MULTILINE

SPACY_MODELS = {"en": "en_core_web_lg", "it": "it_core_news_lg"}
HF_MODELS = {"en": "dslim/bert-base-NER", "it": "osiria/bert-italian-uncased-ner"}

# Values the NLP model reads at once
MODEL_BATCH_SIZE = 32


class ModelUnavailable(RuntimeError):
    """A spaCy or Hugging Face model is not installed and offline=True forbids
    downloading it."""


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
        global_regex_flags=ADDRESS_REGEX_FLAGS,
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


def build_presidio_analyzer(
    lang: str = "en",
    *,
    add_addresses_recognizer: bool = True,
    additional_addresses: Optional[List] = None,
    score_threshold: float = 0.0,
    offline: bool = False,
) -> BatchAnalyzerEngine:
    """
    Build a Presidio BatchAnalyzerEngine, to pass to one or more NamedEntityRecognizer
    instances so that the spaCy model is loaded once.

    Parameters
    ----------
    lang : str, optional
        Language, "en" (en_core_web_lg) or "it" (it_core_news_lg), by default "en"
    add_addresses_recognizer : bool, optional
        Whether to add a customized address recognizer, by default True
    additional_addresses : Optional[List], optional
        Address-related words to add to the address recognizer, by default None
    score_threshold : float, optional
        Detections scoring below this are discarded, by default 0.0 (Presidio's
        default, which keeps them all)
    offline : bool, optional
        Raise ModelUnavailable instead of downloading a missing spaCy model, by
        default False

    Returns
    -------
    BatchAnalyzerEngine
        A Presidio batch analyzer
    """
    lang = "it" if lang == "it" else "en"
    model_name = SPACY_MODELS[lang]
    if offline and not spacy.util.is_package(model_name):
        raise ModelUnavailable(
            f"the spaCy model {model_name} is not installed "
            f"(python -m spacy download {model_name})"
        )
    if lang == "it":
        configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "it", "model_name": model_name}],
        }
        provider = NlpEngineProvider(nlp_configuration=configuration)
    else:
        # Presidio's default configuration: en_core_web_lg, with spaCy's ORG
        # and number labels ignored
        provider = NlpEngineProvider()
    nlp_engine = provider.create_engine()
    analyzer = AnalyzerEngine(
        nlp_engine=nlp_engine,
        supported_languages=[lang],
        default_score_threshold=score_threshold,
    )
    if add_addresses_recognizer:
        analyzer.registry.add_recognizer(
            add_address_entity(lang, list(additional_addresses or []))
        )
    return BatchAnalyzerEngine(analyzer_engine=analyzer)


def build_model(
    lang: str = "en", *, batch_size: int = MODEL_BATCH_SIZE, offline: bool = False
) -> Any:
    """
    Build the Hugging Face token-classification pipeline that finds organizations,
    to pass to one or more NamedEntityRecognizer instances so that it is loaded once.

    Parameters
    ----------
    lang : str, optional
        Language, "en" (dslim/bert-base-NER) or "it"
        (osiria/bert-italian-uncased-ner), by default "en"
    batch_size : int, optional
        Values the model reads at once, by default 32
    offline : bool, optional
        Only use the local Hugging Face cache, and raise ModelUnavailable if the
        model is not in it, by default False

    Returns
    -------
    Any
        A transformers "ner" pipeline
    """
    model_name = HF_MODELS["it" if lang == "it" else "en"]
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=offline)
        model = AutoModelForTokenClassification.from_pretrained(
            model_name, local_files_only=offline
        )
    except OSError as exc:
        if offline:
            raise ModelUnavailable(
                f"the Hugging Face model {model_name} is not in the local cache "
                f"(huggingface-cli download {model_name})"
            ) from exc
        raise
    return pipeline("ner", model=model, tokenizer=tokenizer, batch_size=batch_size)


def _is_primitive(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool))


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
        The text columns of the dataset that are analyzed: object or string columns
        whose values are all strings, numbers or booleans.
    presidio_analyzer : BatchAnalyzerEngine
        A Presidio BatchAnalyzerEngine instance, passed in or created on first use.
    assigned_entities_cols : List
        A list of the object columns of the dataset for which entities have been
         assigned.
    model : Any
        A pretrained nlp model from Hugging Face, passed in or loaded on first use
    model_entities : Dict
        A dictionary whose keys are object column names and whose values are lists
        with the labels the model assigned to the tokens of each value
    dict_global_entities : Dict
        A dictionary whose keys have the same names of the dataframe columns and values
        are dictionaries in which the entity associated to the column and its confidence
        score are reported.
    column_stats : Dict
        For each analyzed column, how many of its sampled values are not missing
        ("n_values"), have any Presidio detection ("n_detected"), contain each entity
        ("entity_values"), have each entity as their highest-scoring one
        ("top_entity_values") and, after the NLP model ran, contain an organization
        ("organization_values"). Unlike the confidence score, these count missing
        values out, so a share of n_values is the share of the column.
    """

    original_dataset: pd.DataFrame
    dataset: pd.DataFrame
    object_columns: List
    presidio_analyzer: BatchAnalyzerEngine
    assigned_entities_cols: List
    model: Any
    model_entities: Dict
    dict_global_entities: Dict
    column_stats: Dict
    lang: str
    offline: bool

    def __init__(
        self,
        df_input: Union[str, pd.DataFrame],
        data_sample: Optional[int] = 500,
        nan_filler: str = "?",
        lang: Optional[str] = "en",
        get_gender_option: Optional[bool] = False,
        random_state: Optional[int] = None,
        columns: Optional[Sequence[str]] = None,
        presidio_analyzer: Optional[BatchAnalyzerEngine] = None,
        model: Any = None,
        offline: bool = False,
    ) -> "NamedEntityRecognizer":
        """
        Create a NamedEntityRecognizer instance.

        The Presidio analyzer and the Hugging Face model are loaded the first time
        they are needed, unless they are passed in (see build_presidio_analyzer and
        build_model). Presidio downloads its spaCy model if it is not installed,
        unless offline is True.

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
        columns : Optional[Sequence[str]], optional
            Analyze only these columns, by default all the text columns
        presidio_analyzer : Optional[BatchAnalyzerEngine], optional
            A Presidio analyzer to use instead of building one, by default None
        model : Any, optional
            An NLP pipeline to use instead of loading one, by default None
        offline : bool, optional
            Never download a model: raise ModelUnavailable if one is not installed,
            by default False

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
        # "string" picks up pandas 3's default str dtype for text columns. Object
        # columns of dates or other objects are left out: Presidio rejects them.
        self.object_columns = [
            col
            for col in self.dataset.select_dtypes(["object", "string"]).columns
            if (columns is None or col in columns)
            and self.dataset[col].dropna().map(_is_primitive).all()
        ]
        self._missing = self.dataset[self.object_columns].isna()
        # fill NaN values for object columns
        self.dataset.loc[:, self.object_columns] = self.dataset.loc[
            :, self.object_columns
        ].fillna(nan_filler)
        self.lang = lang
        self.offline = offline

        self.presidio_analyzer = presidio_analyzer
        self.model = model

        self.dict_global_entities = dict.fromkeys(list(self.dataset.columns))
        self.model_entities = {}
        self.assigned_entities_cols = []
        self.column_stats = {}

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

        self.presidio_analyzer = build_presidio_analyzer(
            "it" if self.lang == "it" else "en",
            add_addresses_recognizer=add_addresses_recognizer,
            additional_addresses=additional_addresses,
            offline=self.offline,
        )

    def set_model(self) -> None:
        """
        Set a pretrained nlp model downloaded from Hugging Face
        (https://huggingface.co/dslim/bert-base-NER) used to recognize ORGANIZATION
        entities.
        """
        self.model = build_model(
            "it" if self.lang == "it" else "en", offline=self.offline
        )

    def get_presidio_analyzer_results(self) -> List:
        """
        Get the results of the Presidio BatchAnalyzer: assign entities to each record
        in each columns of the dataset.

        Returns
        -------
        List
            A list containing the results of the analyzer.
        """
        if not self.object_columns:
            return []
        if self.presidio_analyzer is None:
            self.set_presidio_analyzer()

        # Only the text columns: the results for the others were never used
        return list(
            self.presidio_analyzer.analyze_dict(
                self.dataset[self.object_columns].to_dict(orient="list"),
                language="it" if self.lang == "it" else "en",
            )
        )

    def assign_presidio_entities_list(self) -> None:
        """
        Get Presidio Analyzer results and assign entities to each object column of the
        dataset.
        """
        analyzer_results = self.get_presidio_analyzer_results()
        for col in analyzer_results:
            col_name = col.key
            if col_name in self.object_columns:
                self.column_stats[col_name] = self._presidio_stats(
                    col_name, col.recognizer_results
                )
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

    def _presidio_stats(self, col: str, recognizer_results: List) -> Dict:
        entity_values: Dict[str, int] = {}
        top_entity_values: Dict[str, int] = {}
        n_detected = 0
        for missing, value_results in zip(
            self._missing[col], recognizer_results, strict=True
        ):
            if missing or not value_results:
                continue
            n_detected += 1
            for entity in {result.entity_type for result in value_results}:
                entity_values[entity] = entity_values.get(entity, 0) + 1
            top = max(value_results, key=lambda result: result.score).entity_type
            top_entity_values[top] = top_entity_values.get(top, 0) + 1
        return {
            "n_values": int((~self._missing[col]).sum()),
            "n_detected": n_detected,
            "entity_values": entity_values,
            "top_entity_values": top_entity_values,
        }

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

    def assign_model_entities_list(self, skip_columns: Sequence[str] = ()) -> None:
        """
        Assign entities to each object column which didn't get an entity from the
        Presidio Analyzer using the NLP model.

        The model reads each distinct value once.

        Parameters
        ----------
        skip_columns : Sequence[str], optional
            Columns not to pass to the model, e.g. long free text, by default ()
        """
        columns = [
            col
            for col in self.object_columns
            if self.dict_global_entities[col] is None and col not in skip_columns
        ]
        if not columns:
            return
        if self.model is None:
            self.set_model()

        for col in columns:
            values = self.dataset[col].astype(str).tolist()
            distinct = list(dict.fromkeys(values))
            labels = {
                value: [token["entity"] for token in tokens]
                for value, tokens in zip(distinct, self.model(distinct), strict=True)
            }
            self.model_entities[col] = [labels[value] for value in values]
            self.column_stats.setdefault(col, {})["organization_values"] = sum(
                has_organization(labels[value])
                for value, missing in zip(values, self._missing[col], strict=True)
                if not missing
            )

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

    def assign_organization_entity_with_model(
        self, skip_columns: Sequence[str] = ()
    ) -> None:
        """
        Assign the ORGANIZATION entity with a confidence score to each object column
        of the dataset using the NLP model.

        Parameters
        ----------
        skip_columns : Sequence[str], optional
            Columns not to pass to the model, e.g. long free text, by default ()
        """
        self.assign_model_entities_list(skip_columns)
        self.assign_organization_entity()
