import datetime as dt
from decimal import Decimal
from unittest.mock import Mock

import pandas as pd
from presidio_analyzer import (
    BatchAnalyzerEngine,
    DictAnalyzerResult,
    PatternRecognizer,
    RecognizerResult,
)
import pytest
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from nerpii.named_entity_recognizer import (
    ADDRESS_WORDS,
    build_model,
    build_presidio_analyzer,
    en_add_address_entity,
    frequency,
    get_gender,
    it_add_address_entity,
    ModelUnavailable,
    NamedEntityRecognizer,
    split_name,
)


def test_frequency():
    assert frequency(values=[2, 5, 5, 5, 7, 8, 9, 10], element=5) == 0.375
    assert (
        frequency(
            values=[
                "apple",
                "apple",
                "banana",
                "pineapple",
                "apple",
                "apple",
                "pear",
                "peach",
            ],
            element="apple",
        )
        == 0.5
    )
    assert frequency(values=[], element=1) == 0


def test_en_add_address_entity():
    recognizer = en_add_address_entity()
    assert isinstance(recognizer, PatternRecognizer)
    assert recognizer.supported_language == "en"
    assert recognizer.deny_list == ADDRESS_WORDS
    assert "Highway" in recognizer.deny_list
    assert "Drive" in recognizer.deny_list

    recognizer = en_add_address_entity(["Alley", "Court"])
    assert recognizer.deny_list == ADDRESS_WORDS + ["Alley", "Court"]


def test_it_add_address_entity():
    recognizer = it_add_address_entity(["Vicolo"])
    assert recognizer.supported_language == "it"
    assert recognizer.deny_list == ADDRESS_WORDS + ["Vicolo"]


def test_add_address_entity_with_invalid_input():
    with pytest.raises(TypeError):
        en_add_address_entity("Invalid input")


@pytest.fixture
def dataset():
    return pd.DataFrame(
        {
            "email": ["John@email.com.", "Snow@email.com", "frank@email.com"],
            "city": ["New York", "Chicago", "Phoenix"],
            "state": ["Washington", "Florida", "Texas"],
            "university": [
                "University of London",
                "University of Georgia",
                "University of California",
            ],
            "person": ["George Bush", None, "Hillary Clinton"],
            "zipcode": ["10145", "N11RG", "56178"],
        }
    )


@pytest.fixture
def instance(dataset):
    return NamedEntityRecognizer(dataset)


def test_split_name_with_dataframe(dataset):
    result = split_name(dataset, "person")
    assert "first_name" in result.columns
    assert "last_name" in result.columns
    assert result.iloc[0]["first_name"] == "George"
    assert result.iloc[0]["last_name"] == "Bush"
    assert pd.isna(result.iloc[1]["first_name"])
    assert pd.isna(result.iloc[1]["last_name"])
    assert result.iloc[2]["first_name"] == "Hillary"
    assert result.iloc[2]["last_name"] == "Clinton"


def test_split_name_with_invalid_input():
    with pytest.raises(ValueError):
        split_name(None, "name")


def test__init__(instance):
    assert isinstance(instance.dataset, pd.DataFrame)
    assert instance.dataset.loc[:, instance.object_columns].isna().values.any() == False

    with pytest.raises(ValueError):
        instance.__init__(None)


def test_set_presidio_analyzer(instance):
    instance.set_presidio_analyzer()
    assert type(instance.presidio_analyzer) is BatchAnalyzerEngine


def test_set_model(instance, nlp_model="dslim/bert-base-NER"):
    tokenizer = AutoTokenizer.from_pretrained(nlp_model)
    model = AutoModelForTokenClassification.from_pretrained(nlp_model)
    instance.set_model()
    assert isinstance(
        instance.model, type(pipeline("ner", model=model, tokenizer=tokenizer))
    )


def test_get_presidio_analyzer_results(instance):
    instance.set_presidio_analyzer()
    results = instance.get_presidio_analyzer_results()
    assert type(results) is list


def test_assign_presidio_entity_list(instance):
    instance.set_presidio_analyzer()
    instance.get_presidio_analyzer_results()
    instance.assign_presidio_entities_list()
    assert instance.dict_global_entities == {
        "email": ["EMAIL_ADDRESS", "EMAIL_ADDRESS", "EMAIL_ADDRESS"],
        "city": ["LOCATION", "LOCATION", "LOCATION"],
        "state": ["LOCATION", "LOCATION", "LOCATION"],
        "university": None,
        "person": ["PERSON", "PERSON"],
        # Presidio tags 5-digit zipcodes as dates; assign_entities_manually then
        # assigns ZIPCODE from the column name
        "zipcode": ["DATE_TIME", "DATE_TIME"],
    }
    assert instance.assigned_entities_cols == [
        "email",
        "city",
        "state",
        "person",
        "zipcode",
    ]


def test_assign_location_entity(instance):
    instance.set_presidio_analyzer()
    instance.get_presidio_analyzer_results()
    instance.assign_presidio_entities_list()
    instance.assign_location_entity()
    assert instance.dict_global_entities == {
        "email": ["EMAIL_ADDRESS", "EMAIL_ADDRESS", "EMAIL_ADDRESS"],
        "city": {"entity": "LOCATION", "confidence_score": 1.0},
        "state": {"entity": "LOCATION", "confidence_score": 1.0},
        "university": None,
        "person": ["PERSON", "PERSON"],
        "zipcode": ["DATE_TIME", "DATE_TIME"],
    }


def test_assign_location_entity_not_enough_confidence_score(instance):
    instance.dict_global_entities = {
        "email": ["EMAIL_ADDRESS", "EMAIL_ADDRESS", "EMAIL_ADDRESS"],
        "city": ["LOCATION", "LOCATION", "LOCATION"],
        "state": ["GPE", "GPE"],
        "university": None,
        "person": ["PERSON", "PERSON"],
        "zipcode": None,
    }
    instance.assigned_entities_cols = ["city", "state"]
    instance.assign_location_entity()
    assert instance.dict_global_entities == {
        "email": ["EMAIL_ADDRESS", "EMAIL_ADDRESS", "EMAIL_ADDRESS"],
        "city": {"entity": "LOCATION", "confidence_score": 1.0},
        "state": ["GPE", "GPE"],
        "university": None,
        "person": ["PERSON", "PERSON"],
        "zipcode": None,
    }


def test_assign_entities_and_score(instance):
    instance.set_presidio_analyzer()
    instance.assign_presidio_entities_list()
    instance.assign_entities_and_score()
    assert instance.dict_global_entities == {
        "email": {"entity": "EMAIL_ADDRESS", "confidence_score": 1.0},
        "city": {"entity": "LOCATION", "confidence_score": 1.0},
        "state": {"entity": "LOCATION", "confidence_score": 1.0},
        "university": None,
        "person": {"entity": "PERSON", "confidence_score": 1.0},
        "zipcode": {"entity": "DATE_TIME", "confidence_score": 1.0},
    }


# def test_assign_model_entities_list not tested perché il risultato del modello non è
# sempre uguale a se stesso.


def test_assign_organization_entity(instance):
    # One label list per value: an organization in two of the three values
    instance.model_entities = {"university": [["B-ORG", "I-ORG"], [], ["B-ORG"]]}
    instance.assign_organization_entity()
    assert instance.dict_global_entities == {
        "email": None,
        "city": None,
        "state": None,
        "university": {
            "entity": "ORGANIZATION",
            "confidence_score": 2 / 3,
        },
        "person": None,
        "zipcode": None,
    }


def test_organization_score_counts_values_not_tokens(instance):
    # Many organization tokens in one value still count as one value
    instance.model_entities = {
        "university": [["B-ORG", "I-ORG", "I-ORG", "B-ORG"], ["B-PER"], ["B-LOC"]]
    }
    instance.assign_organization_entity()
    assert instance.dict_global_entities["university"]["confidence_score"] == 1 / 3


def test_organization_with_italian_labels(instance):
    # The Italian model labels organizations ORG, without a B-/I- prefix
    instance.model_entities = {"university": [["ORG"], ["ORG"], ["LOC"]]}
    instance.assign_organization_entity()
    assert instance.dict_global_entities["university"] == {
        "entity": "ORGANIZATION",
        "confidence_score": 2 / 3,
    }


def test_assign_entities_manually(instance):
    instance.assign_entities_manually()
    assert instance.dict_global_entities == {
        "email": None,
        "city": None,
        "state": None,
        "university": None,
        "person": None,
        "zipcode": {"entity": "ZIPCODE", "confidence_score": 1.0},
    }


def test_init_does_not_load_models(dataset, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("model loaded in __init__")

    monkeypatch.setattr("spacy.load", fail)
    monkeypatch.setattr("spacy.cli.download", fail)
    monkeypatch.setattr(
        "nerpii.named_entity_recognizer.AutoModelForTokenClassification"
        ".from_pretrained",
        fail,
    )
    recognizer = NamedEntityRecognizer(dataset)
    assert recognizer.presidio_analyzer is None
    assert recognizer.model is None


def test_random_state_makes_sample_reproducible():
    df = pd.DataFrame({"n": range(100)})
    first = NamedEntityRecognizer(df, data_sample=10, random_state=0)
    second = NamedEntityRecognizer(df, data_sample=10, random_state=0)
    assert list(first.dataset.index) == list(second.dataset.index)


def test_presidio_analyzer_is_reused(instance):
    analyzer = Mock()
    analyzer.analyze_dict.return_value = []
    instance.presidio_analyzer = analyzer
    instance.assign_entities_with_presidio()
    instance.assign_entities_with_presidio()
    assert instance.presidio_analyzer is analyzer
    assert analyzer.analyze_dict.call_count == 2


def test_model_is_reused(instance):
    model = Mock(side_effect=lambda values: [[{"entity": "B-ORG"}] for _ in values])
    instance.model = model
    instance.assign_organization_entity_with_model()
    assert instance.model is model
    assert instance.dict_global_entities["university"] == {
        "entity": "ORGANIZATION",
        "confidence_score": 1.0,
    }


@pytest.fixture
def names():
    return pd.DataFrame(
        {
            "name": ["Alexis R. Graves", None, "Vacant", "  ", " Mary  Smith "],
            "phone": ["1", "2", "3", "4", "5"],
        },
        index=[40, 30, 20, 10, 0],
    )


def test_split_name_keeps_middle_names_in_last_name(names):
    result = split_name(names, "name")
    assert result.loc[40, "first_name"] == "Alexis"
    assert result.loc[40, "last_name"] == "R. Graves"
    assert result.loc[0, "first_name"] == "Mary"
    assert result.loc[0, "last_name"] == "Smith"


def test_split_name_keeps_missing_names_missing(names):
    result = split_name(names, "name")
    assert result.loc[[30, 10], ["first_name", "last_name"]].isna().all().all()
    assert result.loc[20, "first_name"] == "Vacant"
    assert pd.isna(result.loc[20, "last_name"])


def test_split_name_keeps_index_and_input(names):
    original = names.copy()
    result = split_name(names, "name")
    assert list(result.index) == [40, 30, 20, 10, 0]
    assert list(result["phone"]) == ["1", "2", "3", "4", "5"]
    assert "name" not in result.columns
    pd.testing.assert_frame_equal(names, original)


def test_get_gender_follows_index_and_keeps_input():
    df = pd.DataFrame(
        {"first_name": ["Anna", None, "John"]},
        index=[2, 1, 0],
    )
    original = df.copy()
    result = get_gender(df)
    assert result["first_name_gender"].to_dict() == {
        2: "female",
        1: "Nan value",
        0: "male",
    }
    pd.testing.assert_frame_equal(df, original)


def test_get_gender_uses_first_first_name_column():
    df = pd.DataFrame(
        {"first_name": ["Anna", "John"], "partner_first_name": ["John", "Anna"]}
    )
    result = get_gender(df)
    assert list(result["first_name_gender"]) == ["female", "male"]


def test_get_gender_without_first_name_column():
    df = pd.DataFrame({"city": ["Rome", "Milan"]})
    assert list(get_gender(df).columns) == ["city"]


def test_get_gender_option_is_deprecated_and_keeps_input():
    df = pd.DataFrame({"first_name": ["Mary", "John"]})
    with pytest.warns(FutureWarning, match="get_gender"):
        NamedEntityRecognizer(df, get_gender_option=True)
    assert list(df.columns) == ["first_name"]


@pytest.mark.parametrize(
    "column",
    [
        "zip",
        "Zip Code",
        "ZipCode",
        "zip5",
        "postcode",
        "postal_code",
        "CAP",
        "cap_residenza",
        "codice_postale",
    ],
)
def test_zipcode_columns(column):
    recognizer = NamedEntityRecognizer(pd.DataFrame({column: ["x"]}))
    recognizer.assign_entities_manually()
    assert recognizer.dict_global_entities[column]["entity"] == "ZIPCODE"


@pytest.mark.parametrize(
    "column", ["capital", "capacity", "caption", "escape_flag", "Recap", "unzipped"]
)
def test_columns_containing_zip_or_cap_are_not_zipcodes(column):
    recognizer = NamedEntityRecognizer(pd.DataFrame({column: ["x"]}))
    recognizer.assign_entities_manually()
    assert recognizer.dict_global_entities[column] is None


def test_presidio_keeps_highest_scoring_entity(instance):
    def value(*results):
        return [RecognizerResult(entity, 0, 1, score) for entity, score in results]

    analyzer = Mock()
    analyzer.analyze_dict.return_value = [
        DictAnalyzerResult(
            key="city",
            value=[],
            recognizer_results=[
                value(("PERSON", 0.4), ("LOCATION", 0.85)),
                value(("LOCATION", 0.85)),
                value(("DATE_TIME", 0.6), ("LOCATION", 0.85)),
            ],
        )
    ]
    instance.presidio_analyzer = analyzer
    instance.assign_entities_with_presidio()
    assert instance.dict_global_entities["city"] == {
        "entity": "LOCATION",
        "confidence_score": 1.0,
    }


def _results(key, *values):
    """A DictAnalyzerResult with one list of (entity, score) pairs per value."""
    return DictAnalyzerResult(
        key=key,
        value=[],
        recognizer_results=[
            [RecognizerResult(entity, 0, 1, score) for entity, score in value]
            for value in values
        ],
    )


def _fail(*args, **kwargs):
    raise AssertionError("a model was loaded")


def test_build_presidio_analyzer_offline_does_not_download(monkeypatch):
    monkeypatch.setattr("spacy.util.is_package", lambda name: False)
    monkeypatch.setattr("spacy.cli.download", _fail)
    monkeypatch.setattr("spacy.load", _fail)
    with pytest.raises(ModelUnavailable, match="it_core_news_lg"):
        build_presidio_analyzer("it", offline=True)


def test_build_model_offline_uses_the_local_cache_only(monkeypatch):
    calls = []

    def missing(name, **kwargs):
        calls.append(kwargs)
        raise OSError("not in the cache")

    monkeypatch.setattr(
        "nerpii.named_entity_recognizer.AutoTokenizer.from_pretrained", missing
    )
    with pytest.raises(ModelUnavailable, match="dslim/bert-base-NER"):
        build_model("en", offline=True)
    assert calls == [{"local_files_only": True}]


def test_build_model_online_errors_are_not_hidden(monkeypatch):
    def missing(name, **kwargs):
        raise OSError("no network")

    monkeypatch.setattr(
        "nerpii.named_entity_recognizer.AutoTokenizer.from_pretrained", missing
    )
    with pytest.raises(OSError, match="no network"):
        build_model("en")


def test_engines_passed_in_are_used(dataset, monkeypatch):
    monkeypatch.setattr("spacy.load", _fail)
    monkeypatch.setattr(
        "nerpii.named_entity_recognizer.AutoModelForTokenClassification"
        ".from_pretrained",
        _fail,
    )
    analyzer = Mock()
    analyzer.analyze_dict.return_value = []
    model = Mock(side_effect=lambda values: [[] for _ in values])
    recognizer = NamedEntityRecognizer(dataset, presidio_analyzer=analyzer, model=model)
    recognizer.assign_entities_with_presidio()
    recognizer.assign_organization_entity_with_model()
    assert analyzer.analyze_dict.call_count == 1
    assert model.call_count == len(recognizer.object_columns)


def test_only_text_columns_are_analyzed():
    df = pd.DataFrame(
        {
            "name": ["Mario Rossi", "Anna Bianchi"],
            "born": [dt.date(1980, 1, 1), dt.date(1990, 2, 2)],
            "amount": [Decimal("1.5"), Decimal("2.5")],
            "mixed": ["a", 1],
            "count": [1, 2],
            "seen": pd.to_datetime(["2020-01-01", "2021-01-01"]),
        }
    )
    analyzer = Mock()
    analyzer.analyze_dict.return_value = []
    recognizer = NamedEntityRecognizer(df, presidio_analyzer=analyzer)
    assert recognizer.object_columns == ["name", "mixed"]
    recognizer.assign_entities_with_presidio()
    sent = analyzer.analyze_dict.call_args.args[0]
    assert list(sent) == ["name", "mixed"]
    assert list(recognizer.dict_global_entities) == list(df.columns)


def test_columns_restricts_the_analysis(dataset):
    recognizer = NamedEntityRecognizer(dataset, columns=["email", "person"])
    assert recognizer.object_columns == ["email", "person"]


def test_no_text_columns_loads_nothing(monkeypatch):
    monkeypatch.setattr("spacy.load", _fail)
    recognizer = NamedEntityRecognizer(pd.DataFrame({"n": [1, 2, 3]}))
    recognizer.assign_entities_with_presidio()
    assert recognizer.presidio_analyzer is None
    assert recognizer.dict_global_entities == {"n": None}


def test_column_stats_count_missing_values_out():
    df = pd.DataFrame({"notes": ["Seen by Dr. Rossi", None, "stable", "Rossi, Via Po"]})
    found = {
        "Seen by Dr. Rossi": [("PERSON", 0.85)],
        "Rossi, Via Po": [("PERSON", 0.85), ("ADDRESS", 1.0)],
    }
    analyzer = Mock()
    # Results follow the sampled (shuffled) rows; the missing value arrives as "?"
    analyzer.analyze_dict.side_effect = lambda values, language: [
        _results("notes", *(found.get(v, []) for v in values["notes"]))
    ]
    recognizer = NamedEntityRecognizer(df, presidio_analyzer=analyzer)
    recognizer.assign_entities_with_presidio()
    assert recognizer.column_stats["notes"] == {
        "n_values": 3,
        "n_detected": 2,
        "entity_values": {"PERSON": 2, "ADDRESS": 1},
        "top_entity_values": {"PERSON": 1, "ADDRESS": 1},
    }
    # The confidence score is unchanged: a share of the values with a detection
    assert recognizer.dict_global_entities["notes"]["confidence_score"] == 0.5


def test_model_reads_each_distinct_value_once():
    df = pd.DataFrame(
        {
            "company": ["Acme", "Acme", "Globex", None, "Acme"],
            "notes": ["long text"] * 5,
        }
    )
    read = []

    def model(values):
        read.append(list(values))
        return [[{"entity": "B-ORG"}] if v == "Acme" else [] for v in values]

    analyzer = Mock()
    analyzer.analyze_dict.return_value = []
    recognizer = NamedEntityRecognizer(df, presidio_analyzer=analyzer, model=model)
    recognizer.assign_entities_with_presidio()
    recognizer.assign_organization_entity_with_model(skip_columns=["notes"])
    # One call, each distinct value once (in the order of the shuffled sample)
    assert len(read) == 1 and sorted(read[0]) == ["?", "Acme", "Globex"]
    expected = [
        ["B-ORG"] if value == "Acme" else [] for value in recognizer.dataset["company"]
    ]
    assert recognizer.model_entities["company"] == expected
    assert recognizer.dict_global_entities["company"] == {
        "entity": "ORGANIZATION",
        "confidence_score": 3 / 5,
    }
    assert recognizer.column_stats["company"]["organization_values"] == 3
    assert recognizer.dict_global_entities["notes"] is None


def test_model_is_not_loaded_when_no_column_needs_it(monkeypatch):
    monkeypatch.setattr(
        "nerpii.named_entity_recognizer.AutoModelForTokenClassification"
        ".from_pretrained",
        _fail,
    )
    recognizer = NamedEntityRecognizer(pd.DataFrame({"notes": ["a", "b"]}))
    recognizer.assign_organization_entity_with_model(skip_columns=["notes"])
    assert recognizer.model is None


@pytest.mark.parametrize(
    "text, found",
    [
        ("Via Roma 12, Milano", True),
        ("221B Baker Street", True),
        ("morphine given via IV", False),
        ("patient in stable place", False),
    ],
)
def test_address_words_match_their_case(text, found):
    recognizer = en_add_address_entity()
    results = recognizer.analyze(text, ["ADDRESS"])
    assert bool(results) is found
