import pandas as pd
from presidio_analyzer import BatchAnalyzerEngine, PatternRecognizer
import pytest
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from nerpii.named_entity_recognizer import (
    en_add_address_entity,
    frequency,
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
    # Test with default arguments
    recognizer = en_add_address_entity()
    assert isinstance(recognizer, PatternRecognizer)
    assert recognizer.deny_list == [
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
        "HighwayDrive",
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

    # Test with additional addresses
    additional_addresses = ["Alley", "Court"]
    recognizer = en_add_address_entity(additional_addresses)
    assert recognizer.deny_list == [
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
        "HighwayDrive",
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
        "Alley",
        "Court",
    ]

    # Test with empty list as additional_addresses
    recognizer = en_add_address_entity([])
    assert recognizer.deny_list == [
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
        "HighwayDrive",
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

    # Test with invalid input
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
    assert result.iloc[1]["first_name"] == "-"
    assert result.iloc[1]["last_name"] == "-"
    assert result.iloc[2]["first_name"] == "Hillary"
    assert result.iloc[2]["last_name"] == "Clinton"


def test_split_name_with_invalid_input():
    with pytest.raises(ValueError):
        split_name(None, "name")


def test__init__(instance):
    assert type(instance.dataset) is str or pd.DataFrame
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
        "zipcode": None,
    }
    assert instance.assigned_entities_cols == ["email", "city", "state", "person"]


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
        "zipcode": None,
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
        "zipcode": None,
    }


# def test_assign_model_entities_list not tested perché il risultato del modello non è
# sempre uguale a se stesso.


def test_assign_organization_entity(instance):
    instance.model_entities = {
        "university": [
            "B-ORG",
            "B-ORG",
            "I-ORG",
            "B-ORG",
            "B-ORG",
            "I-ORG",
            "B-ORG",
            "B-ORG",
            "I-ORG",
        ]
    }
    instance.assign_organization_entity()
    assert instance.dict_global_entities == {
        "email": None,
        "city": None,
        "state": None,
        "university": {
            "entity": "ORGANIZATION",
            "confidence_score": 0.6666666666666666,
        },
        "person": None,
        "zipcode": None,
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
