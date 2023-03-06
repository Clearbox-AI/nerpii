from typing import Dict

from faker import Faker
import pandas as pd
import pytest

from nerpii.faker_generator import FakerGenerator
from nerpii.named_entity_recognizer import NamedEntityRecognizer, split_name


@pytest.fixture
def dataset():
    return pd.DataFrame(
        {
            "email": ["John@email.com.", "Snow@email.com", "frank@email.com"],
            "city": ["New York", "Chicago", "Phoenix"],
            "state": ["Washigton", "Rhode Island", "Texas"],
            "university": [
                "University of London",
                "University of Georgia",
                "University of California",
            ],
            "person": ["George Bush", None, "Hillary Clinton"],
            "zipcode": ["10145", "N11RG", "56178"],
            "phone number": ["5678-0987", "1234-4321", "0987-1234"],
            "address": [
                "Piazza Gae Aulenti 45",
                "171 Upper Street",
                "29, Russel Square",
            ],
            "url": ["www.levante.com", "www.amazon.it", "www.pandas.org"],
            "credit card number": [
                "5467-9765-0987-0000",
                "1234-5678-9101",
                "0987-6543-2109",
            ],
            "ssn": ["865-50-6891", "042-34-8377", "498-52-4970"],
            "country": ["United Kingdom", "Hungary", "Italy"],
        }
    )


@pytest.fixture
def dict_global_entities(dataset):
    dataset = split_name(dataset, "person")
    recognizer = NamedEntityRecognizer(dataset)
    recognizer.assign_entities_with_presidio()
    recognizer.assign_entities_manually()
    recognizer.assign_organization_entity_with_model()
    return recognizer.dict_global_entities


@pytest.fixture
def instance(dataset, dict_global_entities):
    return FakerGenerator(dataset, dict_global_entities)


def test__init__(instance):
    assert isinstance(instance.dataset, pd.DataFrame)
    assert isinstance(instance.dict_global_entities, Dict)
    assert isinstance(instance.faker, Faker)


def test_get_columns_with_assigned_entity(instance):
    instance.get_columns_with_assigned_entity()
    assert instance.columns_with_assigned_entity == [
        ["email", "EMAIL_ADDRESS"],
        ["city", "LOCATION"],
        ["state", "LOCATION"],
        ["university", "ORGANIZATION"],
        ["zipcode", "ZIPCODE"],
        ["phone number", "PHONE_NUMBER"],
        ["address", "ADDRESS"],
        ["url", "URL"],
        ["credit card number", "CREDIT_CARD_NUMBER"],
        ["ssn", "US_SSN"],
        ["country", "LOCATION"],
        ["first_name", "PERSON"],
        ["last_name", "PERSON"],
    ]
    assert len(instance.columns_not_synthesized) == 0


def test_get_address(instance):
    instance.get_columns_with_assigned_entity()
    instance.get_address()
    assert len(instance.list_faker) > 0
    assert instance.dataset["address"][0] != ""


def test_get_phone_number(instance):
    instance.get_columns_with_assigned_entity()
    instance.get_phone_number()
    assert len(instance.list_faker) > 0
    assert instance.dataset["phone number"][0] != ""


def test_get_email_address(instance):
    instance.get_columns_with_assigned_entity()
    instance.get_email_address()
    assert len(instance.list_faker) > 0
    assert instance.dataset["email"][0] != ""


def test_get_first_name(instance):
    instance.get_columns_with_assigned_entity()
    instance.get_first_name()
    assert len(instance.list_faker) > 0
    assert instance.dataset["first_name"][0] != ""
    assert len(instance.dataset["first_name_gender"]) > 0


def test_get_last_name(instance):
    instance.get_columns_with_assigned_entity()
    instance.get_last_name()
    assert len(instance.list_faker) > 0
    assert instance.dataset["last_name"][0] != ""


def test_get_city(instance):
    instance.get_columns_with_assigned_entity()
    instance.get_city()
    assert len(instance.list_faker) > 0
    assert instance.dataset["city"][0] != ""


def test_get_state(instance):
    instance.get_columns_with_assigned_entity()
    instance.get_state()
    assert len(instance.list_faker) > 0
    assert instance.dataset["state"][0] != ""


def test_get_url(instance):
    instance.get_columns_with_assigned_entity()
    instance.get_url()
    assert len(instance.list_faker) > 0
    assert instance.dataset["url"][0] != ""


def test_get_zipcode(instance):
    instance.get_columns_with_assigned_entity()
    instance.get_zipcode()
    assert len(instance.list_faker) > 0
    assert instance.dataset["zipcode"][0] != ""


def test_get_credit_card(instance):
    instance.columns_with_assigned_entity = [
        ["email", "EMAIL_ADDRESS"],
        ["city", "LOCATION"],
        ["state", "LOCATION"],
        ["university", "ORGANIZATION"],
        ["zipcode", "ZIPCODE"],
        ["phone number", "PHONE_NUMBER"],
        ["address", "ADDRESS"],
        ["url", "URL"],
        ["credit_card", "CREDIT_CARD_NUMBER"],
        ["ssn", "US_SSN"],
        ["country", "LOCATION"],
        ["first_name", "PERSON"],
        ["last_name", "PERSON"],
    ]
    instance.get_credit_card()
    assert len(instance.list_faker) > 0
    assert instance.dataset["credit_card"][0] != ""


def test_get_ssn(instance):
    instance.get_columns_with_assigned_entity()
    instance.get_ssn()
    assert len(instance.list_faker) > 0
    assert instance.dataset["ssn"][0] != ""


def test_get_country(instance):
    instance.get_columns_with_assigned_entity()
    instance.get_country()
    assert len(instance.list_faker) > 0
    assert instance.dataset["country"][0] != ""
