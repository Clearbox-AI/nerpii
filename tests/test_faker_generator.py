from typing import Dict
from unittest.mock import Mock

from faker import Faker
import pandas as pd
import pytest

from nerpii.faker_generator import email_local_part, FakerGenerator
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
            "phone number": ["212-555-0187", "(415) 555-0132", "312-555-0199"],
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
            "first_name_gender": ["female", "unknown", "male"],
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
    names = ["Mary", "David", "Emma"]
    last_names = ["Williams", "Jones", "Clark"]
    instance.get_columns_with_assigned_entity()
    instance.get_email_address(names, last_names)
    for email in instance.dataset["email"]:
        assert "@" in email


def test_get_first_name(instance):
    instance.get_columns_with_assigned_entity()
    instance.get_first_name()
    assert len(instance.list_faker) > 0
    assert instance.dataset["first_name"][0] != ""


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
    instance.get_columns_with_assigned_entity()
    instance.get_credit_card()
    assert len(instance.list_faker) > 0
    assert instance.dataset["credit card number"][0] != ""


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


# The tests below build dict_global_entities by hand, so they don't need the NER
# models.


def entities(**columns):
    return {
        column: {"entity": entity, "confidence_score": 1.0}
        for column, entity in columns.items()
    }


def test_get_faker_generation_without_name_columns():
    df = pd.DataFrame({"city": ["Rome", "Milan"]})
    result = FakerGenerator(df, entities(city="LOCATION")).get_faker_generation()
    assert list(result.columns) == ["city"]
    assert result["city"].notna().all()


def test_input_dataframe_is_not_modified():
    df = pd.DataFrame({"city": ["Rome", "Milan"]})
    generator = FakerGenerator(df, entities(city="LOCATION"))
    generator.get_faker_generation()
    assert list(df["city"]) == ["Rome", "Milan"]
    assert generator.dataset is not df


def test_empty_entities():
    generator = FakerGenerator(pd.DataFrame({"a": [1]}), {})
    generator.get_columns_with_assigned_entity()
    assert generator.columns_with_assigned_entity == []


def test_unscored_entities_are_ignored():
    generator = FakerGenerator(
        pd.DataFrame({"city": ["Rome"], "n": [1]}),
        {"city": ["LOCATION", "LOCATION"], "n": None},
    )
    generator.get_columns_with_assigned_entity()
    assert generator.columns_with_assigned_entity == []


def test_get_first_name_follows_gender_per_row():
    df = pd.DataFrame(
        {
            "first_name": ["Anna", "Marco", "Sam", None],
            "first_name_gender": ["female", "mostly_male", "andy", "Nan value"],
        }
    )
    generator = FakerGenerator(df, entities(first_name="PERSON"))
    generator.faker = Mock()
    generator.faker.first_name_female.return_value = "F"
    generator.faker.first_name_male.return_value = "M"
    generator.faker.first_name.return_value = "U"
    generator.get_columns_with_assigned_entity()
    names = generator.get_first_name()
    assert names[:3] == ["F", "M", "U"]
    assert pd.isna(names[3])
    assert "first_name_gender" not in generator.dataset.columns


def test_all_last_name_columns_are_synthesized():
    df = pd.DataFrame({"last_name": ["Rossi"], "mother_last_name": ["Bianchi"]})
    generator = FakerGenerator(df, {})
    generator.get_last_name()
    assert generator.list_faker == ["last_name", "mother_last_name"]
    assert generator.dataset["last_name"][0] != "Rossi"
    assert generator.dataset["mother_last_name"][0] != "Bianchi"


def test_missing_values_are_kept():
    df = pd.DataFrame({"city": ["Rome", None]})
    generator = FakerGenerator(df, entities(city="LOCATION"))
    generator.get_faker_generation()
    assert generator.dataset["city"][0] != "Rome"
    assert pd.isna(generator.dataset["city"][1])


@pytest.mark.parametrize("mark", ["*", "#"])
def test_generation_mark_replaces_only_marked_cells(mark):
    df = pd.DataFrame(
        {
            "first_name": [mark, "Bob"],
            "last_name": [mark, "Ross"],
            "email": [mark, "bob@ross.com"],
        }
    )
    generator = FakerGenerator(
        df,
        entities(first_name="PERSON", last_name="PERSON", email="EMAIL_ADDRESS"),
        generation_mark=mark,
    )
    result = generator.get_faker_generation()
    assert list(result.iloc[1]) == ["Bob", "Ross", "bob@ross.com"]
    first_name, last_name, email = result.iloc[0]
    assert mark not in (first_name, last_name, email)
    assert email.startswith(
        f"{email_local_part(first_name)}.{email_local_part(last_name)}@"
    )


def test_email_uses_names_with_non_default_index():
    df = pd.DataFrame(
        {
            "first_name": ["Anna", "Marco"],
            "last_name": ["Rossi", "Bianchi"],
            "email": ["anna@rossi.it", "marco@bianchi.it"],
        },
        index=[10, 20],
    )
    generator = FakerGenerator(
        df, entities(first_name="PERSON", last_name="PERSON", email="EMAIL_ADDRESS")
    )
    result = generator.get_faker_generation()
    for _, row in result.iterrows():
        local_part = (
            f"{email_local_part(row['first_name'])}."
            f"{email_local_part(row['last_name'])}"
        )
        assert row["email"].split("@")[0] == local_part


def test_email_without_names():
    df = pd.DataFrame({"email": ["anna@rossi.it"]})
    generator = FakerGenerator(df, entities(email="EMAIL_ADDRESS"))
    generator.get_faker_generation()
    assert "@" in generator.dataset["email"][0]
    assert generator.dataset["email"][0] != "anna@rossi.it"


def test_email_local_part():
    assert email_local_part("De Luca") == "deluca"
    assert email_local_part("Zoë") == "zoe"
    assert email_local_part("O'Brien") == "obrien"
    assert email_local_part("-") == ""
    assert email_local_part(float("nan")) == ""


def test_get_state_with_leading_missing_value():
    df = pd.DataFrame({"state": [None, "TX", "CA"]})
    generator = FakerGenerator(df, entities(state="LOCATION"))
    generator.get_columns_with_assigned_entity()
    generator.get_state()
    assert pd.isna(generator.dataset["state"][0])
    assert all(len(state) == 2 for state in generator.dataset["state"][1:])


def test_numeric_zipcode_column():
    df = pd.DataFrame({"zip": [70116, 48116]})
    generator = FakerGenerator(df, entities(zip="ZIPCODE"))
    generator.get_faker_generation()
    assert all(isinstance(zipcode, str) for zipcode in generator.dataset["zip"])


def test_generation_mark_with_nullable_dtype():
    df = pd.DataFrame({"city": ["*", None, "Rome"]}).convert_dtypes()
    generator = FakerGenerator(df, entities(city="LOCATION"), generation_mark="*")
    generator.get_faker_generation()
    assert generator.dataset["city"][0] != "*"
    assert pd.isna(generator.dataset["city"][1])
    assert generator.dataset["city"][2] == "Rome"
