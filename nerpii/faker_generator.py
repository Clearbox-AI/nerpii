import re
from typing import Any, Callable, Dict, List, Optional, Union
import unicodedata

from faker import Faker
import numpy as np
import pandas as pd

GENDER_COLUMN = "first_name_gender"


def is_first_name_column(column: str) -> bool:
    column = column.lower()
    return "first" in column and "name" in column


def is_last_name_column(column: str) -> bool:
    column = column.lower()
    return "last" in column and "name" in column


def email_local_part(name: Any) -> str:
    """
    Reduce a name to lowercase ASCII letters and digits, so that it can be used in
    an email address ("De Luca" -> "deluca", "Zoë" -> "zoe").

    Returns an empty string if the name is missing or has no usable characters.
    """
    if not isinstance(name, str):
        return ""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore")
    return re.sub(r"[^a-z0-9]", "", ascii_name.decode().lower())


class FakerGenerator:
    """
    A class used to generate faker objects in a dataframe

    Attributes
    -------
    dataset : pd.DataFrame
        A copy of the input dataframe, in which columns are synthesized
    dict_global_entities : Dict
        A dictionary whose keys have the same names of the dataframe columns and values
        are dictionaries in which the entity associated to the column and its confidence
        score are reported.
    faker : Any
        A generator to obtain synthetisized objects
    columns_with_assigned_entity : List
        A list of columns with an assigned entity
    columns_not_synthesized : List
        A list of those columns which are not synthesized by faker
    list_faker : List
        A list of those columns which are synthesized by faker
    lang : str
        A string to set italian as language of generation. Default: en
    generation_mark : Any
        If set, only cells equal to this value (e.g. "*") are synthesized.
        If None, every non-null cell of a synthesized column is replaced.

    City, state, zipcode and country columns are generated from a single location per
    row, taken from the generator's locale, so that they are consistent with each
    other: the country is always the locale's country (United States or Italy).
    """

    dataset: pd.DataFrame
    dict_global_entities: Dict
    faker: Any
    columns_with_assigned_entity: List
    columns_not_synthesized: List
    list_faker: List
    lang: str
    generation_mark: Any

    def __init__(
        self,
        df_input: Union[str, pd.DataFrame],
        dict_global_entities: Dict,
        lang: str = "en",
        generation_mark: Optional[Any] = None,
    ) -> "FakerGenerator":
        """
        Create a FakerGenerator instance

        Parameters
        ----------
        df_input : Union[str, pd.DataFrame]
            A pandas dataframe or a path to a csv file. The dataframe is copied, so
            the synthesized data is found in the `dataset` attribute.
        dict_global_entities : Dict
            A dictionary whose keys have the same names of the dataframe columns and
            values are dictionaries in which the entity associated to the column and
            its confidence score are reported.
        lang : str, optional
            Input language by default "en". Set this parameter to "it" to generate
            Italian data.
        generation_mark : Optional[Any], optional
            If set, only cells equal to this value are synthesized, by default None

        Returns
        -------
        FakerGenerator
            A Fakergenerator instance.
        """

        if not isinstance(df_input, pd.DataFrame):
            df_input = pd.read_csv(df_input)

        self.dataset = df_input.copy()
        self.dict_global_entities = dict_global_entities
        self.lang = lang
        if self.lang == "it":
            self.faker = Faker(["it_IT"])
        else:
            self.faker = Faker()
        self.columns_with_assigned_entity = []
        self.columns_not_synthesized = []
        self.list_faker = []
        self.generation_mark = generation_mark
        self._row_locations = {}
        self._places = None

    def _columns_with_entity(
        self, entity: str, name_filter: Callable[[str], bool] = lambda column: True
    ) -> List[str]:
        """
        Return the columns with the given assigned entity whose name passes the filter.
        """
        return [
            column
            for column, column_entity in self.columns_with_assigned_entity
            if column_entity == entity and name_filter(column)
        ]

    def _synthesize(self, column: str, make_value: Callable[[int], Any]) -> None:
        """
        Replace the cells of a column with make_value(row_position).

        If generation_mark is set, only cells equal to it are replaced; otherwise every
        non-null cell is replaced and missing values are kept.
        """
        original = self.dataset[column]
        if self.generation_mark is None:
            to_replace = original.notna().to_numpy()
        else:
            # Nullable dtypes compare missing values to <NA>, which are not replaced
            to_replace = (original == self.generation_mark).to_numpy(
                dtype=bool, na_value=False
            )

        values = original.to_numpy(dtype=object, copy=True)
        for position in np.flatnonzero(to_replace):
            values[position] = make_value(position)

        self.dataset[column] = values
        if column not in self.list_faker:
            self.list_faker.append(column)

    def _locale_places(self) -> List[Dict[str, str]]:
        """
        Return the places of the generator's locale as dictionaries with consistent
        city (None if the locale has no list of real cities), state, state_abbr and
        zipcode (None if it has to be generated from the state).
        """
        if self._places is None:
            address = self.faker.provider("faker.providers.address")
            if self.lang == "it":
                provinces = dict(zip(address.states_abbr, address.states, strict=True))
                # Skip provinces missing from the list of states (abolished ones)
                self._places = [
                    {
                        "city": city,
                        "state": provinces[province],
                        "state_abbr": province,
                        "zipcode": zipcode,
                    }
                    for zipcode, places in address.cap_city_province.items()
                    for city, province in places
                    if province in provinces
                ]
            else:
                # states_abbr also lists DC, which is not in states
                abbreviations = [abbr for abbr in address.states_abbr if abbr != "DC"]
                self._places = [
                    {"city": None, "state": state, "state_abbr": abbr, "zipcode": None}
                    for state, abbr in zip(address.states, abbreviations, strict=True)
                ]
        return self._places

    def _location(self, position: int) -> Dict[str, str]:
        """
        Return the location of a row: a city, state, zipcode and country that belong
        together. The same row always gets the same location.
        """
        if position not in self._row_locations:
            location = dict(self.faker.random_element(self._locale_places()))
            if location["city"] is None:
                location["city"] = self.faker.city()
            if location["zipcode"] is None:
                location["zipcode"] = self.faker.zipcode_in_state(
                    location["state_abbr"]
                )
            location["country"] = self.faker.current_country()
            location["country_code"] = self.faker.current_country_code()
            self._row_locations[position] = location
        return self._row_locations[position]

    def _synthesize_location(self, column: str, field: str) -> None:
        """
        Replace the cells of a column with a field of the rows' locations.
        """
        self._synthesize(column, lambda position: self._location(position)[field])

    def _is_abbreviated(self, column: str) -> bool:
        """
        Whether the first non-null value of a column is two characters long.
        """
        values = self.dataset[column].dropna()
        return len(values) > 0 and len(str(values.iloc[0])) == 2

    def get_columns_with_assigned_entity(self) -> None:
        """
        Get a list containing those columns with an assigned entity and confidence
        score > 0.3.

        """

        # Columns without an entity are None, or still a list of entities if the
        # entities have not been scored yet
        scored = {
            column: entity
            for column, entity in self.dict_global_entities.items()
            if isinstance(entity, dict)
        }
        self.columns_with_assigned_entity = [
            [column, entity["entity"]]
            for column, entity in scored.items()
            if entity["confidence_score"] > 0.3
        ]
        self.columns_not_synthesized = [
            [column, entity["entity"]]
            for column, entity in scored.items()
            if entity["confidence_score"] <= 0.3
            and not re.match(".*?last.*?name.*?", column.lower())
        ]

        if not self.columns_with_assigned_entity:
            print("Impossible to generate Faker data: no assigned entities.")

    def get_address(self) -> None:
        """
        Synthesize address columns in a pandas dataframe

        """

        addresses = [
            column
            for column, entity in self.columns_with_assigned_entity
            if entity == "ADDRESS"
            or "indirizzo" in column.lower()
            or (entity == "LOCATION" and "address" in column.lower())
        ]

        for column in addresses:
            self._synthesize(column, lambda _: self.faker.street_address())

    def get_phone_number(self) -> None:
        """
        Synthesize phone number columns in a pandas dataframe

        """

        for column in self._columns_with_entity("PHONE_NUMBER"):
            self._synthesize(column, lambda _: self.faker.phone_number())

    def _first_name(self, gender: Any) -> str:
        """
        Return a first name matching a gender returned by gender_guesser.
        """
        if gender in ("female", "mostly_female"):
            return self.faker.first_name_female()
        if gender in ("male", "mostly_male"):
            return self.faker.first_name_male()
        return self.faker.first_name()

    def get_first_name(self) -> Optional[List]:
        """
        Synthesize first name columns in a pandas dataframe.

        If the dataframe has a first_name_gender column (see get_gender), names are
        generated with the gender of the original name and the column is dropped.

        Returns
        -------
        Optional[List]
            The values of the first synthesized column, None if there is none.
        """
        first_names = self._columns_with_entity("PERSON", is_first_name_column)

        if GENDER_COLUMN in self.dataset.columns:
            genders = self.dataset[GENDER_COLUMN].to_numpy()
            for column in first_names:
                self._synthesize(
                    column, lambda position: self._first_name(genders[position])
                )
            self.dataset = self.dataset.drop(columns=GENDER_COLUMN)
        else:
            for column in first_names:
                self._synthesize(column, lambda _: self.faker.first_name())

        return list(self.dataset[first_names[0]]) if first_names else None

    def get_last_name(self) -> Optional[List]:
        """
        Synthesize last name columns in a pandas dataframe

        Returns
        -------
        Optional[List]
            The values of the first synthesized column, None if there is none.
        """

        last_names = self._columns_with_entity("PERSON", is_last_name_column)
        if not last_names:
            # Last names are often not recognized as PERSON, so fall back to the
            # column names
            last_names = [
                column for column in self.dataset.columns if is_last_name_column(column)
            ]

        for column in last_names:
            self._synthesize(column, lambda _: self.faker.last_name())

        return list(self.dataset[last_names[0]]) if last_names else None

    def _email_address(
        self, names: Optional[List], last_names: Optional[List], position: int
    ) -> str:
        """
        Return a "name.last_name@domain" email address for a row, or a random email
        address if the row has no usable name and last name.
        """
        name = email_local_part(names[position]) if names is not None else ""
        last_name = (
            email_local_part(last_names[position]) if last_names is not None else ""
        )
        if name and last_name:
            return f"{name}.{last_name}@{self.faker.free_email_domain()}"
        return self.faker.free_email()

    def get_email_address(
        self, names: Optional[List] = None, last_names: Optional[List] = None
    ) -> None:
        """
        Synthesize email address columns in a pandas dataframe

        Parameters
        ----------
        names : Optional[List], optional
            First names to build the addresses from, one per row, by default None
        last_names : Optional[List], optional
            Last names to build the addresses from, one per row, by default None
        """

        for column in self._columns_with_entity("EMAIL_ADDRESS"):
            self._synthesize(
                column,
                lambda position: self._email_address(names, last_names, position),
            )

    def get_city(self) -> None:
        """
        Synthesize city columns in a pandas dataframe

        """

        cities = self._columns_with_entity(
            "LOCATION",
            lambda column: "city" in column.lower() or "cities" in column.lower(),
        )
        for column in cities:
            self._synthesize_location(column, "city")

    def get_state(self) -> None:
        """
        Synthesize state columns in a pandas dataframe. States are abbreviated if
        the first value of the column is two characters long. Italian states are
        provinces.

        """

        states = self._columns_with_entity(
            "LOCATION", lambda column: "state" in column.lower()
        )
        for column in states:
            if self._is_abbreviated(column):
                self._synthesize_location(column, "state_abbr")
            else:
                self._synthesize_location(column, "state")

    def get_url(self) -> None:
        """
        Synthesize url columns in a pandas dataframe

        """

        for column in self._columns_with_entity("URL"):
            self._synthesize(column, lambda _: self.faker.url())

    def get_zipcode(self) -> None:
        """
        Synthesize zipcode columns in a pandas dataframe

        """

        for column in self._columns_with_entity("ZIPCODE"):
            self._synthesize_location(column, "zipcode")

    def get_credit_card(self) -> None:
        """
        Synthesize credit card columns in a pandas dataframe

        """

        for column in self._columns_with_entity("CREDIT_CARD_NUMBER"):
            self._synthesize(column, lambda _: self.faker.credit_card_number())

    def get_ssn(self) -> None:
        """
        Synthesize ssn columns in a pandas dataframe

        """

        for column in self._columns_with_entity("US_SSN"):
            self._synthesize(column, lambda _: self.faker.ssn())

    def get_country(self) -> None:
        """
        Synthesize country columns in a pandas dataframe with the locale's country.
        Countries are written as two-letter codes if the first value of the column is
        two characters long.

        """

        countries = self._columns_with_entity(
            "LOCATION", lambda column: "country" in column.lower()
        )
        for column in countries:
            if self._is_abbreviated(column):
                self._synthesize_location(column, "country_code")
            else:
                self._synthesize_location(column, "country")

    def get_columns_not_synthesized(self) -> None:
        """
        Get a list of all non-synthesized columns.

        """

        for column in self.columns_with_assigned_entity:
            if (
                column[0] not in self.list_faker
                and column not in self.columns_not_synthesized
            ):
                self.columns_not_synthesized.append(column)

    def synthesis_message(self) -> None:
        """
        Get a message with synthesized and unsynthesized columns.

        """

        for column in self.list_faker:
            print(f"Column {column} synthesized with Faker.")

        for column in self.columns_not_synthesized:
            print(f"Column {column[0]} not synthesized with Faker.")

    def get_faker_generation(self) -> pd.DataFrame:
        """
        Get faker objects for columns in a pandas dataframe

        Returns
        -------
        pd.DataFrame
            The synthesized dataframe, also available as the `dataset` attribute.
        """
        self.get_columns_with_assigned_entity()
        self.get_address()
        self.get_phone_number()
        name = self.get_first_name()
        last_name = self.get_last_name()
        self.get_email_address(name, last_name)
        self.get_city()
        self.get_state()
        self.get_url()
        self.get_zipcode()
        self.get_credit_card()
        self.get_ssn()
        self.get_country()

        self.get_columns_not_synthesized()

        self.synthesis_message()

        return self.dataset
