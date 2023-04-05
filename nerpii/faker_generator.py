import re
from typing import Any, Dict, List, Union, Optional

from faker import Faker

import numpy as np
import pandas as pd
from simple_colors import green, red


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
    spacy_model : Any
        An english spacy model

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
    generation_mark: str

    def __init__(
        self,
        df_input: Union[str, pd.DataFrame],
        dict_global_entities: Dict,
        generation_mark: Optional[str] = None,
    ) -> "FakerGenerator":
        """
        Create a FakerGenerator instance

        Parameters
        ----------
        df_input : Union[str, pd.DataFrame]
            A pandas dataframe or a path to a csv file.
        dict_global_entities : Dict
            A dictionary whose keys have the same names of the dataframe columns and
            values are dictionaries in which the entity associated to the column and
            its confidence score are reported.

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
        self.generation_mark = generation_mark

    def get_columns_with_assigned_entity(self) -> None:
        """
        Get a list containing those columns with an assigned entity and confidence
        score > 0.3.

        """

        if len(self.dict_global_entities) > 0:
            columns_with_assigned_entity = [
                [i, self.dict_global_entities[i]["entity"]]
                for i in self.dict_global_entities
                if self.dict_global_entities[i] is not None
                and self.dict_global_entities[i]["confidence_score"] > 0.3
            ]
            self.columns_not_synthesized = [
                [i, self.dict_global_entities[i]["entity"]]
                for i in self.dict_global_entities
                if self.dict_global_entities[i] is not None
                and self.dict_global_entities[i]["confidence_score"] <= 0.3
                and not re.match(".*?last.*?name.*?", i.lower())
            ]

        if len(columns_with_assigned_entity) > 0:
            self.columns_with_assigned_entity = columns_with_assigned_entity
        else:
            return print("Impossible to generate Faker data: no assigned entities.")

    def get_address(self) -> None:
        """
        Synthesize address columns in a pandas dataframe

        """

        addresses = [
            i[0] for i in self.columns_with_assigned_entity if i[1] == "ADDRESS"
        ]

        for i in addresses:
            if self.generation_mark == "*":
                self.dataset[i] = self.dataset[i].apply(
                    lambda row: (
                        self.faker.street_address()
                        if row == self.generation_mark
                        else row
                    )
                )
                self.list_faker.append(i)
            else:
                self.dataset[i] = self.dataset[i].apply(
                    lambda row: (
                        self.faker.street_address() if not pd.isnull(row) else np.NaN
                    )
                )

                self.list_faker.append(i)

    def get_phone_number(self) -> None:
        """
        Synthesize phone number columns in a pandas dataframe

        """

        phone_number = [
            i[0] for i in self.columns_with_assigned_entity if i[1] == "PHONE_NUMBER"
        ]

        for i in phone_number:
            if self.generation_mark == "*":
                self.dataset[i] = self.dataset[i].apply(
                    lambda row: (
                        self.faker.phone_number()
                        if row == self.generation_mark
                        else row
                    )
                )
                self.list_faker.append(i)
            else:
                self.dataset[i] = self.dataset[i].apply(
                    lambda row: (
                        self.faker.phone_number() if not pd.isnull(row) else np.NaN
                    )
                )

                self.list_faker.append(i)

    def get_email_address(self) -> None:
        """
        Synthesize email address columns in a pandas dataframe

        """

        email_address = [
            i[0] for i in self.columns_with_assigned_entity if i[1] == "EMAIL_ADDRESS"
        ]

        for i in email_address:
            if self.generation_mark == "*":
                self.dataset[i] = self.dataset[i].apply(
                    lambda row: (
                        self.faker.free_email() if row == self.generation_mark else row
                    )
                )
                self.list_faker.append(i)
            else:
                self.dataset[i] = self.dataset[i].apply(
                    lambda row: (
                        self.faker.free_email() if not pd.isnull(row) else np.NaN
                    )
                )

                self.list_faker.append(i)

    def get_first_name(self) -> None:
        """
        Synthesize first name columns in a pandas dataframe

        """

        first_name_person = [
            i[0]
            for i in self.columns_with_assigned_entity
            if i[1] == "PERSON"
            and (("first" in i[0].lower()) and ("name" in i[0].lower()))
        ]

        for i in first_name_person:
            if self.generation_mark == "*":
                for row in range(0, len(self.dataset[i])):
                    if self.dataset[i][row] == self.generation_mark:
                        if (
                            self.dataset["first_name_gender"][row] == "female"
                            or self.dataset["first_name_gender"][row] == "mostly_female"
                        ):
                            self.dataset[i] = self.dataset[i].apply(
                                lambda row: (self.faker.first_name_female())
                            )
                        elif (
                            self.dataset["first_name_gender"][row] == "male"
                            or self.dataset["first_name_gender"][row] == "mostly_male"
                        ):
                            self.dataset[i] = self.dataset[i].apply(
                                lambda row: (self.faker.first_name_male())
                            )
                        elif (
                            self.dataset["first_name_gender"][row] == "unknown"
                            or self.dataset["first_name_gender"][row] == "andy"
                        ):
                            self.dataset[i] = self.dataset[i].apply(
                                lambda row: (self.faker.first_name())
                            )
                        else:
                            self.dataset[i][row]

                self.list_faker.append(i)
            else:
                for row in range(0, len(self.dataset[i])):
                    if not pd.isnull(self.dataset[i][row]):
                        if (
                            self.dataset["first_name_gender"][row] == "female"
                            or self.dataset["first_name_gender"][row] == "mostly_female"
                        ):
                            self.dataset[i] = self.dataset[i].apply(
                                lambda row: (self.faker.first_name_female())
                            )
                        elif (
                            self.dataset["first_name_gender"][row] == "male"
                            or self.dataset["first_name_gender"][row] == "mostly_male"
                        ):
                            self.dataset[i] = self.dataset[i].apply(
                                lambda row: (self.faker.first_name_male())
                            )
                        elif (
                            self.dataset["first_name_gender"][row] == "unknown"
                            or self.dataset["first_name_gender"][row] == "andy"
                        ):
                            self.dataset[i] = self.dataset[i].apply(
                                lambda row: (self.faker.first_name())
                            )
                        else:
                            self.dataset[i][row] = np.NaN

                self.list_faker.append(i)

        if "first_name_gender" in self.dataset.columns:
            del self.dataset["first_name_gender"]

    def get_last_name(self) -> None:
        """
        Synthesize last name columns in a pandas dataframe

        """

        last_name_person = [
            i[0]
            for i in self.columns_with_assigned_entity
            if i[1] == "PERSON"
            and (("last" in i[0].lower()) and ("name" in i[0].lower()))
        ]

        if len(last_name_person) > 0:
            for i in last_name_person:
                if self.generation_mark == "*":
                    self.dataset[i] = self.dataset[i].apply(
                        lambda row: (self.faker.last_name() if row == "*" else row)
                    )
                    self.list_faker.append(i)

                else:
                    self.dataset[i] = self.dataset[i].apply(
                        lambda row: (
                            self.faker.last_name() if not pd.isnull(row) else np.NaN
                        )
                    )

                    self.list_faker.append(i)

        else:
            last_name_person = [
                i
                for i in self.dataset.columns
                if (("last" in i.lower()) and ("name" in i.lower()))
            ]
            for i in last_name_person:
                if self.generation_mark == "*":
                    self.dataset[i] = self.dataset[i].apply(
                        lambda row: (self.faker.last_name() if row == "*" else row)
                    )
                    self.list_faker.append(i)
                else:
                    self.dataset[i] = self.dataset[i].apply(
                        lambda row: (
                            self.faker.last_name() if not pd.isnull(row) else np.NaN
                        )
                    )

                    self.list_faker.append(i)

    def get_city(self) -> None:
        """
        Synthesize city columns in a pandas dataframe

        """

        city = [
            i[0]
            for i in self.columns_with_assigned_entity
            if i[1] == "LOCATION"
            and (("city" in i[0].lower()) or ("cities" in i[0].lower()))
        ]

        for i in city:
            if self.generation_mark == "*":
                self.dataset[i] = self.dataset[i].apply(
                    lambda row: (
                        self.faker.city() if row == self.generation_mark else row
                    )
                )
                self.list_faker.append(i)
            else:
                self.dataset[i] = self.dataset[i].apply(
                    lambda row: (self.faker.city() if not pd.isnull(row) else np.NaN)
                )

                self.list_faker.append(i)

    def get_state(self) -> None:
        """
        Synthesize state columns in a pandas dataframe

        """

        state = [
            i[0]
            for i in self.columns_with_assigned_entity
            if i[1] == "LOCATION" and ("state" in i[0].lower())
        ]

        for i in state:
            if len(self.dataset[i].iloc[0]) == 2:
                if self.generation_mark == "*":
                    self.dataset[i] = self.dataset[i].apply(
                        lambda row: (
                            self.faker.state_abbr()
                            if row == self.generation_mark
                            else row
                        )
                    )
                    self.list_faker.append(i)
                else:
                    self.dataset[i] = self.dataset[i].apply(
                        lambda row: (
                            self.faker.state_abbr() if not pd.isnull(row) else np.NaN
                        )
                    )

                    self.list_faker.append(i)
            else:
                if self.generation_mark == "*":
                    self.dataset[i] = self.dataset[i].apply(
                        lambda row: (
                            self.faker.state() if row == self.generation_mark else row
                        )
                    )
                    self.list_faker.append(i)
                else:
                    self.dataset[i] = self.dataset[i].apply(
                        lambda row: (
                            self.faker.state() if not pd.isnull(row) else np.NaN
                        )
                    )

                    self.list_faker.append(i)

    def get_url(self) -> None:
        """
        Synthesize url columns in a pandas dataframe

        """

        url = [i[0] for i in self.columns_with_assigned_entity if i[1] == "URL"]

        for i in url:
            if self.generation_mark == "*":
                self.dataset[i] = self.dataset[i].apply(
                    lambda row: (
                        self.faker.url() if row == self.generation_mark else row
                    )
                )
                self.list_faker.append(i)
            else:
                self.dataset[i] = self.dataset[i].apply(
                    lambda row: (self.faker.url() if not pd.isnull(row) else np.NaN)
                )

                self.list_faker.append(i)

    def get_zipcode(self) -> None:
        """
        Synthesize zipcode columns in a pandas dataframe

        """

        zipcode = [i[0] for i in self.columns_with_assigned_entity if i[1] == "ZIPCODE"]

        for i in zipcode:
            if self.generation_mark == "*":
                self.dataset[i] = self.dataset[i].apply(
                    lambda row: (
                        self.faker.zipcode() if row == self.generation_mark else row
                    )
                )
                self.list_faker.append(i)
            else:
                self.dataset[i] = self.dataset[i].apply(
                    lambda row: (self.faker.zipcode() if not pd.isnull(row) else np.NaN)
                )

                self.list_faker.append(i)

    def get_credit_card(self) -> None:
        """
        Synthesize credit card columns in a pandas dataframe

        """

        credit_card = [
            i[0]
            for i in self.columns_with_assigned_entity
            if i[1] == "CREDIT_CARD_NUMBER"
        ]

        for i in credit_card:
            if self.generation_mark == "*":
                self.dataset[i] = self.dataset[i].apply(
                    lambda row: (
                        self.faker.credit_card_number()
                        if row == self.generation_mark
                        else row
                    )
                )
                self.list_faker.append(i)
            else:
                self.dataset[i] = self.dataset[i].apply(
                    lambda row: (
                        self.faker.credit_card_number()
                        if not pd.isnull(row)
                        else np.NaN
                    )
                )

                self.list_faker.append(i)

    def get_ssn(self) -> None:
        """
        Synthesize ssn columns in a pandas dataframe

        """

        ssn = [i[0] for i in self.columns_with_assigned_entity if i[1] == "US_SSN"]

        for i in ssn:
            if self.generation_mark == "*":
                self.dataset[i] = self.dataset[i].apply(
                    lambda row: (
                        self.faker.ssn() if row == self.generation_mark else row
                    )
                )
                self.list_faker.append(i)
            else:
                self.dataset[i] = self.dataset[i].apply(
                    lambda row: (self.faker.ssn() if not pd.isnull(row) else np.NaN)
                )

                self.list_faker.append(i)

    def get_country(self) -> None:
        """
        Synthesize country columns in a pandas dataframe

        """

        country = [
            i[0]
            for i in self.columns_with_assigned_entity
            if i[1] == "LOCATION" and ("country" in i[0].lower())
        ]

        for i in country:
            if self.generation_mark == "*":
                self.dataset[i] = self.dataset[i].apply(
                    lambda row: (
                        self.faker.country() if row == self.generation_mark else row
                    )
                )
                self.list_faker.append(i)
            else:
                self.dataset[i] = self.dataset[i].apply(
                    lambda row: (self.faker.country() if not pd.isnull(row) else np.NaN)
                )

                self.list_faker.append(i)

    def get_columns_not_synthesized(self) -> None:
        """
        Get a list of all non-synthesized columns.

        """

        for i in self.columns_with_assigned_entity:
            if i[0] not in self.list_faker:
                self.columns_not_synthesized.append(i)

    def synthesis_message(self) -> None:
        """
        Get a message with synthesized and unsynthesized columns.

        """

        for col in self.list_faker:
            message = "Column " + green(col, "bold") + " synthesized with Faker."
            print(message)

        for col in self.columns_not_synthesized:
            message = "Column " + red(col[0], "bold") + " not synthesized with Faker."
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
