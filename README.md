# Nerpii 
Nerpii is a Python library developed to perform Named Entity Recognition (NER) on structured datasets and synthesize Personal Identifiable Information (PII).

NER is performed with [Presidio](https://github.com/microsoft/presidio) and with a [NLP model](https://huggingface.co/dslim/bert-base-NER) available on HuggingFace, while the PII generation is based on [Faker](https://faker.readthedocs.io/en/master/).

## Installation
You can install Nerpii by using pip: 

```python
pip install nerpii
```
## Quickstart
### Named Entity Recognition
You can import the NamedEntityRecognizer using
```python
from nerpii.named_entity_recognizer import NamedEntityRecognizer
```
You can create a recognizer passing as parameter a path to a csv file or a Pandas Dataframe

```python
recognizer = NamedEntityRecognizer('./csv_path.csv')
```
Please note that if there are columns in the dataset containing names of people consisting of first and last names (e.g. John Smith), before creating a recognizer, it is necessary to split the name into two different columns called <strong>first_name</strong> and <strong>last_name</strong> using the function `split_name()`.

```python
from nerpii.named_entity_recognizer import split_name

df = split_name('./csv_path.csv', name_of_column_to_split)
```
The NamedEntityRecognizer class contains three methods to perform NER on a dataset:

```python
recognizer.assign_entities_with_presidio()
```
which assigns Presidio entities, listed [here](https://microsoft.github.io/presidio/supported_entities/)

```python
recognizer.assign_entities_manually()
```
which assigns manually ZIPCODE and CREDIT_CARD_NUMBER entities 

```python
recognizer.assign_organization_entity_with_model()
```
which assigns ORGANIZATION entity using a [NLP model](https://huggingface.co/dslim/bert-base-NER) available on HuggingFace.

To perform NER, you have to run these three methods sequentially, as reported below:

```python
recognizer.assign_entities_with_presidio()
recognizer.assign_entities_manually()
recognizer.assign_organization_entity_with_model()
```

The final output is a dictionary in which column names are given as keys and assigned entities and a confidence score as values.

This dictionary can be accessed using

```python
recognizer.dict_global_entities
```

### PII generation 

After performing NER on a dataset, you can generate new PII using Faker. 

You can import the FakerGenerator using 

```python
from nerpii.faker_generator import FakerGenerator
```

You can create a generator using

```python
generator = FakerGenerator(dataset, recognizer.dict_global_entities)
```
To generate new PII you can run

```python
generator.get_faker_generation()
```
The method above can generate the following PII:
* address
* phone number
* email naddress
* first name
* last name
* city
* state
* url
* zipcode
* credit card
* ssn
* country

## Examples

You can find a notebook example in the [notebook](https://github.com/Clearbox-AI/nerpii/tree/main/notebooks) folder. 


