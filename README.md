# Nerpii 
Nerpii is a Python library developed to perform Named Entity Recognition (NER) on structured datasets and synthesize Personal Identifiable Information (PII).

NER is performed with [Presidio](https://github.com/microsoft/presidio) and with a [NLP model](https://huggingface.co/dslim/bert-base-NER) available on HuggingFace, while the PII generation is based on [Faker](https://faker.readthedocs.io/en/master/).

## Installation
Nerpii requires Python 3.10 or later. You can install it by using pip:

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
recognizer = NamedEntityRecognizer('./csv_path.csv', lang='en')
```
The <strong>lang</strong> parameter is used to define the language of the dataset. The default value is <strong>en</strong> (english), but it can also be set to <strong>it</strong> (italian).

NER runs on a random sample of rows (500 by default, set with <strong>data_sample</strong>). Pass <strong>random_state</strong> to get the same sample every time. The Presidio analyzer and the Hugging Face model are loaded the first time they are needed, and are then reused.

Please note that if there are columns in the dataset containing names of people consisting of first and last names (e.g. John Smith), before creating a recognizer, it is necessary to split the name into two different columns called <strong>first_name</strong> and <strong>last_name</strong> using the function `split_name()`.

```python
from nerpii.named_entity_recognizer import split_name

df = split_name('./csv_path.csv', name_of_column_to_split)
```
The first word of each name becomes the first name and the rest becomes the last name (e.g. Alexis R. Graves gives Alexis and R. Graves). Missing names stay missing, and the dataframe you pass in is not modified.

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

`recognizer.column_stats` says, for each analyzed column, how many sampled values are not missing (`n_values`), how many have any Presidio detection (`n_detected`) and how many contain each entity (`entity_values`) or have it as their highest-scoring one (`top_entity_values`), plus, after the NLP model ran, how many contain an organization (`organization_values`). Unlike the confidence score, these leave missing values out.

Only text columns are analyzed: object or string columns whose values are strings, numbers or booleans (dates and other objects are left out). Pass `columns` to analyze only some of them. Address words (Street, Via, ...) are matched with their case, so "given via IV" is not an address.

#### Sharing the models and running offline

Loading the spaCy and Hugging Face models takes seconds. To analyze several datasets, build them once and pass them to each recognizer:

```python
from nerpii.named_entity_recognizer import build_model, build_presidio_analyzer

analyzer = build_presidio_analyzer("en", score_threshold=0.4, offline=True)
model = build_model("en", offline=True)
recognizer = NamedEntityRecognizer(df, presidio_analyzer=analyzer, model=model)
```

With `offline=True` (also a `NamedEntityRecognizer` argument) nothing is downloaded: a missing spaCy model or Hugging Face model raises `ModelUnavailable`, whose message says how to install it. `score_threshold` drops Presidio detections scoring below it. `assign_organization_entity_with_model(skip_columns=[...])` keeps columns such as long free text away from the NLP model, which reads each distinct value once, in batches.

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
If you want to generate Italian PII, add ```lang = "it"``` as parameter to the previous object (default: ```lang = "en"```)

First and last names are generated for PERSON columns whose names mark them as name columns: e.g. `first_name`, `FirstName` or `nome` for first names, and `last_name`, `surname` or `cognome` for last names.

To generate first names of the same gender as the original ones, add a gender column with `get_gender()` before creating the generator. The generator drops the column from its output.

```python
from nerpii.named_entity_recognizer import get_gender

dataset = get_gender(dataset)
generator = FakerGenerator(dataset, recognizer.dict_global_entities)
```

To generate new PII you can run

```python
synthetic_dataset = generator.get_faker_generation()
```
The generator works on a copy of the dataset, so the original dataframe is left unchanged. The synthesized dataframe is returned and is also available as `generator.dataset`.

By default every non-null value of a recognized column is replaced. To replace only some values, mark them in the dataset (e.g. with `*`) and pass the mark as `generation_mark`:

```python
generator = FakerGenerator(dataset, recognizer.dict_global_entities, generation_mark="*")
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

City, state, zipcode and country are generated together for each row, so they are consistent with each other. They come from the generator's locale: the country is always United States (or Italy with `lang = "it"`), and for Italian data the state is the province.

## Examples

You can find a notebook example in the [notebook](https://github.com/Clearbox-AI/nerpii/tree/main/notebooks) folder.

## Development

Nerpii uses [uv](https://docs.astral.sh/uv/) to manage dependencies. To install them and run the checks that CI runs:

```bash
uv sync
uv run flake8 nerpii tests
uv run pytest
```

## License

Nerpii is licensed under the [GNU General Public License v3.0](LICENSE).
