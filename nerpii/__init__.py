from .faker_generator import FakerGenerator
from .named_entity_recognizer import (
    frequency,
    get_gender,
    NamedEntityRecognizer,
    split_name,
)

__all__ = [
    "FakerGenerator",
    "NamedEntityRecognizer",
    "split_name",
    "get_gender",
    "frequency",
]
