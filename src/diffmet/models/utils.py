from typing import Type
from . import MODEL_DICT
from . import MODEL_CONFIG_DICT
from .base import Model
from .base import ModelConfig


def find_model_cls(name: str) -> Type[Model]:
    """case-insensitive"""
    return MODEL_DICT[name.lower()]


def build_model(config) -> Model:
    model_cls = find_model_cls(config.name)
    return model_cls.from_config(config)


def find_model_config_cls(name: str) -> Type[ModelConfig]:
    """case-insensitive"""
    return MODEL_CONFIG_DICT[name.lower()]
