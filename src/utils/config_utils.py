import json
import os
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List


def get_config(settings_module: str = "") -> ModuleType:
    """
    Load configs from janitor/config folder using environment variable.

    Arguments:
        settings_module (str): the settings module to load

    Returns:
        [ModuleType]: config module, available to use via `config.<param>`
    """
    if not settings_module:
        settings_module = os.environ["SETTINGS_MODULE"]

    config = import_module(settings_module)
    return config


def get_wd() -> Path:
    """
    Returns path to app.py

    Returns:
        (Path): Path to app.py
    """
    return Path(os.path.realpath(os.path.dirname(__file__))).parent


def get_config_folder() -> Path:
    """
    Returns path to config folder.

    Returns:
        (Path): Path to config folder
    """
    wd = get_wd()
    return Path.joinpath(wd, "config")


def parse_configs(config_names: List[str], config_values: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Parse config files into dictionary form for easier indexing.

    Arguments:
        config_names (List(str)): List of config file names
        config_values (Dict(str, Any)): Config settings

    Returns:
        (Dict): Dictionary of config names and corresponding settings
    """
    return dict(zip(config_names, config_values))


def load_json(filepath: str) -> Dict[str, Any]:
    """Load json file.

    Arguments:
        filepath (str): Path to json file

    Returns:
        (Dict(str, Any)): json values
    """
    with open(filepath, "r") as file:
        return json.load(file)


def load_configs(config_names: List[str]) -> Dict[str, Any]:
    """Load all configs into a dictionary.

    Arguments:
        config_names (List(str)): List of config file names

    Returns:
        (Dict(str, Any)): Dictionary of config names and settings
    """
    config_values = []
    config_folder = get_config_folder()

    for name in config_names:
        config_values.append(load_json(os.path.join(config_folder, f"{name}_config.json")))

    return parse_configs(config_names, config_values)
