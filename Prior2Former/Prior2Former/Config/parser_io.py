import importlib

import argparse
import json
import os
import warnings
from dataclasses import is_dataclass, fields

import yaml
from typing import Dict, Any, _GenericAlias
from typing_extensions import get_origin
from Config.config import (
    update_dict,
    StoreBooleanAction,
    TRUE, FALSE
)


def get_default_config(path): #used by Prior2Former
    """returns the DefaultConfig contained in the same directory as the path file. The DefaultConfig class must be in a file called default_config.py

    Args:
        path (str): yaml config file

    Returns:
        class DefaultConfig: Default Configuration for yaml file
    """

    # Get the directory of the current file
    current_file_directory = os.path.dirname(os.path.abspath(path))

    # Construct the full path to the default_config.py file
    default_config_path = os.path.join(current_file_directory, "default_config.py").replace("yaml", "dataclass")

    # Load the module from the file path
    spec = importlib.util.spec_from_file_location("default_config", default_config_path)
    default_config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(default_config_module)

    # Access the DefaultConfig class
    DefaultConfig = default_config_module.DefaultConfig
    return DefaultConfig

    


def create_file_argument_parser():
    #used by Prior2Former
    parser = argparse.ArgumentParser()
    parser.add_argument(
        f"--config-path",
        type=str,
        required=False,
        help="Optional config loading interface, needs to be either yaml or json",
        nargs='*'
    )
    return parser


def parse_arg_config(args, unknown_args, filter_None=False): #used by Prior2Former
    if len(unknown_args) > 0:
        warnings.warn(f"{unknown_args} could not be parsed")
    parsed_config = vars(args)
    if filter_None:
        parsed_config = {k: v for k, v in parsed_config.items() if v is not None}

    if (
        parsed_config.get("experiment_folder", None) is None
        or parsed_config.get("experiment_folder") == ""
    ):
        parsed_config["experiment_folder"] = os.getenv("MY_POD_NAME","" )
    # Nested experiment read
    config_dict = {}
    for key, value in parsed_config.items():
        if "." in key:
            keys = key.split(".")
            current = config_dict
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[keys[-1]] = value
        else:
            config_dict[key] = value
    # arg_config = {k: v for k, v in parsed_config.items() if v is not None}
    return config_dict


def parse_from_files_no_defaults(config_file_str):
    configs = [parse_from_file_no_defaults(config_file) for config_file in config_file_str]
    ret_config={}
    for conf in configs:
        ret_config = update_dict(ret_config,conf,expand=True)
    return ret_config


def parse_from_file_no_defaults(config_file): #used by Prior2Former
    with open(config_file, "r") as f:
        if "yaml" in config_file or "yml" in config_file:
            parsed_config = yaml.safe_load(f)
        elif "json" in config_file:
            parsed_config = json.load(f)
        else:
            raise NotImplementedError("Config file type not supported")
    return parsed_config


def add_dataclass_to_args(parser, dataclass_obj, prefix=""): #used by Prior2Former
    for field in fields(dataclass_obj):
        if is_dataclass(field.type):
            # if is_dataclass(getattr(dataclass_obj, field.name)):
            add_dataclass_to_args(
                parser,
                # getattr(dataclass_obj, field.name),
                field.type,
                prefix=f"{prefix}{field.name}.",
            )

        elif get_origin(field.type) is dict:
            parser.add_argument(
                f"--{prefix}{field.name}",
                type=json.loads,
                help=f"Description for {field.name}",
            )
        elif isinstance(field.type, _GenericAlias):
            parser.add_argument(
                f"--{prefix}{field.name}",
                type=json.loads,
                help=f"Description for {field.name}",
            )
        elif field.type == bool:
            parser.add_argument(
                f"--{prefix}{field.name}",
                choices=(TRUE, FALSE),
                action=StoreBooleanAction,
                help=f"Description for {field.name}",
            )
        else:
            parser.add_argument(
                f"--{prefix}{field.name}",
                type=field.type,
                help=f"Description for {field.name}",
            )
    return parser


def parse_nested_args(namespace):
    nested_args = {}
    for key, value in vars(namespace).items():
        if "." in key:
            keys = key.split(".")
            d = nested_args
            for k in keys[:-1]:
                if k not in d:
                    d[k] = {}
                d = d[k]
            d[keys[-1]] = value
        else:
            nested_args[key] = value
    return nested_args



def parse_open_seg_config():
    # --config-path
    #used by Prior2Former
    file_args, unknown_args = create_file_argument_parser().parse_known_args()
    return parse_file_cli_hierarchy_seg(file_args.config_path)


def parse_file_cli_hierarchy_seg(config_file) -> Dict:
    #used by Prior2Former
    if isinstance(config_file, list):
        config_file = config_file[0]
    parsed_config = parse_from_file_no_defaults(config_file)
    parser = add_dataclass_to_args(
        argparse.ArgumentParser(), get_default_config(config_file)
    )
    args, unknown_args = parser.parse_known_args()
    # args = argparse.Namespace(**parse_nested_args(args))
    idx = unknown_args.index("--config-path")
    del unknown_args[idx]  # del --config-path
    del unknown_args[idx]  # del Argument
    arg_config = parse_arg_config(args, unknown_args, filter_None=True)
    # Update Defaults with File Config
    default_config = dataclass_to_dict(get_default_config(config_file)())
    experiment_config = update_dict(default_config, parsed_config, expand = True)
    # Update Merged config with cli config
    experiment_config = update_dict(experiment_config, arg_config,  expand = True)

    return experiment_config


def dataclass_to_dict(obj: Any) -> Dict[str, Any]: #used by Prior2Former
    """
    Recursively converts a nested dataclass structure into a dictionary.
    Args:
        obj (Any): The object to convert.
    Returns:
        Dict[str, Any]: The dictionary representation of the input object.
    """
    if is_dataclass(obj):
        return {
            field.name: dataclass_to_dict(getattr(obj, field.name))
            for field in fields(obj)
        }
    elif isinstance(obj, (list, tuple)):
        return [dataclass_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}
    else:
        return obj
