

from functools import reduce


import torch
import torch.nn as nn

MODULES = {
    "Linear": nn.Linear,
    "SVD_Linear": "",
}

LORA_CONVERT_MAP = {
    "Linear": {
        "SVD_Linear": ""
    }
}

def get_module_key(module):
    if isinstance(module, torch.nn.Linear):
        return 'Linear'
    for module_name, module_type in MODULES.items():
        if isinstance(module, module_type):
            return module_name
    raise ValueError("Unknown module")


def get_new_module(old_module: nn.Module, config: dict):
    new_module_key = config["module"]        
    old_module_key = get_module_key(old_module)
    return LORA_CONVERT_MAP[old_module_key][new_module_key](old_module, **config["kwargs"])


def get_module_by_name(module, module_name):
    names = module_name.split(sep='.')
    return reduce(getattr, names, module)
