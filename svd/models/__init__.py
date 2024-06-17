

from functools import reduce


import torch
import torch.nn as nn

from .llama import LoraLlamaConfig, LoraLlamaForCausalLM

AVAILABLE_MODELS = {
    'llama': {
        'config': LoraLlamaConfig,
        'ModelForCausalLM': LoraLlamaForCausalLM
    }
}
