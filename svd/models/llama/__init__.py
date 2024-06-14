from .config import LoraLlamaConfig
from .model import LoraLlamaForCausalLM
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaTokenizer

# register model to hf
AutoConfig.register("lorallama", LoraLlamaConfig)
AutoModelForCausalLM.register(LoraLlamaConfig, LoraLlamaForCausalLM)
AutoTokenizer.register(LoraLlamaConfig, LlamaTokenizer)