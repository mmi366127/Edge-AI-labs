from .config import LoraLlamaConfig
from transformers import LlamaForCausalLM

from .. import get_new_module, get_module_by_name

from types import SimpleNamespace


class LoraLlamaForCausalLM(LlamaForCausalLM):
    config_class = LoraLlamaConfig
    def __init__(self, config: LoraLlamaConfig, llama=None):
        super().__init__(config)
        
        self.module_map = config.module_map
        
        model_info = {}
        modules = [(self, None)]
        while len(modules) > 0:
            module, full_name = modules.pop()
            for name, child in module.named_children():
                if full_name is not None:
                    full_child_name = '.'.join([full_name, name])
                else:
                    full_child_name = name
                model_info.update({
                    full_child_name: {
                        'father': module,
                        'module': child,
                        'name': name
                    }
                })
                modules.append((child, full_child_name))
        
        if llama is not None:
            self.load_state_dict(llama.state_dict())

        for name, module in self.named_modules():
            if name in self.module_map.keys():
                info = model_info[name]
                if llama is None:
                    old_module = info["module"]
                else:
                    old_module = get_module_by_name(llama, name)

                new_module = get_new_module(old_module, self.module_map[name])
                setattr(info['father'], info['name'], new_module)
        
    
    @staticmethod
    def from_original(
        llama: LlamaForCausalLM, 
        module_map: dict
    ):
        config = LoraLlamaConfig.from_config(llama.config)
        config.module_map = module_map
        return LoraLlamaForCausalLM(config, llama)
        
        
    @staticmethod
    def get_info(llama: LlamaForCausalLM, num_heads_in_lr_groups: int, module_name: str):


        num_lr_groups = llama.config.num_attention_heads // num_heads_in_lr_groups
        num_lr_kv_groups = llama.config.num_key_value_heads // num_heads_in_lr_groups
        head_dim = llama.config.hidden_size // llama.config.num_attention_heads
        lr_group_dims = head_dim * num_heads_in_lr_groups
        
        if num_lr_groups * num_heads_in_lr_groups != llama.config.num_attention_heads:
            raise ValueError(
                f"num_heads must be divisible by num_heads_in_lr_groups (got `num_heads`: {llama.config.num_attention_heads}"
                f" and `num_heads_in_lr_groups`: {num_heads_in_lr_groups})."
            )
    
        if num_lr_kv_groups * num_heads_in_lr_groups != llama.config.num_key_value_heads:
            raise ValueError(
                f"num_key_value_heads must be divisible by num_heads_in_lr_groups (got `num_key_value_heads`: {llama.condfig.num_key_value_heads}"
                f" and `num_heads_in_lr_groups`: {num_heads_in_lr_groups})."
            )
        
        
        split_in_features = False
        for name in ["k_proj", "v_proj", "q_proj"]:
            if name in module_name:
                split_in_features = True
                break
        
        
        weight_shape = get_module_by_name(llama, module_name).weight.data.shape

        if "k_proj" in module_name or "v_proj" in module_name:
            return SimpleNamespace(
                weight_shape=weight_shape, 
                num_lr_groups=num_lr_kv_groups,
                lr_group_dims=lr_group_dims,
                split_in_features=split_in_features
            )
        else:
            return SimpleNamespace(
                weight_shape=weight_shape, 
                num_lr_groups=num_lr_groups,
                lr_group_dims=lr_group_dims,
                split_in_features=split_in_features
            )
        

    @staticmethod
    def to_module_map(selected_result: dict):

        module_map = {}

        for name, ranks in selected_result.items():

            layer_name, module_name = name.rsplit(".", 1)

            if layer_name in module_map.keys():
                module_map[layer_name]["kwargs"]["module_config"].update({module_name: ranks})
            
            else:
                module_map.update({
                    layer_name: {
                        "module": "LRLlamaAttention", "kwargs": {
                            "module_config" : {
                                module_name: ranks
                            }
                        }
                    }
                })

        return module_map
                
