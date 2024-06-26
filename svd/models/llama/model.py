from .config import LoraLlamaConfig
from transformers import LlamaForCausalLM

from ..module import get_new_module, get_module_by_name



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
    def to_module_map(selected_result: dict):

        module_map = {}

        for name, rank in selected_result.items():
            module_map.update({
                name: {
                    "kwargs": {
                        "rank": rank
                    },
                    "module": "SVDLinear"
                }
            })

        return module_map
                
