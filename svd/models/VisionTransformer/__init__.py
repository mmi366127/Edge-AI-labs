
from ..module import get_new_module, get_module_by_name
from  timm.models import VisionTransformer


class LoraVisionTransformer(VisionTransformer):
    def __init__(self, 
            module_map:dict=None,
            visiontransformer=None, *args, **kwargs
    ):
        super().__init__(args, kwargs)

        self.module_map = module_map
        
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
        
        if visiontransformer is not None:
            self.load_state_dict(visiontransformer.state_dict())

        for name, module in self.named_modules():
            if name in self.module_map.keys():
                info = model_info[name]
                if visiontransformer is None:
                    old_module = info["module"]
                else:
                    old_module = get_module_by_name(visiontransformer, name)

                new_module = get_new_module(old_module, self.module_map[name])
                setattr(info['father'], info['name'], new_module)

    
    @staticmethod
    def from_orginal(
        visiontransformer: VisionTransformer,
        module_map: dict
    ):
        kwargs = {}

    