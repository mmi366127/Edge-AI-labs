import torch.nn as nn
import click

def binary_search_truncation_rank(model, sensitivity_dict, args):
    
    module_dict = {}
    full_name_dict = {}
    for name, module in model.named_modules():
        for search_module in args.only_search:
            if search_module in name:
                module_dict.update({name: module})
                full_name_dict.update({module: name})

    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, nn.Linear) and raw_linear in full_name_dict.keys():
                full_name = full_name_dict[raw_linear]
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)

    sensitivity_list = []
    
    for layername, v in sensitivity_dict.items():
        if layername not in module_dict.keys(): continue
        w = module_dict[layername].in_features
        h = module_dict[layername].out_features 
        for ratio, ppl in v.items():
            # Map param ratios in sensitivity_dict to ranks
            rank = int(w * h * ratio) // (w + h)
            sensitivity_list.append((layername, rank, ppl))
            # print(f"{layername} rank={rank} ppl={ppl}")
    sorted_sensitive_list = sorted(sensitivity_list, key=lambda x: -x[2])


    # full_rank = min(kv_weight_shape)
    
    # binary search
    high = len(sorted_sensitive_list) - 1
    low = 0
    # assert args.ppl_target > 0 or args.param_ratio_target > 0

    # input_ids = torch.cat([_["input_ids"] for _ in calib_loader], 0)
    
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()

    while low < high:
        mid = (low + high) // 2
        
        layers_min_rank = {
            layername: min(module_dict[layername].in_features, module_dict[layername].out_features) 
            for layername in module_dict.keys()
        }

        for layername, rank, ppl in sorted_sensitive_list[mid:]:
            layers_min_rank[layername] = min(layers_min_rank[layername], rank)
        
        compressed_params = 0 
        
        for layername, rank in layers_min_rank.items():
            w = module_dict[layername].in_features
            h = module_dict[layername].out_features 
            compressed_params += h * w - (w * rank + h * rank)

        param_ratio = 1 - compressed_params / total_params
        click.secho(f"low={low} mid={mid}, high={high}, param_ratio={param_ratio}, target_param_ratio={args.param_ratio_target}", fg="green")
        
        if param_ratio > args.param_ratio_target:
            high = mid
        else:
            low = mid + 1
    
    # Dump the searching result
    layers_min_rank = {
        layername: min(module_dict[layername].in_features, module_dict[layername].out_features) 
        for layername in module_dict.keys()
    }
    
    for layername, rank, ppl in sorted_sensitive_list[mid:]:
        layers_min_rank[layername] = min(layers_min_rank[layername], rank)
    
    result = {}
    for layername, rank, ppl in sorted_sensitive_list[mid:]:
        rank = layers_min_rank[layername]
        w = module_dict[layername].in_features
        h = module_dict[layername].out_features 
        if w * h > (w * rank + h * rank):
            result.update({layername: rank})

    return result