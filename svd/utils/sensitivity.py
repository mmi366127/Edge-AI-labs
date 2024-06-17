import os
import torch
import torch.nn as nn
from models.module import SVDLinear
from .eval import evaluate_perplexity, eval_accuracy
from tqdm import tqdm
import numpy as np
import click

from models.module import SVDLinear

@torch.no_grad()
def calib_sensitivity_ppl(model, calib_loader, args):
    step = args.step
    use_cache = args.use_cache
    model_id = model.config._name_or_path

    cache_file = f"cache/lists/{model_id.replace('/','_')}_sensitivity_{args.scaling_method}_{args.alpha}_{args.n_calib_samples}_{args.calib_dataset}_step_{args.step}.pt"
    click.secho(f"Search cache_file={cache_file}", fg="yellow")
    
    if not os.path.exists("cache/lists"):
        os.makedirs("cache/lists")
    
    if os.path.exists(cache_file) and use_cache:
        click.secho(f"Load cache_file={cache_file}", fg="green")
        sensitivity_dict = torch.load(cache_file, map_location="cpu")
        return sensitivity_dict
    model.eval()
    
    click.secho(f"No cache_file={cache_file}", fg="red")

    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)

    sensitivity_dict = {}
    # generate a list in range 0 to 1 with step 
    param_ratio_candidates = np.arange(step, 1.0, step=step).tolist()
    # Round to 2 decimal places
    param_ratio_candidates = [round(_, 2) for _ in param_ratio_candidates]
    
    input_ids = torch.cat([_["input_ids"] for _ in calib_loader], 0)
    print(f"input_ids.shape={input_ids.shape}")
    pbar = tqdm(total=len(linear_info) * len(param_ratio_candidates))
    for raw_linear, info in linear_info.items():
        if info["full_name"] == "lm_head":
            continue
        sensitivity_dict[info["full_name"]] = {}
        for param_ratio in param_ratio_candidates:
            n_params = raw_linear.weight.numel()
            compressed_params = int(n_params * param_ratio)
            rank = compressed_params // (raw_linear.in_features + raw_linear.out_features)
            svd_linear = SVDLinear.from_Linear(
                raw_linear, 
                rank
            )
            setattr(info["father"], info["name"], svd_linear)

            ppl = evaluate_perplexity(model, input_ids, args.n_calib_samples)
            sensitivity_dict[info["full_name"]][param_ratio] = ppl
            print(f"{info['full_name']} {param_ratio} {ppl}")
            pbar.update(1)
        setattr(info["father"], info["name"], raw_linear)
    torch.save(sensitivity_dict, cache_file)
    return sensitivity_dict


@torch.no_grad()
def calib_sensitivity_acc(model, calib_loader, args):
    # for vision model
    step = args.step
    use_cache = args.use_cache
    model_id = args.model_id

    cache_file = f"cache/lists/{model_id.replace('/','_')}_sensitivity_{args.scaling_method}_{args.n_calib_samples}_{args.calib_dataset}_step_{args.step}.pt"
    click.secho(f"Search cache_file={cache_file}", fg="yellow")
    
    if not os.path.exists("cache/lists"):
        os.makedirs("cache/lists")
    
    if os.path.exists(cache_file) and use_cache:
        click.secho(f"Load cache_file={cache_file}", fg="green")
        sensitivity_dict = torch.load(cache_file, map_location="cpu")
        return sensitivity_dict
    model.eval()
    
    click.secho(f"No cache_file={cache_file}", fg="red")

    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]
                if hasattr(raw_linear, "scaling_diag_matrix"):
                    linear_info[raw_linear] = {
                        "father": submodule,
                        "name": name,
                        "full_name": full_name,
                    }
            else:
                modules.append(raw_linear)

    sensitivity_dict = {}
    # generate a list in range 0 to 1 with step 
    param_ratio_candidates = np.arange(step, 1.0, step=step).tolist()
    # Round to 2 decimal places
    param_ratio_candidates = [round(_, 2) for _ in param_ratio_candidates]
    
    pbar = tqdm(total=len(linear_info) * len(param_ratio_candidates))
    for raw_linear, info in linear_info.items():
        if info["full_name"] == "lm_head":
            continue
        sensitivity_dict[info["full_name"]] = {}
        for param_ratio in param_ratio_candidates:
            n_params = raw_linear.weight.numel()
            compressed_params = int(n_params * param_ratio)
            rank = compressed_params // (raw_linear.in_features + raw_linear.out_features)
            svd_linear = SVDLinear.from_Linear(
                raw_linear, 
                rank
            )
            setattr(info["father"], info["name"], svd_linear)

            acc = eval_accuracy(model, calib_loader)
            sensitivity_dict[info["full_name"]][param_ratio] = acc
            click.secho(f"{info['full_name']} {param_ratio} {acc}", fg="yellow")
            pbar.update(1)
        setattr(info["father"], info["name"], raw_linear)
    torch.save(sensitivity_dict, cache_file)
    return sensitivity_dict