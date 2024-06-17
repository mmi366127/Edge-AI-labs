import os
import argparse
from tqdm import tqdm
import click

import torch
import torch.nn as nn
import numpy as np

    
def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


@torch.no_grad()
def get_whiten_scale_matrix_vision(model, calib_loader, args, dev=torch.device("cuda")):
    if not os.path.exists("cache/whiten"):
        os.makedirs("cache/whiten") 
    
    model_id = args.model_id
    cache_file = f"cache/whiten/{model_id.replace('/', '_')}_{args.calib_dataset}_{args.n_calib_samples}_whiten_scaling_matrices_fp16.pt"

    """
    cache format:
    [
        {
            "attn.q_proj": torch.Tensor,
            "attn.k_proj": torch.Tensor,
            "attn.v_proj": torch.Tensor,
            "attn.o_proj": torch.Tensor,
            "mlp.gate_proj": torch.Tensor,
            "mlp.up_proj": torch.Tensor,
            "mlp.down_proj": torch.Tensor
        },
        ... (stacked n times, in the order of model layers)
    ]
    """

    click.secho(f"[whiten] Calibration dataset: {args.calib_dataset}", fg="yellow")
    click.secho(f"[whiten] Search cache_file={cache_file}", fg="yellow")

    if os.path.exists(cache_file) and args.use_cache:
        scaling_matrics = torch.load(cache_file, map_location="cpu")
        
        layers = model.blocks
        for i in tqdm(range(len(layers))):
            layer = layers[i]
            subset = find_layers(layer) # Collect all linear layers
            for name in subset:
                if name in scaling_matrics[i]:
                    scaling_diag_matrix = scaling_matrics[i][name]
                    subset[name].scaling_diag_matrix = scaling_diag_matrix

        return 
    
    click.secho(f"No cache_file={cache_file}", fg="red")
    click.secho(f"Create whiten scale matrix dict...", fg="yellow")

    layers = model.blocks
    model.to(dev)

    dtype = next(iter(model.parameters())).dtype
    click.secho(f"data type: {dtype}", fg="green")

    inps = []
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        
        def forward(self, inp, **kwargs):
            inps.append(inp)
            raise ValueError

    layers[0] = Catcher(layers[0])
    for image, target in calib_loader:
        try:
            model(image.to(dev))
        
        except ValueError:
            pass
    
    layers[0] = layers[0].module
    torch.cuda.empty_cache()
    scaling_matrices = []
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        subset = find_layers(layer)
        def hook(module, input, output):
            inp = input[0].detach().float()
            if inp.dim() == 2:
                inp = inp.unsqueeze(0)
            adds = torch.matmul(inp.transpose(1,2), inp)
            adds_sum = torch.sum(adds, dim=0)
            module.scaling_diag_matrix += adds_sum
            del inp, adds, adds_sum, output
            torch.cuda.empty_cache()
        
        handles = []
        for name in subset:
            subset[name].scaling_diag_matrix = 0
            handles.append(subset[name].register_forward_hook(hook))
        outs = []
        for j in range(len(inps)):
            outs.append(layer(inps[j]))
        for h in handles:
            h.remove()
        layer = layer.cpu()
        for name in subset:
            subset[name].scaling_diag_matrix = subset[name].scaling_diag_matrix.cpu()
        torch.cuda.empty_cache()
        layer_scaling_matrices = {}
        for name in subset:
            W = subset[name].weight.data.float().cuda()
            raw_scaling_diag_matrix = subset[name].scaling_diag_matrix.double().cuda()
            try:
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix).float()
            except Exception as e:
                print("Warning: eigen scaling_diag_matrix is not positive!")
                if torch.isnan(raw_scaling_diag_matrix).any():
                    print("Warning: raw scaling_diag_matrix contains NaN!")
                elif torch.isinf(raw_scaling_diag_matrix).any():
                    print("Warning: raw scaling_diag_matrix contains Inf!")
                if not torch.equal(raw_scaling_diag_matrix, raw_scaling_diag_matrix.T):
                    print("Warning: raw scaling_diag_matrix is not a symmetric matrix!")
                eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-3) * torch.eye(raw_scaling_diag_matrix.shape[0]).cuda()
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix).float()
                if torch.isnan(scaling_diag_matrix).any():
                    print("Warning: scaling_diag_matrix contains NaN!")
                elif torch.isinf(scaling_diag_matrix).any():
                    print("Warning: scaling_diag_matrix contains Inf!")
                del eigenvalues
            try:
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            except Exception as e:
                print("Warning: scaling_diag_matrix is not full rank!")
                reg_inv =  1e-3 * torch.eye(scaling_diag_matrix.shape[0]).cuda() 
                scaling_diag_matrix += reg_inv
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
                del reg_inv
            layer_scaling_matrices[name] = scaling_diag_matrix.to(torch.float16).cpu()
            torch.cuda.empty_cache()
        scaling_matrices.append(layer_scaling_matrices)
        layers[i] = layer.cpu()
        inps = outs
        torch.cuda.empty_cache()

        torch.save(scaling_matrices, cache_file)
        click.secho(f"Save the whiten scale matrix dict to:  {cache_file}", fg="yellow")









@torch.no_grad()
def get_whiten_scale_matrix(model, calib_loader, args, dev=torch.device("cuda")):
    if hasattr(model, "config"):
        model_id = model.config._name_or_path
    else:
        return get_whiten_scale_matrix_vision(model, calib_loader, args, dev)


    if not os.path.exists("cache/whiten"):
        os.makedirs("cache/whiten") 
    
    cache_file = f"cache/whiten/{model_id.replace('/','_')}_{args.calib_dataset}_{args.n_calib_samples}_{args.calib_seqlen}_whiten_scaling_matrices_fp16.pt"
    
    """
    cache format:
    [
        {
            layername: torch.Tensor,
        }    
        ... (stacked n times, in the order of model layers)
    ]
    """

    click.secho(f"[whiten] Calibration dataset: {args.calib_dataset}", fg="yellow")
    click.secho(f"[whiten] Search cache_file={cache_file}", fg="yellow")
    if os.path.exists(cache_file) and args.use_cache:
        click.secho(f"[whiten] File {cache_file} exist.", fg="green")
        click.secho(f"[whiten] Load scaling diag matrix from cache: {cache_file}", fg="yellow")
        scaling_matrics = torch.load(cache_file, map_location="cpu")

        layers = model.model.layers
        for i in tqdm(range(len(layers))):
            layer = layers[i]
            subset = find_layers(layer) # Collect all linear layers
            for name in subset:
                if name in scaling_matrics[i]:
                    scaling_diag_matrix = scaling_matrics[i][name]
                    subset[name].scaling_diag_matrix = scaling_diag_matrix

        return 
    
    click.secho(f"No cache_file={cache_file}", fg="red")
    click.secho(f"Create whiten scale matrix dict...", fg="yellow")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    if "llama" in model_id or "mistral" in model_id or "vicuna" in model_id:
        layers = model.model.layers
    elif "opt" in model_id:
        layers = model.model.decoder.layers
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    click.secho(f"data type: {dtype}", fg="green")
    inps = torch.zeros(
        (len(calib_loader), args.calib_seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            if cache['attention_mask'] is None:
                cache['attention_mask'] = kwargs['attention_mask']
                cache['position_ids'] = kwargs['position_ids']
            else:
                cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask']), dim=0)
                cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids']), dim=0)
            raise ValueError
    
    layers[0] = Catcher(layers[0])
    for batch in calib_loader:
        try:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            model(**batch)
        except ValueError:
            pass
    
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_masks = cache['attention_mask']
    position_ids = cache['position_ids']
    scaling_matrices = []
    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)
        subset = find_layers(layer)
        def hook(module, input, output):
            inp = input[0].detach().float()
            # if "opt" in name:
            if inp.dim() == 2:
                inp = inp.unsqueeze(0)
            adds = torch.matmul(inp.transpose(1,2), inp)
            adds_sum = torch.sum(adds, dim=0)
            module.scaling_diag_matrix += adds_sum
            del inp, adds, adds_sum, output
            torch.cuda.empty_cache()
        handles = []
        for name in subset:
            subset[name].scaling_diag_matrix = 0
            handles.append(subset[name].register_forward_hook(hook))
        for j in range(inps.shape[0]):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_masks, position_ids=position_ids[0].unsqueeze(0))[0]
        for h in handles:
            h.remove()
        layer = layer.cpu()
        for name in subset:
            subset[name].scaling_diag_matrix = subset[name].scaling_diag_matrix.cpu()
        torch.cuda.empty_cache()
        layer_scaling_matrices = {}
        for name in subset:
            W = subset[name].weight.data.float().cuda()
            raw_scaling_diag_matrix = subset[name].scaling_diag_matrix.double().cuda()
            try:
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix).float()
            except Exception as e:
                print("Warning: eigen scaling_diag_matrix is not positive!")
                if torch.isnan(raw_scaling_diag_matrix).any():
                    print("Warning: raw scaling_diag_matrix contains NaN!")
                elif torch.isinf(raw_scaling_diag_matrix).any():
                    print("Warning: raw scaling_diag_matrix contains Inf!")
                if not torch.equal(raw_scaling_diag_matrix, raw_scaling_diag_matrix.T):
                    print("Warning: raw scaling_diag_matrix is not a symmetric matrix!")
                eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-3) * torch.eye(raw_scaling_diag_matrix.shape[0]).cuda()
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix).float()
                if torch.isnan(scaling_diag_matrix).any():
                    print("Warning: scaling_diag_matrix contains NaN!")
                elif torch.isinf(scaling_diag_matrix).any():
                    print("Warning: scaling_diag_matrix contains Inf!")
                del eigenvalues
            try:
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            except Exception as e:
                print("Warning: scaling_diag_matrix is not full rank!")
                reg_inv =  1e-3 * torch.eye(scaling_diag_matrix.shape[0]).cuda() 
                scaling_diag_matrix += reg_inv
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
                del reg_inv
            layer_scaling_matrices[name] = scaling_diag_matrix.to(torch.float16).cpu()
            torch.cuda.empty_cache()
        scaling_matrices.append(layer_scaling_matrices)
        layers[i] = layer.cpu()
        inps = outs
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    
    torch.save(scaling_matrices, cache_file)
    click.secho(f"Save the whiten scale matrix dict to:  {cache_file}", fg="yellow")



