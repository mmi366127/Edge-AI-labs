
import json
import argparse
import click

import torch
import torch.nn as nn 

from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.text_data import get_calib_data
from utils.sensitivity import calib_sensitivity_ppl
from utils.eval import eval_ppl
from utils import set_seed, rounding_result
from models import AVAILABLE_MODELS
from models.module import get_module_by_name





def rank_search(model: nn.Module, args, calib_loader):
    click.secho(f"[Rank search] Do rank searching. Search method: {args.search_method}", fg="yellow")

    if args.search_method == "STRS":
        sensitivity = calib_sensitivity_ppl(model, calib_loader, args)

        if args.only_search:
            click.secho(f"[Rank search][Debug] Only search on {args.only_search}", fg="yellow")
            # filter out the layers that are not in the only search list
            layer_names = list(sensitivity.keys())
            new_sensitivity = {}
            for layer in layer_names:
                # Only support the first layer type now
                for layer_type in args.only_search:
                    if layer_type in layer:
                        new_sensitivity[layer] = sensitivity[layer]
            sensitivity = new_sensitivity
            click.secho(f"[Rank search][Debug] Only search on {[name for name in list(sensitivity.keys())]}", fg="yellow")


        if args.search_method == "STRS":
            from utils.binary_search import binary_search_truncation_rank
            select_result = binary_search_truncation_rank(model, sensitivity, args)


    

    select_result = rounding_result(select_result)
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()
    
    compressed_params = 0
    for layername, rank in select_result.items():
        module = get_module_by_name(model, layername)
        w = module.in_features
        h = module.out_features 
        compressed_params += h * w - (w * rank + h * rank)

    return select_result, total_params, total_params - compressed_params








def main(args):
    
    set_seed(args.seed)

    model = AutoModelForCausalLM.from_pretrained(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model.to(torch.bfloat16)
    model.to(torch.device(args.device))


    if args.provide_config is not None:
        try:
            with open(args.provide_config) as f:
                module_map = json.load(f)
        except:
            raise ValueError("Not a valid config file.")
        
    
    else:

        # get calibration data loader 
        calib_loader = get_calib_data(args.calib_dataset, tokenizer, args.model_id, 2048, seqlen=args.calib_seqlen, seed=args.seed)

        # insert information 
        # if "fisher" in args.scaling_method or "fisher" in args.search_metric:
        #     calib_fisher_info(model, calib_loader, torch.device(args.device), args.use_cache)
        
        if "whiten" in args.scaling_method:
            from utils.whiten import get_whiten_scale_matrix
            small_calib_loader = get_calib_data(args.calib_dataset, tokenizer, args.model_id, args.n_calib_samples, seqlen=args.calib_seqlen, seed=args.seed)
            get_whiten_scale_matrix(model, small_calib_loader, args)
        
        # asvd method
        # if "abs" in args.scaling_method:
        #     calib_input_distribution(model, calib_loader, args.scaling_method, torch.device(args.device), args.use_cache)

        select_result, original_params, params = rank_search(model, args, calib_loader)
    

        click.secho(f"compress ratio = {params / original_params}", fg="green")

    
    target_model_class = AVAILABLE_MODELS[model.config.model_type]["ModelForCausalLM"]
    module_map = target_model_class.to_module_map(select_result)
    model_lora = target_model_class.from_original(model, module_map)

    # just delete the old model
    del model

    model_lora.half()
    model_lora.to(torch.device(args.device))

    res = {}

    res_ppl = eval_ppl(model_lora, tokenizer, args.model_id, "wikitext2", device=args.device)
    res.update(res_ppl)

    res_ppl = eval_ppl(model_lora, tokenizer, args.model_id, "c4", device=args.device)
    res.update(res_ppl)

    print(res)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Pretrained model ID"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random Seed"
    )

    parser.add_argument(
        "--save_path", 
        type=str,
        default=None,
        help="Path to save the compressed model."
    )
    
    parser.add_argument(
        "--save_tokenizer", 
        action="store_true",
        help="Whether to save the tokenizer or not."
    )

    parser.add_argument(
        "--use_cache",
        action="store_true",
        help="Whether to use cached calibration results or not.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda"
    )

    # ASVD and Whiten hyper-parameters
    parser.add_argument(
        "--scaling_method",
        type=str,
        default="whiten",
        choices=["whiten", "abs_mean", "abs_max", "fisher", "fisher_abs_mean"],
        help="Scaling method",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="hyper-parameter alpha for ASVD."
    )

    parser.add_argument(
        "--sigma_fuse",
        type=str,
        default="UV",
        help="Sigma fuse method",
        choices=["U", "V", "UV"],
    )

    parser.add_argument(
        "--n_calib_samples",
        type=int,
        default=32,
        help="Number of samples used for calibration.",
    )

    parser.add_argument(
        "--calib_dataset",
        type=str,
        default="wikitext2",
        choices=["wikitext2", "c4", "ptb"],
        help="Calibration dataset",
    )

    parser.add_argument(
        "--calib_seqlen",
        type=int,
        default=1024,
        help="Sequence length of the calibration dataset."
    )


    # Rank Search hyper-paramters
    parser.add_argument(
        "--param_ratio_target", 
        type=float,
        default=-1,
        help="Target param ratio"
    )

    parser.add_argument(
        "--provide_config",
        type=str,
        default=None,
        help="Path to the provided low-rank config json file.",
    )

    parser.add_argument(
        "--only_search",
        nargs="+",
        default=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        help="only search for the specified linear layers",
    )

    parser.add_argument(
        "--search_method",
        type=str,
        default="STRS",
        choices=["STRS"],
        help="Search method",
    )

    parser.add_argument(
        "--search_metric",
        type=str,
        default="svd",
        choices=["svd"],
        help="Determine search metric"
    )

    parser.add_argument(
        "--sensitivity_metric",
        type=str,
        default="ppl",
        choices=["ppl"],
        help="Metric for sensitivity",
    )

    parser.add_argument(
        "--step",
        type=float,
        default=0.1,
        help="step for rank ratio",
    )

    parser.add_argument(
        "--ratio_type",
        type=str,
        default="param_ratio",
        choices=["param_ratio", "rank_ratio"],
        help="Use param_ratio to reduce to overall model size, rank_ratio to reduce KV-cache."
    )

    parser.add_argument(
        "--save_eval_result",
        type=str,
        default=None,
        help="Path to dump the eval result."
    )

    args = parser.parse_args()
    main(args)