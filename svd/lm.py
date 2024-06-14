
import json
import argparse


import torch
import torch.nn as nn 

from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.text_data import get_calib_data
from utils import set_seed

def rank_search(model: nn.Module, args, calib_loader):
    pass

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
        calib_loader = get_calib_data(args.calib_dataset, tokenizer, args.model_id, 2048, seqlen=args.calib_seqlen)

        if "whiten" in args.scaling_method:
            from utils.whiten import get_whiten_scale_matrix
            get_whiten_scale_matrix(model, calib_loader, args)
        

        module_map, full_rank, rank_sum = rank_search(model, args, calib_loader)
    
    



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