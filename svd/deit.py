import os
import math
import time
import json
from tqdm import tqdm
import argparse

from typing import Optional, Tuple, Union
import numpy as np
import click

import torch
import torch.nn as nn


from timm.models import VisionTransformer
from utils.vision_data import get_calib_data
from utils.sensitivity import calib_sensitivity_acc
from utils import set_seed


def get_model(model_id):
    if os.path.exists(model_id):
        return torch.load(model_id)

    return VisionTransformer()


def rank_search(model: nn.Module, args, calib_loader):
    click.secho(f"[Rank search] Do rank searching. Search method: {args.search_method}", fg="yellow")

    if args.search_method == "STRS":

        sensitivity = calib_sensitivity_acc(model, calib_loader, args)

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

def main(args):

    model = get_model(args.model_id)
    print(model)
    if args.provide_config is not None:
        try:
            with open(args.provide_config) as f:
                module_map = json.load(f)
        except:
            raise ValueError("Not a valid config file.")

    else:

        # get calibration data loader 
        calib_loader = get_calib_data(args.calib_dataset, args.n_calib_samples, args.batch_size, seed=args.seed)
        

        if "whiten" in args.scaling_method:
            from utils.whiten import get_whiten_scale_matrix
            get_whiten_scale_matrix(model, calib_loader, args)
        
        module_map, full_rank, rank_sum = rank_search(model, args, calib_loader)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        default="weights/0.9099_deit3_small_patch16_224.pth",
        help="Pretrained model"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="deit"
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
        default=1024,
        help="Number of samples used for calibration.",
    )

    parser.add_argument(
        "--batch_size",
        type=int, 
        default=32,
        help="Batch size for calibration"
    )

    parser.add_argument(
        "--calib_dataset",
        type=str,
        default="cifar100",
        choices=["cifar100"],
        help="Calibration dataset",
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
        default=["qkv", "fc1", "fc2"],
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
        default=0.01,
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

