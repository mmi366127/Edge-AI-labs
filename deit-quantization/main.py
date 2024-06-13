

import os
import math
import time
import json
from tqdm import tqdm
import argparse

from typing import Optional, Tuple, Union
import numpy as np

import torch
import torch.nn as nn

from scipy.stats import kurtosis

from torch.export import export, ExportedProgram, dynamic_dim, Dim
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e, prepare_qat_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)


from utils import save_model
from utils.eval import lab4_cifar100_evaluation
from utils.data import data_loader_to_list
from utils.quantizer import PartialXNNPACKQuantizer, replace_linear, new_linear
from utils.train import replace_activation_and_train
from utils.data import getMiniTestDataset, prepare_data




# profiling utils
def torch_profile(model, input_data, device):
    ## With warmup and skip
    # https://pytorch.org/docs/stable/profiler.html

    # Non-default profiler schedule allows user to turn profiler on and off
    # on different iterations of the training loop;
    # trace_handler is called every time a new trace becomes available
    def trace_handler(prof):
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
        prof.export_chrome_trace("./test_trace_" + str(prof.step_num) + ".json")

    with torch.profiler.profile(
        activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
        ],

        # In this example with wait=1, warmup=1, active=2, repeat=1,
        # profiler will skip the first step/iteration,
        # start warming up on the second, record
        # the third and the forth iterations,
        # after which the trace will become available
        # and on_trace_ready (when set) is called;
        # the cycle repeats starting with the next step
        schedule = torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
        on_trace_ready = trace_handler
        # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
        # used when outputting for tensorboard
    ) as p:
        for data in input_data:
            model(data.to(device))
            # send a signal to the profiler that the next iteration has started
            p.step()


def prepare_model(model, data_loader, device):

    _dummy_input_data = (next(iter(data_loader))[0].to(device),)

    model.eval()
    model.to(device)
    dynamic_shapes = {"x": {0: Dim("batch")}}
    model = capture_pre_autograd_graph(
        model, 
        _dummy_input_data,
        dynamic_shapes=dynamic_shapes
    )

    return model


def simple_quantize(model: nn.Module, data_loader, args, per_channel=False) -> None:
    
    device = torch.device(args.device)
    model = prepare_model(model, data_loader, device)

    quantizer = XNNPACKQuantizer()
    quantization_config = get_symmetric_quantization_config(is_per_channel=per_channel, is_qat=False)
    quantizer.set_global(quantization_config)
    # prepare_pt2e folds BatchNorm operators into preceding Conv2d operators, and inserts observers in appropriate places in the model.
    model = prepare_pt2e(model, quantizer)

    num_batches = args.num_samples // args.batch_size
    num_batches = min(num_batches, len(data_loader))
    dataloader = data_loader_to_list(data_loader, num_batches)

    for image, _ in tqdm(dataloader):
        model(image.to(device))
        
    quantized_model = convert_pt2e(model, use_reference_representation=False)
    torch.ao.quantization.move_exported_model_to_eval(quantized_model)

    return quantized_model


# my quantizer
def quantize_all_linear(model, data_loader, args, per_channel=False):
    
    device = torch.device(args.device)
    model.to(device)

    ignore_list = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            continue
        ignore_list.append(name)

    print(f"Ignore list: {ignore_list}")

    model = prepare_model(model, data_loader, device)

    
    quantizer = PartialXNNPACKQuantizer(ignore_list=ignore_list)
    quantization_config = get_symmetric_quantization_config(is_per_channel=per_channel, is_qat=False)
    quantizer.set_global(quantization_config)
    # prepare_pt2e folds BatchNorm operators into preceding Conv2d operators, and inserts observers in appropriate places in the model.
    model = prepare_pt2e(model, quantizer)
    
    num_batches = args.num_samples // args.batch_size
    num_batches = min(num_batches, len(data_loader))
    dataloader = data_loader_to_list(data_loader, num_batches)

    # with torch.autocast(device_type="cuda"):
    for image, _ in tqdm(dataloader):
        model(image.to(device))

    
    quantized_model = convert_pt2e(model, use_reference_representation=False)
    torch.ao.quantization.move_exported_model_to_eval(quantized_model)

    return quantized_model


def quantize_all_matrix(model, data_loader, args, per_channel=False):
    
    device = torch.device(args.device)
    model.to(device)

    model = replace_linear(model)

    ignore_list = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            continue
        ignore_list.append(name)

    print(f"Ignore list: {ignore_list}")

    model = prepare_model(model, data_loader, device)

    
    quantizer = PartialXNNPACKQuantizer(ignore_list=ignore_list)
    quantization_config = get_symmetric_quantization_config(is_per_channel=per_channel, is_qat=False)
    quantizer.set_global(quantization_config)
    # prepare_pt2e folds BatchNorm operators into preceding Conv2d operators, and inserts observers in appropriate places in the model.
    model = prepare_pt2e(model, quantizer)
    
    num_batches = args.num_samples // args.batch_size
    num_batches = min(num_batches, len(data_loader))
    dataloader = data_loader_to_list(data_loader, num_batches)

    # with torch.autocast(device_type="cuda"):
    for image, _ in tqdm(dataloader):
        model(image.to(device))

    
    quantized_model = convert_pt2e(model, use_reference_representation=False)
    torch.ao.quantization.move_exported_model_to_eval(quantized_model)

    return quantized_model


def get_kurtosis(model: nn.Module, args):
    
    cache_file = f"klist.json"
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            klist = json.load(f)
        return klist
    
    k_list = {}
    for name, param in model.named_parameters():
        k = kurtosis(param.detach().view(-1).cpu().numpy())
        k_list.update({name: k})
    
    return k_list


def quantize_by_kurtosis(model, data_loader, args, per_channel=False):
    device = torch.device(args.device)
    model.to(device)

    klist = get_kurtosis(model, args)


    model = replace_linear(model)

    ignore_list = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and klist[name.rsplit(".", 1)[0] + ".weight"] < args.kurtosis_thres:
            continue
        ignore_list.append(name)

    print(f"Ignore list: {ignore_list}")

    model = prepare_model(model, data_loader, device)

    
    quantizer = PartialXNNPACKQuantizer(ignore_list=ignore_list)
    quantization_config = get_symmetric_quantization_config(is_per_channel=per_channel, is_qat=False)
    quantizer.set_global(quantization_config)
    # prepare_pt2e folds BatchNorm operators into preceding Conv2d operators, and inserts observers in appropriate places in the model.
    model = prepare_pt2e(model, quantizer)
    
    num_batches = args.num_samples // args.batch_size
    num_batches = min(num_batches, len(data_loader))
    dataloader = data_loader_to_list(data_loader, num_batches)

    # with torch.autocast(device_type="cuda"):
    for image, _ in tqdm(dataloader):
        model(image.to(device))

    
    quantized_model = convert_pt2e(model, use_reference_representation=False)
    torch.ao.quantization.move_exported_model_to_eval(quantized_model)

    return quantized_model


def main(args):
    
    model = torch.load(args.model_path, map_location=torch.device("cpu"))

    train_loader, test_loader, nb_classes = prepare_data(args.batch_size)
    mini_test_loader = getMiniTestDataset()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if args.do_finetune:
        replace_activation_and_train(model, args, train_loader, mini_test_loader, device)

    # quantize model
    print('Quantizing model...')
    if args.quantizer == "simple":
        quantized_model = simple_quantize(model, train_loader, args, per_channel=args.per_channel)
    elif args.quantizer == "all_linear":
        quantized_model = quantize_all_linear(model, train_loader, args, per_channel=args.per_channel)
    elif args.quantizer == "all_matrix":
        quantized_model = quantize_all_matrix(model, train_loader, args, per_channel=args.per_channel)
    elif args.quantizer == "matrix_by_kurtosis":
        quantized_model = quantize_by_kurtosis(model, train_loader, args, per_channel=args.per_channel)
    
    # export and save the model
    save_model(quantized_model, args.save_path)
    
    # load the module
    loaded_quantized_ep = torch.export.load(args.save_path)
    loaded_quantized_model = loaded_quantized_ep.module()

    # test the module
    lab4_cifar100_evaluation(args.save_path)

    # torch profile
    dummy_input_data = [torch.randn(1, 3, 224, 224) for _ in range(3)]
    torch_profile(loaded_quantized_model, dummy_input_data, torch.device("cpu"))



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        default="0.9099_deit3_small_patch16_224.pth"
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default="quantized_model.pth"
    )

    # finetune argument
    parser.add_argument(
        "--do_finetune",
        action="store_true"
    )

    parser.add_argument(
        "--reg",
        type=float,
        default=1e-5
    )

    parser.add_argument(
        "--batch_size",
        type=int, 
        default=128
    )

    parser.add_argument(
        "--epochs",
        type=int, 
        default=25
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5
    )

    parser.add_argument(
        "--activation",
        type=str,
        default="ReLU6"
    )
    
    # quantize argument
    parser.add_argument(
        "--quantizer",
        type=str,
        choices=["simple", "all_linear", "all_matrix", "matrix_by_kurtosis"],
        default="all_linear"
    )

    parser.add_argument(
        "--per_channel",
        action="store_true"
    )

    parser.add_argument(
        "--kurtosis_thres",
        type=float,
        default=20.0
    )

    parser.add_argument(
        "--num_samples",
        type=int, 
        default=2048
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda"
    )

    args = parser.parse_args()
    main(args)
