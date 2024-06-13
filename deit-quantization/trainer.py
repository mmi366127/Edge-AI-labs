import numpy as np
import torch
from torch import nn
import os
from tqdm.auto import tqdm
import math
import time

from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader

from torch.export import export, ExportedProgram
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e, prepare_qat_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
import argparse

torch.cuda.is_available()

from utils.data import prepare_data, getMiniTestDataset
from utils.eval import evaluate_model
from utils.train import train_one_epoch


def replace_activation(model, args):
    for name, module in model.named_modules():
        last_name = name.rsplit('.', 1)[-1]
        if "mlp" in last_name:
            print(f"replace {module.act} to {args.activation}")
            module.act = getattr(nn, args.activation)()
    return model


@torch.no_grad()
def clip_model_parameters(model: nn.Module, v):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.weight.data.clamp(-v, v)


def main(args):
    # prepare dataset
    train_loader, test_loader, nb_classes = prepare_data(args.batch_size)
    mini = getMiniTestDataset()
    # load model
    model = torch.load(args.model_path)

    # replace model activation
    model = replace_activation(model, args)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    max_acc = 91
    for i in range(args.epochs):
        train_one_epoch(
            model, 
            criterion,
            optimizer,
            train_loader, 
            torch.device(args.device)
        )
        # clip_model_parameters(model, )
        acc = evaluate_model(model, mini, torch.device(args.device))
        if acc > max_acc:
            max_acc = acc            
            torch.save(model, os.path.join('training-results', f'replace-act-{acc}.pt'))




if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int, 
        default=100
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default="0.9099_deit3_small_patch16_224.pth"
    )

    parser.add_argument(
        "--batch_size",
        type=int, 
        default=32
    )

    parser.add_argument(
        "--activation",
        type=str,
        default="ReLU6"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda"
    )

    args = parser.parse_args()
    main(args)

