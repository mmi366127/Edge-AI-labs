import os
import math
import time
import json
from tqdm import tqdm


import torch
import torch.nn as nn
import numpy as np

from .eval import evaluate_model



# training utils
def replace_activation(model, args):
    for name, module in model.named_modules():
        last_name = name.rsplit('.', 1)[-1]
        if "mlp" in last_name:
            print(f"replace {module.act} to {args.activation}")
            module.act = getattr(nn, args.activation)()
    return model


def train_one_epoch(model, criterion, optimizer, data_loader, device):
    
    model.train()
    model.to(device)
    cnt = 0

    for image, target in tqdm(data_loader):
        cnt += 1
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return


def replace_activation_and_train(model, args, train_loader, test_loader, device):

    model = replace_activation(model, args)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.reg)

    max_acc = 0
    for i in range(args.epochs):
        train_one_epoch(
            model,
            criterion,
            optimizer, 
            train_loader,
            device
        )
        acc = evaluate_model(model, test_loader, device)
        if acc > max_acc:
            max_acc = acc
            torch.save(model, os.path.join("training-results", f"replace-act-{args.activation}-{acc}.pt"))