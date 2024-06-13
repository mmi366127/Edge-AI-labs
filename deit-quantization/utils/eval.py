import os
import math
import time
import json
from tqdm import tqdm


import torch


from .data import getMiniTestDataset
from . import get_size_of_model




# evaluation utils
def evaluate_model(model, data_loader, device):
    # model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(data_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy}%')
    return accuracy


# TA Uses the following code to evaluate your score
def lab4_cifar100_evaluation(quantized_model_path='deits_quantized.pth'):
    # Prepare data
    mini_test_dataset = getMiniTestDataset()

    # Load quantized model
    quantized_ep = torch.export.load(quantized_model_path)
    quantized_model = quantized_ep.module()

    # Evaluate model
    start_time = time.time()
    acc = evaluate_model(quantized_model, mini_test_dataset, device="cpu")
    exec_time = time.time() - start_time
    model_size = get_size_of_model(quantized_model)

    print(f"Model Size: {model_size:.2f} MB")
    print(f"Accuracy: {acc:.2f}%")
    print(f"Execution Time: {exec_time:.2f} s")

    score = 0
    if model_size <= 30: score += 10
    if model_size <= 27: score += 2 * math.floor(27-model_size)
    if acc >= 86:
      score += 10 + 2 * math.floor(acc-86)
    print(f'Model Score: {score:.2f}')
    return score