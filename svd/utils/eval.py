
import os
from tqdm import tqdm

import torch
import torch.nn as nn

from .text_data import get_eval_loaders




@torch.no_grad()
def evaluate_perplexity(model, dataset, limit):
    # for creating sensitivity list
    """
    dataset: input ids tensor of shape [batch, sequence length]
    """
    nsamples, seqlen = dataset.size()

    nlls = []
    bs = 8      # batch size
    
    for i in range(0, limit, bs):
        j = min(i + bs, limit)
        input_ids = dataset[i:j, :-1].to(model.device)
        labels = dataset[i:j, 1:].contiguous()
        logits = model(input_ids=input_ids)[0]
        shift_logits = logits[:, :, :]
        shift_labels = labels.to(model.device)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        neg_log_likelihood = loss.float() * seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * seqlen))
    return ppl.item()


@torch.no_grad()
def eval_ppl(model, tokenizer, model_name, datasets, seqlen=2048, device="cuda"):

    if isinstance(device, str):
        device = torch.device(device)

    results = {}

    for dataset in datasets.split(","):
        cache_testloader = (
            f"cache/dataset/{dataset}_testloader_{model_name.replace('/', '_')}_all.cache"
        )
        if os.path.exists(cache_testloader):
            testloader = torch.load(cache_testloader)
        else:
            testloader = get_eval_loaders(dataset, tokenizer)
            torch.save(testloader, cache_testloader)
        
        testenc = testloader.input_ids
        nsamples = testenc.numel() // seqlen
        use_cache = model.config.use_cache
        model.config.use_cache = False
        model.eval()

        nlls = []

        for i in tqdm(range(nsamples)):
            batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to(
                    device
                )
            outputs = model.model(batch)
            hidden_states = outputs[0]
            logits = model.lm_head(hidden_states)  # .contiguous()
            shift_logits = logits[:, :-1, :]  # .contiguous()
            shift_labels = testenc[:, (i * seqlen) : ((i + 1) * seqlen)][
                :, 1:
            ].to(device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
            
            neg_log_likelihood = loss.float() * seqlen
            nlls.append(neg_log_likelihood)

            
        ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * seqlen))
        print(dataset, ppl.item())
        model.config.use_cache = use_cache
        results.update({dataset: ppl.item()})

    return results



@torch.no_grad()
def eval_accuracy(model, dataset, dev=torch.device("cuda")):

    model.to(dev)
    correct = 0
    total = 0
    with torch.no_grad():
        for  images, labels in dataset:
            images, labels = images.to(dev), labels.to(dev)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


@torch.no_grad()
def eval_loss(model, dataset, dev=torch.device("cuda")):

    model.to(dev)
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0
    total_cnt = 0
    with torch.no_grad():
        for  images, labels in dataset:
            images, labels = images.to(dev), labels.to(dev)
            outputs = model(images)
            total_loss += loss_fn(outputs, labels)
            total_cnt += labels.size(0)
    
    return (total_loss / total_cnt).item()