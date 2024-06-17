
import numpy as np
import torch




def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def rounding_result(result: dict, block_size:int = 64):
    for key in result.keys():
        rank = result[key]
        rank = max(1, round(rank / block_size)) * block_size
        result[key] = rank
    return result