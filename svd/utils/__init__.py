
import numpy as np
import torch




def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
