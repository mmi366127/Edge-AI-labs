
import torch
import torch.nn as nn

from functools import reduce




def whiten_decomposition_from_weight(
    w: torch.Tensor,
    scaling_diag_matrix: torch.Tensor,
    rank: int
):
    H, W = w.size()
    # Get the inverse of scaling_diag_matrix
    try:
        scaling_diag_matrix = scaling_diag_matrix.to(w.device)
    except AttributeError:
        raise FileExistsError("Cache may not be loaded correctly")

    # Get the inverse of scaling_diag_matrix
    scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix.to(torch.float32))

    # Multiply scaling_diag_matrix to weight matrix
    W_scale = torch.matmul(w.to(torch.float32), scaling_diag_matrix.to(torch.float32))

    U, S, Vt = torch.linalg.svd(W_scale, full_matrices=False)

    V = torch.matmul(Vt, scaling_matrix_inv)
    
    # Low rank approximation to the target rank
    U = U[:, :rank]
    S = S[:rank]
    V = V[:rank, :]

    # Check for nan
    if (S != S).any():
        print("nan in S")
        raise ValueError("nan in S")
   
    if (U != U).any():
        print("nan in U")
        raise ValueError("nan in U")
    
    if (V != V).any():
        print("nan in V")
        raise ValueError("nan in V")
    
    sqrtSigma = torch.sqrt(torch.diag(S))

    # Fuse the SVD components
    L = torch.matmul(U, sqrtSigma)
    R = torch.matmul(sqrtSigma, V)

    return L, R


class SVDLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        self.U = nn.Linear(rank, out_features, bias=bias)
        self.VT = nn.Linear(in_features, rank, bias=False)
        
    def forward(self, x):
        return self.U(self.VT(x))

    @staticmethod
    def from_Linear(module: nn.Linear, rank: int):

        weight = module.weight.data
        scaling_diag_matrix = module.scaling_diag_matrix.to(torch.float32).to(weight.device)

        l, r = whiten_decomposition_from_weight(weight.to(torch.float32), scaling_diag_matrix, rank) 
        # l: (in_dim, rank), r: (rank, out_dim)
        print(f"rank: {rank}, l: {l.shape}, r:{r.shape}")

        new_module = SVDLinear(module.in_features, module.out_features, rank=rank, bias=(module.bias is not None))
        
        print(f"U: {new_module.U.weight.data.shape}, V: {new_module.VT.weight.data.shape}")
        assert new_module.U.weight.data.shape == l.shape
        new_module.U.weight.data = l

        assert new_module.VT.weight.data.shape == r.shape
        new_module.VT.weight.data = r

        if module.bias is not None:
            new_module.U.bias = module.bias

        new_module.to(module.weight.data.dtype)
        
        return new_module


MODULES = {
    "Linear": nn.Linear,
    "SVDLinear": SVDLinear,
}

LORA_CONVERT_MAP = {
    "Linear": {
        "SVDLinear": SVDLinear.from_Linear
    }
}

def get_module_key(module):
    if isinstance(module, torch.nn.Linear):
        return 'Linear'
    for module_name, module_type in MODULES.items():
        if isinstance(module, module_type):
            return module_name
    raise ValueError("Unknown module")

def get_new_module(old_module: nn.Module, config: dict):
    new_module_key = config["module"]        
    old_module_key = get_module_key(old_module)
    return LORA_CONVERT_MAP[old_module_key][new_module_key](old_module, **config["kwargs"])


def get_module_by_name(module, module_name):
    names = module_name.split(sep='.')
    return reduce(getattr, names, module)
