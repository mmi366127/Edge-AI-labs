import argparse
import torch

from torch.export import export, ExportedProgram, dynamic_dim, Dim
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e, prepare_qat_pt2e
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)


from utils.eval import lab4_cifar100_evaluation
from utils.quantizer import PartialXNNPACKQuantizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="deits_quantized.pth"
    )

    args = parser.parse_args()

    # test the module
    lab4_cifar100_evaluation(quantized_model_path=args.model)
