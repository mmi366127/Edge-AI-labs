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

from transformers import AutoModelForCausalLM




def main(args):

    model = torch.load(args.model_id)
    



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_id",
        type=str,
        default=""
    )

    args = parser.parse_args()


    main(args)

