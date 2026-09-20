import random

import numpy as np
import torch


def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(require_cuda=True):
    if torch.cuda.is_available():
        return torch.device("cuda")

    if require_cuda:
        raise RuntimeError(
            "CUDA is required but "
            "is not available."
        )

    return torch.device("cpu")