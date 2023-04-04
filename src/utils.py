import os
import random
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch import nn

ZOOM_IN_THRESH = 1.1


def get_view_direction(thetas, phis, overhead_threshold, front_threshold, radius):
    #                   phis [B,];          thetas: [B,]
    # front = 0         [0, front)
    # side (left) = 1   [front, 180)
    # back = 2          [180, 180+front)
    # side (right) = 3  [180+front, 360)
    # top = 4                               [0, overhead]
    # bottom = 5                            [180-overhead, 180]
    # zoom in = 6                               radius < 1.1
    res = torch.zeros(thetas.shape[0], dtype=torch.long)
    # first determine by phis

    # res[(phis < front)] = 0
    res[(phis >= (2 * np.pi - front_threshold / 2)) & (phis < front_threshold / 2)] = 0

    # res[(phis >= front) & (phis < np.pi)] = 1
    res[(phis >= front_threshold / 2) & (phis < (np.pi - front_threshold / 2))] = 1

    # res[(phis >= np.pi) & (phis < (np.pi + front))] = 2
    res[(phis >= (np.pi - front_threshold / 2)) & (phis < (np.pi + front_threshold / 2))] = 2

    # res[(phis >= (np.pi + front))] = 3
    res[(phis >= (np.pi + front_threshold / 2)) & (phis < (2 * np.pi - front_threshold / 2))] = 3
    # override by thetas
    res[thetas <= overhead_threshold] = 4
    res[thetas >= (np.pi - overhead_threshold)] = 5
    # override by radius
    res[radius <= ZOOM_IN_THRESH] = 6
    return res


def tensor2numpy(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().numpy()
    tensor = (tensor * 255).astype(np.uint8)
    return tensor


def make_path(path: Path) -> Path:
    path.mkdir(exist_ok=True, parents=True)
    return path


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

