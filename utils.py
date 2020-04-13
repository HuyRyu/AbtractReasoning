import torch
from math import cos, pi
import numpy as np


def cosine_annealing(step, total_step, n_cycles, lrate_max):
    step_per_cycle = total_step / n_cycles
    cos_inner = (pi * (step % total_step)) / (step_per_cycle)
    lr = np.asarray(lrate_max/2 * (cos(cos_inner) + 1), np.float32)
    return lr
