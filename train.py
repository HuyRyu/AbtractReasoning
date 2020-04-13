from dataset import ARCDataset
import torch
from model import RNN
from torch.utils.data import DataLoader
from utils import cosine_annealing
import numpy as np

if __name__ == '__main__':
    torch.manual_seed(0)
    model = RNN(900, 1024, 1, False, .0)
    trainset = ARCDataset('training')
    train_loader = DataLoader(trainset, 8, shuffle=True,  num_workers=1)
    lr_default = 1e-3
    optim = torch.optim.Adamax(list(model.parameters()), lr=lr_default)
    n_cycles = 25
    for epoch in range(50):
        total_step = 50 * len(train_loader)
        for i, (inp, target) in enumerate(train_loader):
            optim.param_groups[0]['lr'] = cosine_annealing(i, total_step, n_cycles, lr_default)
            emb = model(inp)
            print(emb)