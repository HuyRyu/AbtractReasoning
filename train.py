from dataset import ARCDataset
import torch
from model import RNN
from torch.utils.data import DataLoader
from utils import cosine_annealing
import torch.nn as nn
import numpy as np


def compute_score(encoded_img, target):
    score = (encoded_img == target)


if __name__ == '__main__':
    torch.manual_seed(0)
    model = RNN(900, 900, 1, False, .0)
    trainset = ARCDataset('training')
    train_loader = DataLoader(trainset, 8, shuffle=True,  num_workers=1)
    lr_default = 1e-3
    optim = torch.optim.Adamax(list(model.parameters()), lr=lr_default)
    n_cycles = 25
    criterion = nn.MSELoss()
    total_loss = 0
    for epoch in range(50):
        total_step = 50 * len(train_loader)
        for i, (inp, idx_padding, target) in enumerate(train_loader):
            optim.param_groups[0]['lr'] = cosine_annealing(i, total_step, n_cycles, lr_default)
            inp = inp.view(inp.size(0), inp.size(1), -1)
            target = target.view(target.size(0), -1)
            encoded_img = model.forward_all(inp)
            idx_padding = idx_padding.unsqueeze(1).unsqueeze(2).expand(inp.size(0), 1, inp.size(2))
            encoded_img = torch.gather(encoded_img, 1, idx_padding)
            encoded_img = torch.relu(encoded_img)
            loss = torch.sqrt(criterion(encoded_img, target))
            total_loss += loss
            loss.backward()
            optim.step()
            optim.zero_grad()
            print ((encoded_img==target).int().sum(1) / 900)
            print('Loss: %.4f' % loss)