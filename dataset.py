from torch.utils.data import Dataset
from os import listdir
import _pickle as pickle
import torch


class ARCDataset(Dataset):
    def __init__(self, split):
        super(ARCDataset, self).__init__()
        self.entries = pickle.load(open('%s.pkl' % split, 'rb'))

    def __getitem__(self, idx):
        entry = self.entries[idx]
        inp = torch.from_numpy(entry['input']).float()
        target = torch.from_numpy(entry['target']).float()

        idx_padding = entry['idx_padding']
        return inp, idx_padding, target

    def __len__(self):
        return len(self.entries)
