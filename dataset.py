from torch.utils.data import Dataset
from os import listdir
import _pickle as pickle


class ARCDataset(Dataset):
    def __init__(self, split):
        super(ARCDataset, self).__init__()
        self.entries = pickle.load(open('%s.pkl' % split, 'rb'))

    def __getitem__(self, idx):
        entry = self.entries[idx]
        inp = entry['input']
        inp = inp[:3]
        target = entry['target']
        return inp, target

    def __len__(self):
        return len(self.entries)
