from torch.utils import data
import pickle
import numpy as np

class Dataset_boston_housing(data.Dataset):
    # boston housing dataset
    def __init__(self, path):
        # read data pickle as a dict, where keys are X, X_labels
        with open(path, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return self.data['X'].shape[0]

    def __getitem__(self, index):
        index = index % self.data['X'].shape[0]
        return self.data['X'][index].astype('float32'), np.asarray([self.data['X_labels'][index]]).astype('float32')

