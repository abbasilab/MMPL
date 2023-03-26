import numpy as np
import torch
from sktime.datasets import load_from_tsfile_to_dataframe

class BenchmarkDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, item):
        return self.data[item][0], self.data[item][1]

def get_ds(file, class_to_index):
    df, y = load_from_tsfile_to_dataframe(file)
    df = df.to_numpy()
    dataset = []
    for example in range(y.shape[0]):
        stacked = np.stack([df[example][i].to_numpy() for i in range(df[example].shape[0])])
        stacked = stacked.transpose()
        dataset.append(tuple([stacked,class_to_index[y[example]]]))

    return BenchmarkDataset(dataset)