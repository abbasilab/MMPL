import numpy as np
import torch
from sktime.datasets import load_from_tsfile_to_dataframe

from .simulated import DataMiningSimulatedDataset, DataMiningData

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

def get_simulated_ds(reps):
    sim = DataMiningSimulatedDataset()
    data, class_descriptor = sim.generate_dataset(reps)
    dataset_object = DataMiningData(data)
    return dataset_object, class_descriptor

def filter_classes(ds, classes):
    data = ds.data
    filtered = []
    for point in data:
        if point[1] in classes:
            new_point = [None, None]
            new_point[0] = point[0]
            new_point[1] = classes.index(point[1])
            filtered.append(new_point)
    
    filtered_ds = BenchmarkDataset(filtered)
    return filtered_ds

if __name__ == "__main__":
    train_ds, _ = get_simulated_ds(10)
    torch.save(train_ds, "data/simulated/train_10.dat")

    test_ds, _ = get_simulated_ds(10)
    torch.save(test_ds, "data/simulated/test_10.dat")