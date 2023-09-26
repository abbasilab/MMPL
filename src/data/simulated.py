import math
import random
import itertools

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.data.data import get_ds

class FrequencyDomainVariable:
    """A variable which exhibits variation in the frequency domain"""
    def __init__(self, active_frequency, inactive_frequencies):
        """Frequencies is a list that shows the frequencies and whether they are active or inactive"""
        self.active_frequency = active_frequency
        self.inactive_frequency = inactive_frequencies
    def sample_active(self, active):
        return np.sin((np.random.normal(active, 0.1)) * np.linspace(0, 2 * math.pi,100) + np.random.rand() * 2 * math.pi) + np.random.normal(0.0,0.1,100)
    def sample_inactive(self):
        current_frequency = random.choice(self.inactive_frequency)
        return np.sin((np.random.normal(current_frequency, 0.1)) * np.linspace(0, 2 * math.pi,100) + np.random.rand() * 2 * math.pi) + np.random.normal(0.0, 0.1, 100)
class ShiftVariantVariable:
    """A variable which is shift variant"""
    def __init__(self, active_shift_amount, inactive_shift_amounts):
        self.active_shift_amount = active_shift_amount
        self.inactive_shift_amount = inactive_shift_amounts
        self.g1 = np.sin(5 * np.linspace(0, 2 * math.pi, 200))[0:20]
    def sample_active(self, active):
        pattern_type = active
        if pattern_type==0:
            shiftone = np.random.randint(0, 10)
            return np.concatenate([np.zeros(shiftone), (0.5+np.random.rand()) * self.g1, np.zeros(100-(shiftone+20))]) + np.random.normal(0, 0.1, 100)
        if pattern_type == 1:
                shiftone = np.random.randint(25, 35)
                return np.concatenate(
                    [np.zeros(shiftone), (0.5 + np.random.rand()) * self.g1, np.zeros(100 - (shiftone + 20))]) + np.random.normal(0, 0.1, 100)
        if pattern_type==2:
            shiftone = np.random.randint(50, 60)
            return np.concatenate([np.zeros(shiftone), (0.5+np.random.rand()) * self.g1, np.zeros(100-(shiftone+20))]) + np.random.normal(0, 0.1, 100)
        if pattern_type==3:
            shiftone = np.random.randint(75, 80)
            return np.concatenate([np.zeros(shiftone), (0.5+np.random.rand()) * self.g1, np.zeros(100-(shiftone+20))]) + np.random.normal(0, 0.1, 100)
    def sample_inactive(self):
        pattern_type = random.choice(self.inactive_shift_amount)
        if pattern_type==0:
            shiftone = np.random.randint(0, 10)
            return np.concatenate([np.zeros(shiftone), (0.5+np.random.rand()) * self.g1, np.zeros(100-(shiftone+20))]) + np.random.normal(0, 0.1, 100)
        if pattern_type == 1:
                shiftone = np.random.randint(25, 35)
                return np.concatenate(
                    [np.zeros(shiftone), (0.5 + np.random.rand()) * self.g1, np.zeros(100 - (shiftone + 20))]) + np.random.normal(0, 0.1, 100)
        if pattern_type==2:
            shiftone = np.random.randint(50, 60)
            return np.concatenate([np.zeros(shiftone), (0.5+np.random.rand()) * self.g1, np.zeros(100-(shiftone+20))]) + np.random.normal(0, 0.1, 100)
        if pattern_type==3:
            shiftone = np.random.randint(75, 80)
            return np.concatenate([np.zeros(shiftone), (0.5+np.random.rand()) * self.g1, np.zeros(100-(shiftone+20))]) + np.random.normal(0, 0.1, 100)
class ShiftInvariantVariable:
    def __init__(self, active, inactive):
        self.active = active
        self.inactive = inactive
        self.g1 = np.sin(5 * np.linspace(0, 2 * math.pi, 200))[0:20]
        self.g2 = np.sin(5 * np.linspace(0, 2 * math.pi, 200))[20:40]
    def sample_active(self, active):
        pattern_type = active
        if pattern_type==0:
            shiftone = np.random.randint(0, 55)
            return np.concatenate([np.zeros(shiftone), (1 + np.random.normal(0,0.05)) * self.g1, np.zeros(5), (1 + np.random.normal(0,0.05)) * self.g1,
                            np.zeros(100 - (shiftone + 45))]) + np.random.normal(0, 0.1, 100)
        if pattern_type == 1:
            shiftone = np.random.randint(0, 55)
            return np.concatenate([np.zeros(shiftone), (1 + np.random.normal(0,0.05)) * self.g1, np.zeros(5), (1 + np.random.normal(0,0.05)) * self.g2,
                            np.zeros(100 - (shiftone + 45))]) + np.random.normal(0, 0.1, 100)

        if pattern_type == 2:
            shiftone = np.random.randint(0, 55)
            return np.concatenate([np.zeros(shiftone), (1 + np.random.normal(0,0.05)) * self.g2, np.zeros(5), (1 + np.random.normal(0,0.05)) * self.g1,
                            np.zeros(100 - (shiftone + 45))]) + np.random.normal(0, 0.1, 100)

        if pattern_type == 3:
            shiftone = np.random.randint(0, 55)
            return np.concatenate([np.zeros(shiftone), (1 + np.random.normal(0,0.05)) * self.g2, np.zeros(5), (1 + np.random.normal(0,0.05)) * self.g2,
                            np.zeros(100 - (shiftone + 45))]) + np.random.normal(0, 0.1, 100)
    def sample_inactive(self):
        pattern_type = random.choice(self.inactive)
        if pattern_type==0:
            shiftone = np.random.randint(0, 55)
            return np.concatenate([np.zeros(shiftone), (1 + np.random.normal(0,0.05)) * self.g1, np.zeros(5), (1 + np.random.normal(0,0.05)) * self.g1,
                            np.zeros(100 - (shiftone + 45))]) + np.random.normal(0, 0.1, 100)
        if pattern_type == 1:
            shiftone = np.random.randint(0, 55)
            return np.concatenate([np.zeros(shiftone), (1 + np.random.normal(0,0.05)) * self.g1, np.zeros(5), (1 + np.random.normal(0,0.05)) * self.g2,
                            np.zeros(100 - (shiftone + 45))]) + np.random.normal(0, 0.1, 100)

        if pattern_type == 2:
            shiftone = np.random.randint(0, 55)
            return np.concatenate([np.zeros(shiftone), (1 + np.random.normal(0,0.05)) * self.g2, np.zeros(5), (1 + np.random.normal(0,0.05)) * self.g1,
                            np.zeros(100 - (shiftone + 45))]) + np.random.normal(0, 0.1, 100)

        if pattern_type == 3:
            shiftone = np.random.randint(0, 55)
            return np.concatenate([np.zeros(shiftone), (1 + np.random.normal(0,0.05)) * self.g2, np.zeros(5), (1 + np.random.normal(0,0.05)) * self.g2,
                            np.zeros(100 - (shiftone + 45))]) + np.random.normal(0, 0.1, 100)

class IrrelevantVariable:
    def __init__(self):
        pass
    def sample_active(self, active):
        return 1 + np.random.normal(0, 0.05, 100)
    def sample_inactive(self):
        return 1 + np.random.normal(0, 0.05, 100)

class DataMiningSimulatedDataset:
    """Simulated dataset where there are few relevant variables per class """
    def __init__(self):
        self.variables = [] ##Contains a list of variable objects
        inactive = [0,1,2,3]
        self.variables.append(ShiftInvariantVariable(0,inactive))
        self.variables.append(ShiftVariantVariable(0, inactive))
        self.variables.append(FrequencyDomainVariable(0, inactive))
        self.variables.append(IrrelevantVariable())
        self.active_variables = list(enumerate(sorted(list(itertools.product(range(4), repeat=3)))))
    def generate_dataset(self,reps):
        complete_dataset = list()
        for classif in self.active_variables:
            for rep in range(reps):
                this_sample = list()
                for variable in range(3):
                    this_sample.append(self.variables[variable].sample_active(classif[1][variable]))
                this_sample.append(self.variables[3].sample_inactive())
                this_sample = np.array(this_sample).transpose()
                complete_dataset.append([this_sample,classif[0]])
        return complete_dataset, self.active_variables
class DataMiningData(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, item):
        return self.dataset[item][0], self.dataset[item][1]
class ProjectDatasetReal:
    def __init__(self,dataset,encoder):
        self.dataset = dataset
        self.encoder = encoder
    def project(self,prototype_vector):
        data,label = self.dataset
        print(data.shape)
        encoded = self.encoder(data.unsqueeze(2).float())
        difference_vectors = prototype_vector-encoded ##Shape Batch X hidden
        distances = torch.norm(difference_vectors,dim=1)
        minimum = torch.argmin(distances,dim=0)
        return data[minimum],label[minimum]
    
def generate(save_name, num_points_per_class):
    data = DataMiningSimulatedDataset()
    ds, class_descriptor = data.generate_dataset(num_points_per_class)
    data_load = torch.utils.data.DataLoader(ds, len(ds), False)
    with open(save_name, 'w') as f:
        f.write("@problemName simulated\n")
        f.write("@timeStamps false\n")
        f.write("@missing false\n")
        f.write("@univariate false\n")
        f.write("@dimensions 4\n")
        f.write("@equalLength true\n")
        f.write("@seriesLength 100\n")
        f.write("@classLabel true")
        for i in range(64):
            f.write(" " + str(i))
        f.write("\n")
        f.write("@data\n")
        for data_matrix, labels in data_load:
            batch_size, seq_len, num_variables = data_matrix.size()
            # Iterate over all time series
            for i in range(batch_size):
                # Then over all variables
                for j in range(num_variables):
                    # Then over each point in the time series
                    for k in range(seq_len):
                        point = data_matrix[i][k][j]
                        f.write(str(point.item()))
                        if k == seq_len - 1:
                            f.write(":")
                        else:
                            f.write(",")
                f.write(str(labels[i].item()))
                f.write("\n")
    
if __name__ == "__main__":

    # Generate dataset with 10 points per class, write to .ts file
    generate("data/simulated_640/processed/train.ts", 10)
    generate("data/simulated_640/processed/val.ts", 10)
    generate("data/simulated_640/processed/test.ts", 10)

    generate("data/simulated_6400/processed/train.ts", 100)
    generate("data/simulated_6400/processed/val.ts", 100)
    generate("data/simulated_6400/processed/test.ts", 100)
    