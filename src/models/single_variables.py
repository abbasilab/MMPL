import random

import numpy as np
import torch

class SiameseContrastiveLoss(torch.nn.Module):
    """The loss function computed as the siamese loss"""
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, data, labels):
        batch_size = data.shape[0]
        rangeset = torch.arange(batch_size)
        all_combos = torch.combinations(rangeset)
        same_labels = all_combos[(labels[all_combos[:, 0]] == labels[all_combos[:, 1]]).nonzero()].squeeze()
        opposite_labels = all_combos[(labels[all_combos[:, 0]] != labels[all_combos[:, 1]]).nonzero()].squeeze()
        same_distances = torch.norm(data[same_labels][:, 0] - data[same_labels][:, 1], dim=1)
        opposite_distances = torch.norm(data[opposite_labels][:, 0] - data[opposite_labels][:, 1], dim=1)
        same_loss = 0.5*torch.sum(same_distances.pow(2))
        opposite_loss = 0.5*torch.sum(torch.max(torch.tensor(0), self.m - opposite_distances).pow(2))
        final = same_loss + opposite_loss
        return final

class LSTMEncoder(torch.nn.Module):
    """LSTM Autoencoder used for encoding the input sequence"""
    def __init__(self, input_size, hidden):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden
        self.lstm_unit = torch.nn.LSTM(input_size,hidden,3,batch_first=True)
        self.linear1 = torch.nn.Linear(hidden, hidden)
        # self.linear2 = torch.nn.Linear(hidden,hidden)
    def forward(self, data):
        t, hidden = self.lstm_unit(data)
        t = t[:,-1,:]
        return self.linear1(t)
    
class LSTMDecoder(torch.nn.Module):
    def __init__ (self, hidden, seq_len):
        super().__init__()
        self.hidden = hidden
        self.seq_len = seq_len
        self.lstm_unit = torch.nn.LSTM(input_size=1, hidden_size=seq_len, num_layers=3, batch_first=True)
        self.signal = torch.nn.Linear(seq_len, seq_len)
    def forward(self,data):
        data = data.unsqueeze(2)
        x, (hidden_n, cell_n) = self.lstm_unit(data)
        x = x[:, -1, :]
        x = self.signal(x)
        return x

class DecodingModule(torch.nn.Module):
    def __init__(self, hidden, inputsize, num_variables):
        super().__init__()
        self.hidden = hidden
        self.inputsize = inputsize
        self.num_variables = num_variables
        self.decoders = torch.nn.ModuleList([LSTMDecoder(hidden, inputsize) for _ in range(num_variables)])

    def forward(self, data):
        results = []
        for i in range(self.num_variables):
            data_subset = data[i]
            results.append(self.decoders[i](data_subset))
        return torch.stack(results).movedim(0, -1)
    
class PrototypeLayerReal(torch.nn.Module):
    """The class implementing the prototype matching layer"""
    def __init__(self, num_prototypes, hidden, num_classes, fc_layer=True):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.prototype_matrix = torch.nn.Parameter(torch.zeros(num_prototypes, hidden))
        self.fc_layer = fc_layer
        if fc_layer:
            self.fc_layer = torch.nn.Linear(self.num_prototypes, num_classes)
    def forward(self, data):
        """Data is batch X hidden_dim"""
        data_temp = torch.unsqueeze(data,1).repeat_interleave(self.num_prototypes, 1)
        distances = data_temp - self.prototype_matrix
        distances = torch.norm(distances, dim=2) ### Batch X Num
        if self.fc_layer:
            distances = self.fc_layer(distances)
        sim = torch.pow(1 + distances, -1)
        return sim

class SensorLevelModule(torch.nn.Module):
    def __init__(self, hidden, num_prototypes, encoder=True):
        super().__init__()
        self.hidden = hidden
        self.num_prototypes = num_prototypes
        self.encoder = LSTMEncoder(1, self.hidden)
        self.protolayer = PrototypeLayerReal(num_prototypes, hidden, None, False)
    def forward(self,data):
        """Data is of shape Batch X Time Series Length X 1"""
        encoded = self.encoder(data)
        return self.protolayer(encoded)

class EncodingModule(torch.nn.Module):
    def __init__(self, module_list):
        super().__init__()
        self.module_list = module_list
        self.num_variables = len(module_list)
    
    def forward(self, data):
        results = []
        for i in range(self.num_variables):
            data_subset = data[:, :, i]
            results.append(self.module_list[i](data_subset.unsqueeze(2)))
        return results

class SingleVariableModulesWrapper(torch.nn.Module):
    def __init__(self, num_variables, num_classes, hidden, num_prototypes):
        super().__init__()
        self.num_variables = num_variables
        self.num_classes = num_classes
        self.hidden = hidden
        self.num_prototypes = num_prototypes

        self.single_variable_modules = torch.nn.ModuleList()
        for _ in range(self.num_variables):
            self.single_variable_modules.append(SensorLevelModule(hidden, num_prototypes))

        # Used just for training the single variable modules
        self.linear1 = torch.nn.Linear(num_variables * num_prototypes, 2 * num_variables * num_prototypes)

    def forward(self, data):
        concat_features = []
        for i in range(self.num_variables):
            latent = self.single_variable_modules[i](data[:, :, i].unsqueeze(2))
            concat_features.append(latent)
        concat_features = torch.cat(concat_features, dim=1)

        aggregate_features = self.linear1(concat_features)
        return aggregate_features, concat_features

def prototype_similarity_penalty(data, single_variable_module):
    encodings = single_variable_module.encoder(data)
    encodings = encodings.unsqueeze(1).repeat_interleave(single_variable_module.protolayer.num_prototypes, 1)
    distances = encodings - single_variable_module.protolayer.prototype_matrix
    distances = torch.norm(distances, dim=2)
    distances = torch.norm(distances, dim=1)[0]
    return torch.max(distances)

def encoded_space_coverage_penalty(data, single_variable_module):
    encodings = single_variable_module.encoder(data)
    encodings = encodings.unsqueeze(1).repeat_interleave(single_variable_module.protolayer.num_prototypes,1)
    distances = encodings - single_variable_module.protolayer.prototype_matrix
    distances = torch.norm(distances, dim=2)
    distances = torch.min(distances, dim=0)[0]
    return torch.mean(distances)

def prototype_diversity_penalty(prototypes):
    num_prototypes = prototypes.shape[0]
    collect = []
    for i in range(num_prototypes - 1):
        this_prototype = prototypes[i]
        without_this = prototypes[i + 1:]
        this_prototype = this_prototype.unsqueeze(0).repeat_interleave(without_this.shape[0], 0)
        collect.append(torch.min(torch.square(torch.norm(this_prototype-without_this, dim=1))))
    collect = torch.stack(collect)
    mean = torch.pow(torch.log(torch.mean(collect)), -1)
    return mean

def initialize_prototypes(sv_modules_wrapper, data):
    """
    sv_modules_wrapper: SingleVariableModulesWrapper
    data: Dataset
    """
    with torch.no_grad():
        # Iterate through the variables
        for i in range(sv_modules_wrapper.num_variables):
            data_load = torch.utils.data.DataLoader(data, len(data), True)
            sensor_level_module = sv_modules_wrapper.single_variable_modules[i]
            protolayer = sensor_level_module.protolayer
            for data_matrix, _ in data_load:
                encodings = sv_modules_wrapper.single_variable_modules[i].encoder(data_matrix[:,:,i].unsqueeze(2).float())

                # Step 1: Set a random element to be the first prototype
                protolayer.prototype_matrix[0] = random.choice(encodings)
                for j in range(1, sv_modules_wrapper.num_prototypes):
                    # Step 2: For each data point calculate distance to each chosen prototype
                    # Keep distance to closest chosen prototype
                    distances = []
                    for point in encodings:
                        min_distance = float("inf")
                        for k in range(j):
                            prototype = protolayer.prototype_matrix[k]
                            min_distance = min(min_distance, torch.linalg.vector_norm(point - prototype))
                        distances.append(float(min_distance))
                    probabilities = np.array(distances)
                    probabilities = np.square(probabilities)
                    probabilities = probabilities / probabilities.sum()

                    # Step 3: Choose an element at random to be the next prototype with prob proportional to distance
                    found = False
                    while not found:
                        candidate = np.random.choice(list(encodings), p=probabilities)
                        if candidate not in protolayer.prototype_matrix:
                            found = True
                    protolayer.prototype_matrix[j] = candidate