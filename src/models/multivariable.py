import random

import numpy as np
import torch

class AggregatePrototypeLayer(torch.nn.Module):
    def __init__(self, num_prototypes,encoding_size,num_classes):
        super().__init__()
        self.encoding_size = encoding_size
        self.num_prototypes = num_prototypes
        self.protos = torch.nn.Parameter(torch.zeros(self.num_prototypes, self.encoding_size))
        self.fc = torch.nn.Linear(num_prototypes, num_classes)
    def forward(self,data):
        """Data is of shape Batch X Encoding Size"""
        encoded = data
        encoded = encoded.unsqueeze(1).repeat_interleave(self.num_prototypes, 1) ## Now Batch X Num Proto X Hidden
        distances=torch.norm(encoded - self.protos, dim=2)
        result = self.fc(distances)
        return result

class MultivariableModule(torch.nn.Module):
    """A framework for interpretable multivariable """
    def __init__(self, single_variable_modules, num_variables, hidden, num_classes, num_prototypes):
        super().__init__()
        self.num_variables = num_variables
        self.num_classes = num_classes
        self.hidden = hidden
        self.num_prototypes = num_prototypes
        self.single_variable_modules = single_variable_modules
        for k in range(self.num_variables):
            for param in self.single_variable_modules[k].parameters():
                param.requires_grad = False
        self.aggregate_prototype_layer = AggregatePrototypeLayer(num_classes, hidden, num_classes)
    def initialize_prototypes(self, data):
        with torch.no_grad():
            data_load = torch.utils.data.DataLoader(data, len(data), True)
            for data_matrix, _ in data_load:
                concat_features = list()
                for i in range(self.num_variables):
                    latent = self.single_variable_modules[i](data_matrix[:, :, i].unsqueeze(2).float())
                    concat_features.append(latent)
                concat_features = torch.cat(concat_features, dim=1)

                # Step 1: Select random element to be first prototype
                self.aggregate_prototype_layer.protos[0] = random.choice(concat_features)
                for i in range(1, self.num_prototypes):
                    # Step 2: For each concat sim vector calculate distance to each chosen prototype
                    # Keep distance to closest chosen prototype
                    distances = []
                    for concat_sim_vector in concat_features:
                        min_distance = float("inf")
                        for j in range(i):
                            prototype = self.aggregate_prototype_layer.protos[j]
                            min_distance = min(min_distance, torch.norm(concat_sim_vector - prototype))
                        distances.append(float(min_distance))
                    probabilities = np.array(distances)
                    probabilities = probabilities / probabilities.sum()

                    # Step 3: Pick a random concat sim vector to be next prototype
                    # Select with probability proportional to min distance to existing prototypes
                    found = False
                    while not found:
                        index = np.random.choice([ind for ind in range(len(concat_features))], p=probabilities)
                        candidate = concat_features[index]
                        if candidate not in self.aggregate_prototype_layer.protos:
                            found = True
                    self.aggregate_prototype_layer.protos[i] = candidate
                

    def forward(self,data):
        """Data is of the shape Batch X Time X Num Variables"""
        concat_features = list()
        for k in range(self.num_variables):
            latent = self.single_variable_modules[k](data[:, :, k].unsqueeze(2))
            #print(latent.shape)
            concat_features.append(latent)
        concat_features = torch.cat(concat_features,dim=1)
        aggregated_features = self.aggregate_prototype_layer(concat_features)
        return aggregated_features, concat_features

def similarity_penalty1(dataset,prototype_matrix):
    """ Prototype Similarity """
    dataset = dataset.unsqueeze(1).repeat_interleave(prototype_matrix.shape[0],1)
    distances = dataset-prototype_matrix
    distances = torch.square(torch.norm(distances,1,dim=2))
    distances = torch.min(distances, dim=1)[0]
    return torch.max(distances)
def similarity_penalty3(dataset,prototype_matrix):
    """ Encoded Space Coverage """
    dataset = dataset.unsqueeze(1).repeat_interleave(prototype_matrix.shape[0],1)
    distances = dataset-prototype_matrix
    distances = torch.square(torch.norm(distances,1,dim=2))
    distances = torch.min(distances, dim=0)[0]
    return torch.max(distances)
def diversity_penalty(prototypes):
    """ Prototype Diversity """
    num_prototypes = prototypes.shape[0]
    collect = list()
    for k in range(num_prototypes-1):
        this_prototype = prototypes[k]
        without_this = prototypes[k + 1:]
        this_prototype=this_prototype.unsqueeze(0).repeat_interleave(without_this.shape[0],0)
        collect.append(torch.min(torch.square(torch.norm(this_prototype-without_this,dim=1))))
    collect = torch.stack(collect)
    mean = torch.pow(torch.log(torch.mean(collect)),-1)
    return mean