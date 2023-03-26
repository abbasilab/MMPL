import torch

class AggregatePrototypeLayer(torch.nn.Module):
    def __init__(self, num_prototypes,encoding_size,num_classes):
        super().__init__()
        self.encoding_size = encoding_size
        self.num_prototypes = num_prototypes
        self.protos = torch.nn.Parameter(torch.rand(self.num_prototypes, self.encoding_size))
        self.fc = torch.nn.Linear(num_prototypes, num_classes)
    def forward(self,data):
        """Data is of shape Batch X Encoding Size"""
        encoded = data#torch.relu(self.encoder_linear(data))
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