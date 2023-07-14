import torch


class BaselineModel(torch.nn.Module):
    def __init__(self, length, num_prototypes, num_classes):
        super().__init__()
        self.length = length
        self.num_prototypes = num_prototypes
        self.num_classes = num_classes
        self.protos = torch.nn.Parameter(torch.zeros(self.num_prototypes, self.length))
        self.fc = torch.nn.Linear(num_prototypes, num_classes)

    def forward(self, data):
        # data is of size Batch Size x Num Variables x Num Timesteps
        

