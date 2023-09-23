import torch

class SingleVariablePrototypesModule(torch.nn.Module):
    """
    Module that holds encoder + prototype layer
    """
    def __init__(self, encoder, num_prototypes, latent_dim, use_fc=False):
        super().__init__()
        self.encoder = encoder
        self.num_prototypes  = num_prototypes
        self.latent_dim = latent_dim

        self.prototypes = torch.nn.Parameter(torch.rand(num_prototypes, latent_dim))

    def forward(self, x):
        """
        x: (batch_size, seq_len, 1)
        Returns: (batch_size, num_prototypes)
        For a given training point, computes embedding and then similarity vector to prototypes.
        """
        x = self.encoder(x)
        
        x = torch.unsqueeze(x, 1).repeat_interleave(self.num_prototypes, 1)
        distances = x - self.prototypes
        distances = torch.norm(distances, dim=2)
        sim = torch.pow(1 + distances, -1)
        return sim
    
class SingleVariablePrototypesWrapper(torch.nn.Module):
    """
    Wrapper class that holds <num_variables> SingleVariablePrototypesModule classes
    """
    def __init__(self, encoders, num_variables, num_classes, num_prototypes, latent_dim, num_layers, dropout):
        super().__init__()
        self.encoders = encoders
        self.num_variables = num_variables
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.dropout_amt = dropout

        single_variable_prototype_modules = []
        for i in range(num_variables):
            single_variable_prototype_modules.append(
                SingleVariablePrototypesModule(
                    encoder=encoders[i],
                    num_prototypes=num_prototypes,
                    latent_dim=latent_dim
                )
            )
        self.single_variable_prototype_modules = torch.nn.ModuleList(single_variable_prototype_modules)

        if num_layers == 1:
            self.linear = torch.nn.Linear(num_variables * num_prototypes, num_classes)
        else:
            layers = [torch.nn.Linear(num_variables * num_prototypes, num_classes), torch.nn.ReLU()]
            for i in range(num_layers - 1):
                layers.append(torch.nn.Linear(num_classes, num_classes))
                if i != num_layers - 2:
                    layers.append(torch.nn.ReLU())
            self.linear = torch.nn.Sequential(*layers)

        self.dropout = torch.nn.Dropout(self.dropout_amt)

    def forward(self, x):
        """
        x: (batch_size, seq_len, num_variables)
        Returns: (batch_size, num_variables * num_prototypes), (batch_size, num_classes)
        Splits up data by variable and passes through single variable modules, then concatenates similarity vectors.
        Also passes through an FC layer for classification purposes (Single Variable Module Training).
        """
        output = []
        for i in range(self.num_variables):
            single_variable_data = x[:, :, i].unsqueeze(2).float()
            output.append(self.single_variable_prototype_modules[i](single_variable_data))
        output = torch.cat(output, dim=1)
        classification_output = self.linear(output)
        classification_output = self.dropout(classification_output)
        return output, classification_output