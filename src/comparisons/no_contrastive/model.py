import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
class Encoder(torch.nn.Module):
    """
    Single-variable Encoder.
    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.lstm1 = torch.nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.lstm2 = torch.nn.LSTM(2 * hidden_dim, latent_dim, batch_first=True)
        
    def forward(self, x):
        """
        x: (batch_size, seq_len, 1)
        Returns: (batch_size, latent_dim)
        """
        x, (hidden, cell) = self.lstm1(x)
        x, (hidden, cell) = self.lstm2(x)
        return hidden[-1, :, :]
    
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
    
class SingleVariablePrototypesWrapper2(torch.nn.Module):
    """
    Wrapper class that holds <num_variables> SingleVariablePrototypesModule classes
    """
    def __init__(self, num_variables, num_classes, num_prototypes, hidden_dim, latent_dim, num_layers):
        super().__init__()
        self.num_variables = num_variables
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        single_variable_prototype_modules = []
        for i in range(num_variables):
            encoder = Encoder(1, hidden_dim, latent_dim)
            single_variable_prototype_modules.append(
                SingleVariablePrototypesModule(
                    encoder=encoder,
                    num_prototypes=num_prototypes,
                    latent_dim=latent_dim
                )
            )
        self.single_variable_prototype_modules = torch.nn.ModuleList(single_variable_prototype_modules)

        self.linear = torch.nn.Linear(num_variables * num_prototypes, num_classes)

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
        return output, classification_output
    
class MultivariableModule2(torch.nn.Module):
    """
    Module that holds the multivariable prototypes
    """
    def __init__(self, wrapper, num_classes, num_variables, num_sv_prototypes, num_layers):
        super(MultivariableModule2, self).__init__()
        self.wrapper = wrapper
        self.num_classes = num_classes
        self.num_variables = num_variables
        self.num_sv_prototypes = num_sv_prototypes
        self.num_layers = num_layers

        # One prototype per class
        self.prototypes = torch.nn.Parameter(torch.rand(num_classes, num_sv_prototypes*num_variables))

        if num_layers == 0:
            self.linear = torch.nn.Identity()
        elif num_layers == 1:
            self.linear = torch.nn.Linear(num_classes, num_classes, bias=False)
        else:
            layers = [torch.nn.Linear(num_classes, num_classes, bias=False) for _ in range(num_layers)]
            self.linear = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        x: (batch_size, seq_len, num_variables)
        Returns: (batch_size, num_classes)
        Passes x through full model.
        """
        sv_sims, _ = self.wrapper(x)
        x = torch.unsqueeze(sv_sims, 1).repeat_interleave(self.num_classes, 1)
        distances = x - self.prototypes
        distances = torch.norm(distances, dim=2)
        sim = torch.pow(1 + distances, -1)
        return self.linear(sim), sv_sims
    