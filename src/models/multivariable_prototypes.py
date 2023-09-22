import torch

class MultivariableModule(torch.nn.Module):
    """
    Module that holds the multivariable prototypes
    """
    def __init__(self, wrapper, num_classes, num_variables, num_sv_prototypes, num_layers):
        super(MultivariableModule, self).__init__()
        self.wrapper = wrapper
        self.num_classes = num_classes
        self.num_variables = num_variables
        self.num_sv_prototypes = num_sv_prototypes
        self.num_layers = num_layers

        # One prototype per class
        self.prototypes = torch.nn.Parameter(torch.rand(num_classes, num_sv_prototypes*num_variables))

        if num_layers == 1:
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