import torch

class Encoder(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, num_layers):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        self.lstm = torch.nn.LSTM(input_dim, latent_dim, num_layers, batch_first=True)

    def forward(self, x):
        out = self.lstm(x)
        return out
    
class Decoder(torch.nn.Module):
    def __init__(self, latent_dim, output_dim, num_layers):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.lstm = torch.nn.LSTM(latent_dim, output_dim, num_layers, batch_first=True)

    def forward(self, x):
        out = self.lstm(x)
        return out
    
class Autoencoder(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, num_layers):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        self.encoder = Encoder(input_dim, latent_dim, num_layers)
        self.decoder = Decoder(latent_dim, input_dim, num_layers)

    def forward(self, x):
        embeddings, _ = self.encoder(x)
        reconstructions, _ = self.decoder(embeddings)
        return embeddings, reconstructions
    
class PrototypeNetwork(torch.nn.Module):
    def __init__(self, num_prototypes, seq_len, latent_dim):
        super(PrototypeNetwork, self).__init__()
        self.num_prototypes = num_prototypes
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        self.prototypes = torch.nn.Parameter(torch.zeros(num_prototypes, seq_len, latent_dim))

    def forward(self, x):
        x_exp = x.unsqueeze(1)
        prototypes_exp = self.prototypes.unsqueeze(0)

        l2_norm = torch.norm(x_exp - prototypes_exp, p=2, dim=[2, 3])
        return l2_norm

class AutoencoderPrototypeModel(torch.nn.Module):
    def __init__(self, input_dim, latent_dim, autoencoder_num_layers, num_prototypes, seq_len, num_classes, num_layers):
        super(AutoencoderPrototypeModel, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.autoencoder_num_layers = autoencoder_num_layers
        self.num_prototypes = num_prototypes
        self.seq_len = seq_len
        self.num_classes = num_classes

        self.autoencoder = Autoencoder(input_dim, latent_dim, autoencoder_num_layers)
        self.prototype_network = PrototypeNetwork(num_prototypes, seq_len, latent_dim)

        if num_layers == 1:
            self.linear = torch.nn.Linear(num_classes, num_classes)
        else:
            layers = [torch.nn.Linear(num_classes, num_classes), torch.nn.ReLU()]
            for i in range(num_layers - 1):
                layers.append(torch.nn.Linear(num_classes, num_classes))
                if i != num_layers - 2:
                    layers.append(torch.nn.ReLU())
            self.linear = torch.nn.Sequential(*layers)

    def forward(self, x):
        embeddings, reconstructions = self.autoencoder(x)
        similarities = self.prototype_network(embeddings)
        predictions = self.linear(similarities)
        return predictions, reconstructions, embeddings