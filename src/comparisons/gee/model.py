import torch

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
    
class Decoder(torch.nn.Module):
    """
    Single-variable Decoder.
    """
    def __init__(self, latent_dim, hidden_dim, output_dim, seq_len):
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.lstm1 = torch.nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.lstm2 = torch.nn.LSTM(hidden_dim, 2 * hidden_dim, batch_first=True)
        self.linear = torch.nn.Linear(2 * hidden_dim, output_dim)

    def forward(self, x):
        """
        x: (batch_size, latent_dim)
        Returns: (batch_size, seq_len, output_dim)
        """
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)
        x, (hidden, cell) = self.lstm1(x)
        x, (hidden, cell) = self.lstm2(x)
        x = self.linear(x)
        return x

    
class Autoencoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, seq_len):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len

        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, seq_len)

    def forward(self, x):
        embeddings = self.encoder(x)
        reconstructions = self.decoder(embeddings)
        return embeddings, reconstructions
    
class PrototypeNetwork(torch.nn.Module):
    def __init__(self, num_prototypes, seq_len, latent_dim):
        super(PrototypeNetwork, self).__init__()
        self.num_prototypes = num_prototypes
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        self.prototypes = torch.nn.Parameter(torch.rand(num_prototypes, latent_dim))

    def forward(self, x):
        x = torch.unsqueeze(x, 1).repeat_interleave(self.num_prototypes, 1)
        distances = x - self.prototypes
        distances = torch.norm(distances, dim=2)
        sim = torch.pow(1 + distances, -1)
        return sim

class AutoencoderPrototypeModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_prototypes, seq_len, num_classes, num_layers):
        super(AutoencoderPrototypeModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_prototypes = num_prototypes
        self.seq_len = seq_len
        self.num_classes = num_classes

        self.autoencoder = Autoencoder(input_dim, hidden_dim, latent_dim, seq_len)
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