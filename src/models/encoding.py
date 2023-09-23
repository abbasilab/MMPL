import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ContrastiveLoss(torch.nn.Module):
    """
    Loss Function used for single-variable encoding
    """
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, data, labels):
        """
        data: (batch_size, seq_len, 1)
        labels: (batch_size,)
        """
        batch_size = data.shape[0]
        rangeset = torch.arange(batch_size).to(device)
        all_combos = torch.combinations(rangeset).to(device)
        same_labels = all_combos[(labels[all_combos[:, 0]] == labels[all_combos[:, 1]]).nonzero()].squeeze()
        opposite_labels = all_combos[(labels[all_combos[:, 0]] != labels[all_combos[:, 1]]).nonzero()].squeeze()
        same_distances = torch.linalg.norm(data[same_labels][:, 0] - data[same_labels][:, 1], dim=1)
        opposite_distances = torch.linalg.norm(data[opposite_labels][:, 0] - data[opposite_labels][:, 1], dim=1)
        same_loss = 0.5*torch.sum(same_distances.pow(2))
        opposite_loss = 0.5*torch.sum(torch.max(torch.tensor(0), self.m - opposite_distances).pow(2))
        final = same_loss + opposite_loss
        return final
    
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
    
    
