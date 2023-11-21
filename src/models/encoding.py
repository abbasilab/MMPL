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

        # Efficiently compute distances for all pairs
        dist_matrix = torch.norm(data.unsqueeze(1) - data.unsqueeze(0), dim=2)

        # Create masks for same and opposite labels
        labels = labels.unsqueeze(0)
        same_labels_mask = (labels == labels.T)
        opposite_labels_mask = (labels != labels.T)

        # Apply masks to distance matrix
        same_distances = dist_matrix * same_labels_mask
        opposite_distances = dist_matrix * opposite_labels_mask

        # Compute losses
        same_loss = 0.5 * torch.sum(same_distances.pow(2))
        opposite_loss = 0.5 * torch.sum(torch.max(torch.zeros_like(opposite_distances), self.m - opposite_distances).pow(2))

        # Sum of same and opposite losses
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
    
    
