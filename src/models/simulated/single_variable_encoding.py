import torch
import numpy as np
import matplotlib.pyplot as plt

from ...data.data import get_simulated_ds
from ...data.simulated import DataMiningData
from ..single_variables import SiameseContrastiveLoss, EncodingModule
from ...visualizations.umap_visualizer import UMAPLatent

class LSTMEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden):
        super().__init__()
        self.input_size = input_size
        self.hidden = hidden

        self.lstm = torch.nn.LSTM(input_size=1, hidden_size=hidden, num_layers=3, batch_first=True)
        self.linear = torch.nn.Linear(in_features=hidden, out_features=hidden)

    def forward(self, x):
        x, (hidden_n, cell_n) = self.lstm(x)
        x = x[:, -1, :]
        x = self.linear(x)
        return x

class LSTMDecoder(torch.nn.Module):
    def __init__(self, input_size, hidden):
        super(LSTMDecoder, self).__init__()
        self.input_size = input_size
        self.hidden = hidden

        self.lstm = torch.nn.LSTM(input_size=1, hidden_size=input_size, num_layers=2, batch_first=True)
        self.linear = torch.nn.Linear(in_features=input_size, out_features=input_size)

    def forward(self, x):
        x = x.unsqueeze(2)
        x, (hidden_n, cell_n) = self.lstm(x)
        x = x[:, -1, :]
        x = self.linear(x)
        return x
    
class SingleVariableAutoencoder(torch.nn.Module):
    def __init__(self, input_size, hidden):
        super(SingleVariableAutoencoder, self).__init__()
        self.input_size = input_size
        self.hidden = hidden

        self.encoder = LSTMEncoder(input_size, hidden)
        self.decoder = LSTMDecoder(input_size, hidden)

    def forward(self, x):
        encoded = self.encoder(x)
        # Maybe don't just throw encoder output into decoder, bc it's not necessarily a time series
        decoded = self.decoder(encoded)
        return decoded, encoded

if __name__ == "__main__":
    class_to_index={"Pattern 1":0, "Pattern 2":1, "Pattern 3":2,"Pattern 4":3}

    train_ds = torch.load("data/simulated/train_10.dat")
    test_ds = torch.load("data/simulated/test_10.dat")
    _, class_descriptor = get_simulated_ds(10)

    # Initialize an encoding module for each variable
    encoders = [LSTMEncoder(100, 30) for _  in range(4)]
    encoding_module = EncodingModule(torch.nn.ModuleList(encoders))

    data_load = torch.utils.data.DataLoader(train_ds, len(train_ds), True)
    opt = torch.optim.Adam(params=encoding_module.parameters(), lr=0.01)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.999)
    loss = SiameseContrastiveLoss(m=0.1)
    batch_size = len(train_ds)

    epochs = 500
    for epoch in range(epochs):
        for data_matrix, labels in data_load:
            output = encoding_module(data_matrix.float())

            total_loss = 0
            for i in range(encoding_module.num_variables):
                total_loss += loss(output[i], labels)
            opt.zero_grad()
            total_loss.backward()
            opt.step()
        sched.step()
        print("Epoch: ", epoch, " Total Loss: ", float(total_loss))

    torch.save(encoding_module.state_dict(), "models/simulated/enc.dat")

    encoding_module.load_state_dict(torch.load("models/simulated/enc.dat"))
    visualize_moment = torch.utils.data.DataLoader(test_ds, len(test_ds), True)
    for train_sample in visualize_moment:
        inp, out = train_sample[0].detach(), train_sample[1].detach()
        for i in range(4):
            embeddings = encoding_module.module_list[i](inp[:,:,i].unsqueeze(2).float())
            if i != 3:
                act = []
                for label in out:
                    act.append(class_descriptor[label][1][i])
            else:
                act = torch.zeros(len(out))
            visualizer = UMAPLatent()
            visualizer.visualize(embeddings, torch.FloatTensor(act), 4)
    plt.show()
    
        