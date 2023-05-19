import torch
import random
import matplotlib.pyplot as plt
import numpy as np
from ...data.data import get_ds, filter_classes
from ..single_variables import SingleVariableModulesWrapper,EncodingModule, SiameseContrastiveLoss
from ...visualizations.umap_visualizer import UMAPLatent

class LSTMEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden):
        super(LSTMEncoder, self).__init__()
        self.input_size = input_size
        self.hidden = hidden

        self.lstm = torch.nn.LSTM(input_size=1, hidden_size=hidden, num_layers=1, batch_first=True)
        self.lstm1 = torch.nn.LSTM(input_size=1, hidden_size=2*hidden, num_layers=1, batch_first=True)
        self.lstm2 = torch.nn.LSTM(input_size=1, hidden_size=hidden, num_layers=1, batch_first=True)
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

        self.lstm = torch.nn.LSTM(input_size=1, hidden_size=input_size, num_layers=1, batch_first=True)
        self.lstm1 = torch.nn.LSTM(input_size=1, hidden_size=2*hidden, num_layers=1, batch_first=True)
        self.lstm2 = torch.nn.LSTM(input_size=1, hidden_size=input_size, num_layers=1, batch_first=True)
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
        decoded = self.decoder(encoded)
        return decoded, encoded
    
class EncodingModule(torch.nn.Module):
    def __init__(self, module_list):
        super().__init__()
        self.module_list = module_list
        self.num_variables = len(module_list)
    
    def forward(self, data):
        encoder_results = []
        decoder_results = []
        for i in range(self.num_variables):
            data_subset = data[:, :, i]
            decoded, encoded = self.module_list[i](data_subset.unsqueeze(2))
            encoder_results.append(encoded)
            decoder_results.append(decoded)
        return decoder_results, encoder_results 

if __name__ == "__main__":
    class_to_index = {
        "1":1, "2":2, "3":3, "4":4, "5":5, "6":6, "7":7, "8":8,
        "9":9, "10":10, "11":11, "12":12, "13":13, "14":14, "15":15,
        "16":16, "17":17, "18":18, "19":19, "20":20
    }
    
    train_ds, test_ds = get_ds("data/charactertrajectories/CharacterTrajectoriesEq_TRAIN.ts", class_to_index), get_ds("data/charactertrajectories/CharacterTrajectoriesEq_TEST.ts", class_to_index)
    filtered_train, filtered_test = filter_classes(train_ds, [2, 4, 12, 13]), filter_classes(test_ds, [2, 4, 12, 13])
    autoencoders = []
    for i in range(3):
        autoencoders.append(SingleVariableAutoencoder(119, 10))
    encoding_module = EncodingModule(torch.nn.ModuleList(autoencoders))

    data_load = torch.utils.data.DataLoader(filtered_train, len(filtered_train), True)
    opt = torch.optim.Adam(params=encoding_module.parameters(), lr=0.01)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.999)
    contrastive_loss_fn = SiameseContrastiveLoss(m=1.0)
    reconstruction_loss_fn = torch.nn.MSELoss()
    batch_size = len(filtered_train)

    epochs = 2000
    for epoch in range(epochs):
        for data_matrix, labels in data_load:
            opt.zero_grad()
            indices = torch.randperm(len(data_matrix))[:batch_size]
            decoder_output, encoder_output = encoding_module(data_matrix[indices].float())
            contrastive_loss = 0
            reconstruction_loss = 0
            for i in range(encoding_module.num_variables):
                contrastive_loss += contrastive_loss_fn(encoder_output[i], labels[indices])
                reconstruction_loss += reconstruction_loss_fn(decoder_output[i], data_matrix[indices][:, :, i].float())
            total_loss = (0.0005)*contrastive_loss + (1.0)*reconstruction_loss
            total_loss.backward()
            opt.step()
        sched.step()
        print("Epoch: ", epoch, " Total Loss: ", float(total_loss),
               "Contrastive Loss: ", float(contrastive_loss), "Reconstruction Loss: ", float(reconstruction_loss))

    torch.save(encoding_module.state_dict(), "models/charactertrajectories_filtered/autoenc.dat")

    encoding_module.load_state_dict(torch.load("models/charactertrajectories_filtered/autoenc.dat"))
    data_train = torch.utils.data.DataLoader(filtered_train, len(filtered_train), True)
    data_test = torch.utils.data.DataLoader(filtered_test, len(filtered_test), True)
    with torch.no_grad():
        for train_sample in data_test:
            inp, out = train_sample[0].detach(), train_sample[1].detach()
            for i in range(encoding_module.num_variables):
                embeddings = encoding_module.module_list[i].encoder(inp[:,:,i].unsqueeze(2).float())
                visualizer = UMAPLatent()
                visualizer.visualize(embeddings, out, len(class_to_index))
    plt.show()

    encoding_module.load_state_dict(torch.load("models/charactertrajectories_filtered/autoenc.dat"))

    with torch.no_grad():
        for data_matrix, labels in data_test:
            index = random.randint(0, len(data_matrix) - 1)
            print(labels[index])
            orig = data_matrix[index]
            
            decoded_output, _ = encoding_module(data_matrix.float())

            fig, axes = plt.subplots(1, 3, figsize=(12,4))
            for i in range(3):
                ax = axes[i]
                decoded_point = decoded_output[i][index]
                ax.plot(orig[:, i], c='b', lw=0.5)
                ax.plot(decoded_point, c='r', lw=0.5)

            plt.show()



